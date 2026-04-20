from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
LOG_DIR = BASE_DIR / "data" / "agent_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


ANALYST_PROMPT_PATH = PROMPTS_DIR / "agent_analyst_v1.txt"
DECIDER_PROMPT_PATH = PROMPTS_DIR / "agent_decider_v1.txt"


DEFAULT_ANALYST_FALLBACK = {
    "summary": "Aucune réponse IA disponible. Utiliser le scoring et les signaux visibles pour l’analyse.",
    "signals": [],
    "comparison_to_base": "Non calculé via IA.",
    "confidence": 0.0,
}

DEFAULT_DECIDER_FALLBACK = {
    "recommendation": "watch",
    "confidence": 0.0,
    "justification": "Aucune réponse IA disponible. Conserver la décision opérationnelle minimale.",
}


@dataclass
class AgentResult:
    role: str
    model: str
    prompt_version: str
    success: bool
    latency_ms: int
    content: dict[str, Any]
    raw_text: str = ""
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt introuvable : {path}")
    return path.read_text(encoding="utf-8").strip()


def _log_event(payload: dict[str, Any]) -> None:
    log_path = LOG_DIR / "agent_calls.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_json_loads(text: str) -> dict[str, Any]:
    """
    Essaye d'extraire un JSON à partir de la réponse.
    Si le modèle renvoie du texte parasite, on tente de récupérer le premier objet JSON.
    """
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    return {}


def _default_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def build_subscriber_context(row: pd.Series | dict[str, Any], base_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Construit un contexte compact, lisible par le modèle, à partir d'une ligne de features.
    """
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    ctx = {
        "user_id": data.get("user_id"),
        "email": data.get("email"),
        "country": data.get("country"),
        "risk_score": data.get("risk_score"),
        "risk_tier": data.get("risk_tier"),
        "status": data.get("status"),
        "status_is_anomalous": data.get("status_is_anomalous"),
        "is_new_user": data.get("is_new_user"),
        "inactive_6m_flag": data.get("inactive_6m_flag"),
        "has_history_flag": data.get("has_history_flag"),
        "payment_total_count": data.get("payment_total_count"),
        "payment_failed_count": data.get("payment_failed_count"),
        "payment_disputed_count": data.get("payment_disputed_count"),
        "payment_refunded_count": data.get("payment_refunded_count"),
        "payment_pending_count": data.get("payment_pending_count"),
        "payment_canceled_count": data.get("payment_canceled_count"),
        "payment_issue_rate": data.get("payment_issue_rate"),
        "membership_total_count": data.get("membership_total_count"),
        "membership_exit_count": data.get("membership_exit_count"),
        "membership_current_count": data.get("membership_current_count"),
        "short_membership_count": data.get("short_membership_count"),
        "membership_churn_rate": data.get("membership_churn_rate"),
        "brand_switch_rate": data.get("brand_switch_rate"),
        "complaint_total_count": data.get("complaint_total_count"),
        "complaint_open_count": data.get("complaint_open_count"),
        "complaint_escalated_count": data.get("complaint_escalated_count"),
        "complaint_resolved_count": data.get("complaint_resolved_count"),
        "complaint_total_severity": data.get("complaint_total_severity"),
        "days_since_last_payment": data.get("days_since_last_payment"),
        "days_since_last_complaint": data.get("days_since_last_complaint"),
        "days_since_last_membership_end": data.get("days_since_last_membership_end"),
        "days_since_last_seen": data.get("days_since_last_seen"),
    }

    if base_context:
        ctx.update(base_context)

    return ctx


def _build_messages(prompt: str, context: dict[str, Any]) -> list[dict[str, str]]:
    user_content = {
        "instructions": "Réponds uniquement en JSON valide.",
        "context": context,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def _call_openai(
    *,
    model: str,
    prompt: str,
    context: dict[str, Any],
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    max_output_tokens: int = 700,
) -> tuple[dict[str, Any], str]:
    """
    Appelle l'API OpenAI via Responses API et renvoie (json_parse, texte_brut).
    """
    client = _default_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY manquant ou SDK OpenAI indisponible.")

    resp = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=_build_messages(prompt, context),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    raw_text = getattr(resp, "output_text", "") or ""
    parsed = _safe_json_loads(raw_text)
    return parsed, raw_text


def _heuristic_analyst(context: dict[str, Any]) -> dict[str, Any]:
    signals = []
    if (context.get("payment_failed_count") or 0) > 0:
        signals.append("Échecs de paiement")
    if (context.get("payment_disputed_count") or 0) > 0:
        signals.append("Litiges paiements")
    if (context.get("complaint_open_count") or 0) > 0 or (context.get("complaint_escalated_count") or 0) > 0:
        signals.append("Réclamations ouvertes/escaladées")
    if (context.get("membership_churn_rate") or 0) > 0.2:
        signals.append("Churn memberships élevé")
    if (context.get("inactive_6m_flag") or 0) == 1:
        signals.append("Inactif depuis plus de 6 mois")

    summary = (
        f"Subscriber {context.get('user_id')} présente "
        f"{context.get('payment_total_count', 0)} paiements, "
        f"{context.get('complaint_total_count', 0)} réclamations et "
        f"un score de risque de {context.get('risk_score', 0)}."
    )

    return {
        "summary": summary,
        "signals": signals,
        "comparison_to_base": "Heuristique locale : priorité croissante si paiements dégradés, réclamations actives et churn élevé.",
        "confidence": 0.35,
    }


def _heuristic_decider(context: dict[str, Any]) -> dict[str, Any]:
    score = float(context.get("risk_score") or 0.0)
    payments = float(context.get("payment_issue_rate") or 0.0)
    complaints = float(context.get("complaint_total_count") or 0.0)
    churn = float(context.get("membership_churn_rate") or 0.0)
    inactive = int(context.get("inactive_6m_flag") or 0)

    if score >= 75 or (payments >= 0.5 and complaints >= 2):
        recommendation = "block"
        confidence = 0.82
        justification = "Score très élevé avec signaux incidents forts."
    elif score >= 45 or churn >= 0.3 or inactive == 1:
        recommendation = "watch"
        confidence = 0.68
        justification = "Risque intermédiaire nécessitant une surveillance."
    else:
        recommendation = "ignore"
        confidence = 0.55
        justification = "Risque limité au vu des signaux visibles."

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "justification": justification,
    }


def analyze_subscriber(
    row: pd.Series | dict[str, Any],
    *,
    model: str = "gpt-5.4-mini",
    fallback_model: str = "gpt-5.4",
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Produit un résumé structuré du cas + une recommandation d'action.
    En cas d'échec API, bascule sur un fallback heuristique.
    """
    context = build_subscriber_context(row, base_context=base_context)

    analyst_prompt = _read_prompt(ANALYST_PROMPT_PATH)
    decider_prompt = _read_prompt(DECIDER_PROMPT_PATH)

    results: dict[str, Any] = {"context": context}

    # Analyste
    start = time.perf_counter()
    try:
        parsed, raw = _call_openai(
            model=model,
            prompt=analyst_prompt,
            context=context,
            reasoning_effort="medium",
            max_output_tokens=600,
        )
        analyst = parsed if parsed else _heuristic_analyst(context)
        success = bool(parsed)
        error = None
    except Exception as exc:
        analyst = _heuristic_analyst(context)
        raw = ""
        success = False
        error = str(exc)

        _log_event(
            {
                "timestamp": _now_iso(),
                "role": "analyst",
                "model": model,
                "fallback_model": fallback_model,
                "success": False,
                "error": error,
                "user_id": context.get("user_id"),
                "prompt_version": "agent_analyst_v1",
            }
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    _log_event(
        {
            "timestamp": _now_iso(),
            "role": "analyst",
            "model": model,
            "success": success,
            "latency_ms": latency_ms,
            "user_id": context.get("user_id"),
            "prompt_version": "agent_analyst_v1",
            "raw_text": raw[:2000],
            "parsed": analyst,
        }
    )

    results["analyst"] = AgentResult(
        role="analyst",
        model=model,
        prompt_version="agent_analyst_v1",
        success=success,
        latency_ms=latency_ms,
        content=analyst,
        raw_text=raw,
        error=error,
    )

    # Décideur
    start = time.perf_counter()
    try:
        parsed, raw = _call_openai(
            model=model,
            prompt=decider_prompt,
            context=context,
            reasoning_effort="medium",
            max_output_tokens=450,
        )
        decider = parsed if parsed else _heuristic_decider(context)
        success = bool(parsed)
        error = None
    except Exception as exc:
        decider = _heuristic_decider(context)
        raw = ""
        success = False
        error = str(exc)

        _log_event(
            {
                "timestamp": _now_iso(),
                "role": "decider",
                "model": model,
                "fallback_model": fallback_model,
                "success": False,
                "error": error,
                "user_id": context.get("user_id"),
                "prompt_version": "agent_decider_v1",
            }
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    _log_event(
        {
            "timestamp": _now_iso(),
            "role": "decider",
            "model": model,
            "success": success,
            "latency_ms": latency_ms,
            "user_id": context.get("user_id"),
            "prompt_version": "agent_decider_v1",
            "raw_text": raw[:2000],
            "parsed": decider,
        }
    )

    results["decider"] = AgentResult(
        role="decider",
        model=model,
        prompt_version="agent_decider_v1",
        success=success,
        latency_ms=latency_ms,
        content=decider,
        raw_text=raw,
        error=error,
    )

    return results


def analyze_subscriber_as_dict(row: pd.Series | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """
    Version pratique pour Streamlit : renvoie un dictionnaire sérialisable.
    """
    result = analyze_subscriber(row, **kwargs)
    return {
        "context": result["context"],
        "analyst": {
            "role": result["analyst"].role,
            "model": result["analyst"].model,
            "prompt_version": result["analyst"].prompt_version,
            "success": result["analyst"].success,
            "latency_ms": result["analyst"].latency_ms,
            "content": result["analyst"].content,
            "error": result["analyst"].error,
        },
        "decider": {
            "role": result["decider"].role,
            "model": result["decider"].model,
            "prompt_version": result["decider"].prompt_version,
            "success": result["decider"].success,
            "latency_ms": result["decider"].latency_ms,
            "content": result["decider"].content,
            "error": result["decider"].error,
        },
    }


def main() -> None:
    """
    CLI minimale de test.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Tester l'agent IA sur un subscriber.")
    parser.add_argument("--input_csv", type=Path, required=True, help="CSV scoré d'entrée.")
    parser.add_argument("--user_id", type=int, required=True, help="ID du subscriber à analyser.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    row = df.loc[df["user_id"] == args.user_id]
    if row.empty:
        raise ValueError(f"user_id={args.user_id} introuvable dans {args.input_csv}")

    payload = analyze_subscriber_as_dict(row.iloc[0])
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()