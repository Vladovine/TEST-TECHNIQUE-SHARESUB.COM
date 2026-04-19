"""Risk scoring utilities and CLI for the Sharesub risk monitor project.

This module loads the cleaned SQLite database, computes user-level features,
builds a deterministic risk score, and exports the scored CSV.

Usage:
    python -m src.scoring --input data/processed/risk_monitor_clean.sqlite \
                          --output outputs/subscribers_risk_scored.csv
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


PAYMENT_STATUS_WEIGHTS: Dict[str, float] = {
    "succeeded": 0.0,
    "failed": 1.0,
    "disputed": 1.5,
    "refunded": 0.8,
    "pending": 0.3,
    "canceled": 0.2,
}

COMPLAINT_TYPE_WEIGHTS: Dict[str, float] = {
    "access_denied": 1.2,
    "fraud_suspicion": 1.5,
    "owner_unresponsive": 1.1,
    "subscription_inactive": 1.0,
    "billing_issue": 0.8,
    "wrong_credentials": 0.7,
    "other": 0.5,
}

COMPLAINT_STATUS_WEIGHTS: Dict[str, float] = {
    "open": 1.0,
    "in_progress": 0.9,
    "escalated": 1.3,
    "resolved": 0.5,
    "closed": 0.4,
}


# -----------------------------
# Helpers
# -----------------------------
def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def robust_minmax(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (num / den).fillna(0.0)


def days_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    delta = later - earlier
    return delta.dt.total_seconds().div(86400).fillna(0.0)


def clip_days(series: pd.Series, max_days: float = 365.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return s.clip(lower=0.0, upper=max_days)


def canonical_complaint_type(value: str) -> str:
    if pd.isna(value):
        return "other"
    s = str(value).strip().lower()
    mapping = {
        "acces refuse": "access_denied",
        "access denied": "access_denied",
        "billing issue": "billing_issue",
        "billing_issue": "billing_issue",
        "fraud suspicion": "fraud_suspicion",
        "fraud_suspicion": "fraud_suspicion",
        "owner unresponsive": "owner_unresponsive",
        "owner_unresponsive": "owner_unresponsive",
        "subscription inactive": "subscription_inactive",
        "subscription_inactive": "subscription_inactive",
        "wrong credentials": "wrong_credentials",
        "wrong_credentials": "wrong_credentials",
        "other": "other",
    }
    return mapping.get(s, s)


def prepare_export_df(df: pd.DataFrame, cols: list[str], datetime_cols: Iterable[str] | None = None) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    datetime_cols = list(datetime_cols or [])
    for col in datetime_cols:
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


# -----------------------------
# IO
# -----------------------------
def load_clean_data(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {db_path}")

    with sqlite3.connect(db_path) as conn:
        users = pd.read_sql_query("SELECT * FROM users;", conn)
        subscriptions = pd.read_sql_query("SELECT * FROM subscriptions;", conn)
        memberships = pd.read_sql_query("SELECT * FROM memberships;", conn)
        payments = pd.read_sql_query("SELECT * FROM payments;", conn)
        complaints = pd.read_sql_query("SELECT * FROM complaints;", conn)

    return users, subscriptions, memberships, payments, complaints


# -----------------------------
# Reference date
# -----------------------------
def compute_reference_date(
    users: pd.DataFrame,
    subscriptions: pd.DataFrame,
    memberships: pd.DataFrame,
    payments: pd.DataFrame,
    complaints: pd.DataFrame,
) -> pd.Timestamp:
    # Exclure signup_at_utc du calcul de référence car certaines valeurs sont futures/anormales.
    date_series = pd.concat(
        [
            pd.to_datetime(users["last_seen_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(subscriptions["created_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(memberships["joined_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(memberships["left_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(payments["created_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(payments["captured_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(complaints["created_at_utc"], utc=True, errors="coerce"),
            pd.to_datetime(complaints["resolved_at_utc"], utc=True, errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()

    if date_series.empty:
        raise ValueError("Impossible de calculer la date de référence : aucune date opérationnelle valide n'a été trouvée.")

    return date_series.max().normalize() + pd.Timedelta(days=1)


# -----------------------------
# Feature engineering
# -----------------------------
def build_payment_features(
    payments: pd.DataFrame,
    subscriptions: pd.DataFrame,
    reference_date: pd.Timestamp,
    valid_user_ids: set[int],
) -> pd.DataFrame:
    p = payments.copy()
    p["status_key"] = p["status"].astype(str).str.lower()

    for st in PAYMENT_STATUS_WEIGHTS.keys():
        p[f"is_{st}"] = (p["status_key"] == st).astype(int)

    payment_last = p.groupby("user_id", as_index=False).agg(
        payment_last_at=("created_at_utc", "max")
    )

    payment_agg = p.groupby("user_id", as_index=False).agg(
        payment_total_count=("id", "count"),
        payment_success_count=("is_succeeded", "sum"),
        payment_failed_count=("is_failed", "sum"),
        payment_disputed_count=("is_disputed", "sum"),
        payment_refunded_count=("is_refunded", "sum"),
        payment_pending_count=("is_pending", "sum"),
        payment_canceled_count=("is_canceled", "sum"),
        payment_total_amount_cents=("amount_cents", "sum"),
        payment_avg_amount_cents=("amount_cents", "mean"),
        payment_avg_fee_cents=("fee_cents", "mean"),
        payment_distinct_subscription_count=("subscription_id", "nunique"),
    )

    payments_with_brand = p.merge(
        subscriptions[["id", "brand"]],
        left_on="subscription_id",
        right_on="id",
        how="left",
        suffixes=("", "_sub"),
    )

    brand_agg = payments_with_brand.groupby("user_id", as_index=False).agg(
        payment_distinct_brand_count=("brand", "nunique")
    )

    payment_agg = payment_agg.merge(brand_agg, on="user_id", how="left")
    payment_agg = payment_agg.merge(payment_last, on="user_id", how="left")
    payment_agg = payment_agg[payment_agg["user_id"].isin(valid_user_ids)].copy()

    payment_agg["payment_issue_weight_sum"] = (
        1.0 * payment_agg["payment_failed_count"]
        + 1.5 * payment_agg["payment_disputed_count"]
        + 0.8 * payment_agg["payment_refunded_count"]
        + 0.3 * payment_agg["payment_pending_count"]
        + 0.2 * payment_agg["payment_canceled_count"]
    )

    payment_agg["payment_issue_rate"] = safe_div(
        payment_agg["payment_issue_weight_sum"],
        payment_agg["payment_total_count"],
    )

    payment_agg["days_since_last_payment_raw"] = days_between(
        pd.Series(reference_date, index=payment_agg.index),
        payment_agg["payment_last_at"],
    )

    payment_agg["days_since_last_payment"] = clip_days(payment_agg["days_since_last_payment_raw"], 1095)

    payment_agg["payment_recent_event_risk_raw"] = np.where(
        payment_agg["payment_total_count"] > 0,
        1 / (1 + payment_agg["days_since_last_payment"] / 30.0),
        0.0,
    )

    payment_agg["payment_recent_event_risk_raw"] = payment_agg["payment_recent_event_risk_raw"].clip(0, 1)
    return payment_agg


def build_membership_features(
    memberships: pd.DataFrame,
    subscriptions: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    m = memberships.copy()
    m = m.merge(
        subscriptions[["id", "brand"]],
        left_on="subscription_id",
        right_on="id",
        how="left",
        suffixes=("", "_sub"),
    )

    m["joined_at_utc"] = pd.to_datetime(m["joined_at_utc"], utc=True, errors="coerce")
    m["left_at_utc"] = pd.to_datetime(m["left_at_utc"], utc=True, errors="coerce")

    m["membership_end_at"] = m["left_at_utc"].fillna(reference_date)
    m["membership_duration_days"] = days_between(m["membership_end_at"], m["joined_at_utc"]).clip(lower=0)

    m["is_exit"] = m["left_at_utc"].notna().astype(int)
    m["is_current"] = m["left_at_utc"].isna().astype(int)
    m["is_short_membership"] = (m["membership_duration_days"] < 30).astype(int)

    membership_agg = m.groupby("user_id", as_index=False).agg(
        membership_total_count=("id", "count"),
        membership_exit_count=("is_exit", "sum"),
        membership_current_count=("is_current", "sum"),
        short_membership_count=("is_short_membership", "sum"),
        avg_membership_duration_days=("membership_duration_days", "mean"),
        median_membership_duration_days=("membership_duration_days", "median"),
        distinct_subscription_count=("subscription_id", "nunique"),
        distinct_brand_count=("brand", "nunique"),
        last_membership_end_at=("membership_end_at", "max"),
    )

    membership_agg["membership_churn_rate"] = safe_div(
        membership_agg["membership_exit_count"],
        membership_agg["membership_total_count"],
    )

    membership_agg["brand_switch_rate"] = safe_div(
        (membership_agg["distinct_brand_count"] - 1).clip(lower=0),
        membership_agg["membership_total_count"],
    )

    membership_agg["days_since_last_membership_end_raw"] = days_between(
        pd.Series(reference_date, index=membership_agg.index),
        membership_agg["last_membership_end_at"],
    )

    membership_agg["days_since_last_membership_end"] = clip_days(
        membership_agg["days_since_last_membership_end_raw"], 1095
    )

    membership_agg["membership_recent_event_risk_raw"] = np.where(
        membership_agg["membership_total_count"] > 0,
        1 / (1 + membership_agg["days_since_last_membership_end"] / 30.0),
        0.0,
    )
    membership_agg["membership_recent_event_risk_raw"] = membership_agg["membership_recent_event_risk_raw"].clip(0, 1)
    return membership_agg


def build_complaint_features(
    complaints: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    c = complaints.copy()

    c["created_at_utc"] = pd.to_datetime(c["created_at_utc"], utc=True, errors="coerce")
    c["resolved_at_utc"] = pd.to_datetime(c["resolved_at_utc"], utc=True, errors="coerce")
    c["type_key"] = c["type"].apply(canonical_complaint_type)
    c["status_key"] = c["status"].astype(str).str.lower()

    c["complaint_type_weight"] = c["type_key"].map(COMPLAINT_TYPE_WEIGHTS).fillna(0.5)
    c["complaint_status_weight"] = c["status_key"].map(COMPLAINT_STATUS_WEIGHTS).fillna(0.5)
    c["complaint_severity"] = c["complaint_type_weight"] * c["complaint_status_weight"]

    c["is_open"] = c["status_key"].isin(["open", "in_progress"]).astype(int)
    c["is_escalated"] = (c["status_key"] == "escalated").astype(int)
    c["is_resolved"] = (c["status_key"] == "resolved").astype(int)

    reporter_agg = c.groupby("reporter_id", as_index=False).agg(
        complaint_reporter_count=("id", "count"),
        complaint_reporter_open_count=("is_open", "sum"),
        complaint_reporter_escalated_count=("is_escalated", "sum"),
        complaint_reporter_resolved_count=("is_resolved", "sum"),
        complaint_reporter_severity_sum=("complaint_severity", "sum"),
        complaint_reporter_last_at=("created_at_utc", "max"),
    ).rename(columns={"reporter_id": "user_id"})

    target_agg = c.groupby("target_id", as_index=False).agg(
        complaint_target_count=("id", "count"),
        complaint_target_open_count=("is_open", "sum"),
        complaint_target_escalated_count=("is_escalated", "sum"),
        complaint_target_resolved_count=("is_resolved", "sum"),
        complaint_target_severity_sum=("complaint_severity", "sum"),
        complaint_target_last_at=("created_at_utc", "max"),
    ).rename(columns={"target_id": "user_id"})

    complaint_agg = reporter_agg.merge(target_agg, on="user_id", how="outer")

    complaint_agg["complaint_reporter_last_at"] = pd.to_datetime(
        complaint_agg["complaint_reporter_last_at"], utc=True, errors="coerce"
    )
    complaint_agg["complaint_target_last_at"] = pd.to_datetime(
        complaint_agg["complaint_target_last_at"], utc=True, errors="coerce"
    )

    numeric_cols = complaint_agg.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        complaint_agg[col] = complaint_agg[col].fillna(0)

    complaint_agg["complaint_total_count"] = (
        complaint_agg["complaint_reporter_count"] + complaint_agg["complaint_target_count"]
    )
    complaint_agg["complaint_open_count"] = (
        complaint_agg["complaint_reporter_open_count"] + complaint_agg["complaint_target_open_count"]
    )
    complaint_agg["complaint_escalated_count"] = (
        complaint_agg["complaint_reporter_escalated_count"] + complaint_agg["complaint_target_escalated_count"]
    )
    complaint_agg["complaint_resolved_count"] = (
        complaint_agg["complaint_reporter_resolved_count"] + complaint_agg["complaint_target_resolved_count"]
    )
    complaint_agg["complaint_total_severity"] = (
        0.8 * complaint_agg["complaint_reporter_severity_sum"]
        + 1.2 * complaint_agg["complaint_target_severity_sum"]
    )

    complaint_agg["last_complaint_at"] = complaint_agg[
        ["complaint_reporter_last_at", "complaint_target_last_at"]
    ].max(axis=1)

    complaint_agg["days_since_last_complaint_raw"] = days_between(
        pd.Series(reference_date, index=complaint_agg.index),
        complaint_agg["last_complaint_at"],
    )

    complaint_agg["days_since_last_complaint"] = clip_days(
        complaint_agg["days_since_last_complaint_raw"], 1095
    )

    complaint_agg["complaint_recent_event_risk_raw"] = np.where(
        complaint_agg["complaint_total_count"] > 0,
        1 / (1 + complaint_agg["days_since_last_complaint"] / 30.0),
        0.0,
    )
    complaint_agg["complaint_recent_event_risk_raw"] = complaint_agg["complaint_recent_event_risk_raw"].clip(0, 1)
    return complaint_agg


def merge_features(
    users: pd.DataFrame,
    payment_agg: pd.DataFrame,
    membership_agg: pd.DataFrame,
    complaint_agg: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    features = users[[
        "id",
        "email",
        "country",
        "signup_at_utc",
        "last_seen_at_utc",
        "status",
        "status_is_anomalous",
    ]].copy().rename(columns={"id": "user_id"})

    features["signup_at_utc"] = pd.to_datetime(features["signup_at_utc"], utc=True, errors="coerce")
    features["last_seen_at_utc"] = pd.to_datetime(features["last_seen_at_utc"], utc=True, errors="coerce")

    for df in [payment_agg, membership_agg, complaint_agg]:
        features = features.merge(df, on="user_id", how="left")

    for col in [
        "payment_last_at",
        "last_membership_end_at",
        "last_complaint_at",
        "complaint_reporter_last_at",
        "complaint_target_last_at",
    ]:
        if col in features.columns:
            features[col] = pd.to_datetime(features[col], utc=True, errors="coerce")

    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        features[col] = features[col].fillna(0)

    features["account_age_days"] = days_between(
        pd.Series(reference_date, index=features.index),
        features["signup_at_utc"],
    )
    features["account_age_days"] = clip_days(features["account_age_days"], 3650)

    last_seen_ref = features["last_seen_at_utc"].fillna(features["signup_at_utc"]).fillna(reference_date)
    features["days_since_last_seen"] = days_between(
        pd.Series(reference_date, index=features.index),
        last_seen_ref,
    )
    features["days_since_last_seen"] = clip_days(features["days_since_last_seen"], 365)

    features["is_new_user"] = (features["account_age_days"] < 30).astype(int)
    features["inactive_6m_flag"] = (features["days_since_last_seen"] >= 180).astype(int)
    features["has_history_flag"] = (
        (features["payment_total_count"] + features["membership_total_count"] + features["complaint_total_count"]) > 0
    ).astype(int)

    return features


def compute_score(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()

    out["payment_risk_raw"] = (
        np.log1p(out["payment_failed_count"])
        + 1.5 * np.log1p(out["payment_disputed_count"])
        + 0.8 * np.log1p(out["payment_refunded_count"])
        + 0.3 * np.log1p(out["payment_pending_count"])
        + 0.2 * np.log1p(out["payment_canceled_count"])
        + 2.0 * out["payment_issue_rate"]
        + 0.5 * out["payment_recent_event_risk_raw"]
    )

    out["complaint_risk_raw"] = (
        1.2 * np.log1p(out["complaint_open_count"])
        + 1.5 * np.log1p(out["complaint_escalated_count"])
        + 0.8 * np.log1p(out["complaint_reporter_severity_sum"])
        + 1.2 * np.log1p(out["complaint_target_severity_sum"])
        + 0.8 * out["complaint_recent_event_risk_raw"]
    )

    out["membership_risk_raw"] = (
        1.0 * np.log1p(out["membership_exit_count"])
        + 0.7 * np.log1p(out["short_membership_count"])
        + 0.3 * np.log1p(out["distinct_subscription_count"])
        + 0.2 * np.log1p(out["distinct_brand_count"])
        + 0.5 * out["membership_churn_rate"]
        + 0.3 * out["brand_switch_rate"]
        + 0.7 * out["membership_recent_event_risk_raw"]
    )

    out["recency_risk_raw"] = out["days_since_last_seen"] / 365.0

    out["payment_risk_norm"] = robust_minmax(out["payment_risk_raw"])
    out["complaint_risk_norm"] = robust_minmax(out["complaint_risk_raw"])
    out["membership_risk_norm"] = robust_minmax(out["membership_risk_raw"])
    out["recency_risk_norm"] = robust_minmax(out["recency_risk_raw"])

    out["risk_score"] = (
        100 * (
            0.40 * out["payment_risk_norm"]
            + 0.30 * out["complaint_risk_norm"]
            + 0.20 * out["membership_risk_norm"]
            + 0.10 * out["recency_risk_norm"]
        )
    ).round(2)

    out["risk_tier"] = pd.cut(
        out["risk_score"],
        bins=[-0.01, 24.99, 49.99, 74.99, 100.0],
        labels=["low", "watch", "high", "critical"],
    ).astype(str)

    out["risk_tier"] = out["risk_tier"].replace("nan", "low")
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)
    out["risk_rank"] = np.arange(1, len(out) + 1)
    return out


def export_scored_csv(final_df: pd.DataFrame, output_path: Path) -> Path:
    final_cols = [
        "user_id",
        "email",
        "country",
        "signup_at_utc",
        "last_seen_at_utc",
        "status",
        "status_is_anomalous",
        "is_new_user",
        "inactive_6m_flag",
        "has_history_flag",
        "payment_total_count",
        "payment_success_count",
        "payment_failed_count",
        "payment_disputed_count",
        "payment_refunded_count",
        "payment_pending_count",
        "payment_canceled_count",
        "payment_issue_rate",
        "payment_distinct_subscription_count",
        "payment_distinct_brand_count",
        "membership_total_count",
        "membership_exit_count",
        "membership_current_count",
        "short_membership_count",
        "avg_membership_duration_days",
        "median_membership_duration_days",
        "distinct_subscription_count",
        "distinct_brand_count",
        "membership_churn_rate",
        "brand_switch_rate",
        "complaint_total_count",
        "complaint_reporter_count",
        "complaint_target_count",
        "complaint_open_count",
        "complaint_escalated_count",
        "complaint_resolved_count",
        "complaint_total_severity",
        "days_since_last_payment_raw",
        "days_since_last_payment",
        "days_since_last_complaint_raw",
        "days_since_last_complaint",
        "days_since_last_membership_end_raw",
        "days_since_last_membership_end",
        "days_since_last_seen",
        "payment_risk_raw",
        "complaint_risk_raw",
        "membership_risk_raw",
        "recency_risk_raw",
        "payment_risk_norm",
        "complaint_risk_norm",
        "membership_risk_norm",
        "recency_risk_norm",
        "risk_score",
        "risk_tier",
        "risk_rank",
    ]

    missing = [c for c in final_cols if c not in final_df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans le résultat final : {missing}")

    export_df = final_df[final_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)
    return output_path


# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(input_db: Path, output_csv: Path) -> pd.DataFrame:
    users, subscriptions, memberships, payments, complaints = load_clean_data(input_db)

    # Sécurisation des colonnes temporelles utiles
    for df, cols in [
        (users, ["last_seen_at_utc"]),
        (subscriptions, ["created_at_utc"]),
        (memberships, ["joined_at_utc", "left_at_utc"]),
        (payments, ["created_at_utc", "captured_at_utc"]),
        (complaints, ["created_at_utc", "resolved_at_utc"]),
    ]:
        for col in cols:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    reference_date = compute_reference_date(users, subscriptions, memberships, payments, complaints)
    valid_user_ids = set(users["id"].dropna().astype(int))

    payment_agg = build_payment_features(payments, subscriptions, reference_date, valid_user_ids)
    membership_agg = build_membership_features(memberships, subscriptions, reference_date)
    complaint_agg = build_complaint_features(complaints, reference_date)

    features = merge_features(users, payment_agg, membership_agg, complaint_agg, reference_date)
    scored = compute_score(features)

    export_scored_csv(scored, output_csv)
    return scored


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute risk features and export the scored CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/risk_monitor_clean.sqlite"),
        help="Chemin vers la base SQLite nettoyée",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/subscribers_risk_scored.csv"),
        help="Chemin du CSV scoré à générer",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    scored = run_pipeline(args.input, args.output)
    print(f"CSV scoré généré : {args.output}")
    print(scored[["user_id", "risk_score", "risk_tier", "risk_rank"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
