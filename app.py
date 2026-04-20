"""Risk Monitor internal Streamlit app.

Purpose:
- show the highest-risk subscribers first
- let an operator inspect one subscriber in detail below the list
- persist watch/block/reset actions locally

Inputs:
- outputs/subscribers_risk_scored.csv
- data/processed/risk_monitor_clean.sqlite
- data/app_state.sqlite (local actions log)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Risk Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_ROOT = Path(__file__).resolve().parent
SCORING_CSV = APP_ROOT / "outputs" / "subscribers_risk_scored.csv"
CLEAN_DB = APP_ROOT / "data" / "processed" / "risk_monitor_clean.sqlite"
STATE_DB = APP_ROOT / "data" / "app_state.sqlite"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def display_value(value: Any) -> str:
    """Human-friendly display for missing values."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return "—"
    return str(value)


def fmt_dt(value: Any) -> str:
    """Format datetimes for display only."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def fmt_num(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return display_value(value)


def clean_for_display(df: pd.DataFrame, datetime_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Prepare a dataframe for a readable UI without using applymap."""
    out = df.copy()
    for col in datetime_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt_dt)
    for col in out.columns:
        out[col] = out[col].map(display_value)
    return out


def selection_label(row: dict[str, Any]) -> str:
    return (
        f"{int(row['user_id'])} | {display_value(row.get('email'))} | "
        f"score {float(row.get('risk_score', 0)):.2f} | {display_value(row.get('risk_tier'))}"
    )


def risk_reason(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if float(row.get("payment_failed_count", 0)) > 0 or float(row.get("payment_disputed_count", 0)) > 0:
        reasons.append("paiements irréguliers")
    if float(row.get("complaint_total_count", 0)) > 0:
        reasons.append("réclamations")
    if float(row.get("membership_exit_count", 0)) > 0 or float(row.get("short_membership_count", 0)) > 0:
        reasons.append("churn memberships")
    if int(row.get("inactive_6m_flag", 0)) == 1:
        reasons.append("inactif 6 mois")
    if int(row.get("status_is_anomalous", 0)) == 1:
        reasons.append("statut anormal")
    if int(row.get("is_new_user", 0)) == 1:
        reasons.append("nouveau user")
    return reasons[:4]


def friendly_action(value: str | None) -> str:
    return {"watch": "À surveiller", "block": "À bloquer", "reset": "Réinitialisé"}.get(value or "", "—")


def risk_badge(row: pd.Series) -> str:
    tier = str(row.get("risk_tier", "low"))
    if tier == "critical":
        return "Critique"
    if tier == "high":
        return "Élevé"
    if tier == "watch":
        return "À surveiller"
    return "Faible"


# -----------------------------------------------------------------------------
# Local SQLite state
# -----------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    )
    return cur.fetchone() is not None


@st.cache_data(show_spinner=False)
def load_scored_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV scoré introuvable : {path}. Lance l'étape 3 pour générer outputs/subscribers_risk_scored.csv."
        )
    df = pd.read_csv(path)
    for col in [
        "signup_at_utc",
        "last_seen_at_utc",
        "payment_last_at",
        "last_membership_end_at",
        "last_complaint_at",
        "complaint_reporter_last_at",
        "complaint_target_last_at",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_clean_tables(db_path: str) -> dict[str, pd.DataFrame]:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SQLite nettoyé introuvable : {path}. Lance l'étape 2 pour générer data/processed/risk_monitor_clean.sqlite."
        )
    conn = sqlite3.connect(path)
    try:
        tables: dict[str, pd.DataFrame] = {}
        for table in ["users", "subscriptions", "memberships", "payments", "complaints"]:
            tables[table] = pd.read_sql_query(f"SELECT * FROM {table};", conn)

        for table_name, cols in {
            "users": ["signup_at_utc", "last_seen_at_utc"],
            "subscriptions": ["created_at_utc"],
            "memberships": ["joined_at_utc", "left_at_utc"],
            "payments": ["created_at_utc", "captured_at_utc"],
            "complaints": ["created_at_utc", "resolved_at_utc"],
        }.items():
            for col in cols:
                if col in tables[table_name].columns:
                    tables[table_name][col] = pd.to_datetime(tables[table_name][col], utc=True, errors="coerce")
        return tables
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_action_state(db_path: str) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "action", "updated_at", "note"])
    conn = sqlite3.connect(path)
    try:
        if not _table_exists(conn, "actions"):
            return pd.DataFrame(columns=["user_id", "action", "updated_at", "note"])
        df = pd.read_sql_query("SELECT user_id, action, updated_at, note FROM actions ORDER BY updated_at DESC;", conn)
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
        return df
    finally:
        conn.close()


def ensure_state_db() -> None:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                note TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def persist_action(user_id: int, action: str, note: str = "") -> None:
    ensure_state_db()
    conn = sqlite3.connect(STATE_DB)
    try:
        conn.execute(
            "INSERT INTO actions (user_id, action, updated_at, note) VALUES (?, ?, ?, ?);",
            (int(user_id), action, pd.Timestamp.utcnow().isoformat(), note.strip() or None),
        )
        conn.commit()
    finally:
        conn.close()
    load_action_state.clear()


def get_latest_actions() -> pd.DataFrame:
    actions = load_action_state(str(STATE_DB))
    if actions.empty:
        return actions
    latest = actions.sort_values("updated_at").drop_duplicates("user_id", keep="last")
    return latest[["user_id", "action", "updated_at", "note"]]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_all_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    scored = load_scored_csv(str(SCORING_CSV))
    tables = load_clean_tables(str(CLEAN_DB))
    latest_actions = get_latest_actions()
    if not latest_actions.empty:
        scored = scored.merge(latest_actions, on="user_id", how="left", suffixes=("", "_state"))
    else:
        scored["action"] = pd.NA
        scored["updated_at"] = pd.NaT
        scored["note"] = pd.NA
    return scored, tables, latest_actions


# -----------------------------------------------------------------------------
# App content
# -----------------------------------------------------------------------------


title_col, subtitle_col = st.columns([3, 1])
with title_col:
    st.title("Risk Monitor")
    st.write(
        "La vue ci-dessous classe les subscribers du plus risqué au moins risqué. "
        "La liste prioritaire est en haut, puis un détail plus léger apparaît en dessous pour le subscriber sélectionné."
    )
with subtitle_col:
    st.info("Vue opérateur")

if not SCORING_CSV.exists():
    st.error(f"CSV scoré introuvable : {SCORING_CSV}")
    st.stop()

if not CLEAN_DB.exists():
    st.error(f"SQLite nettoyé introuvable : {CLEAN_DB}")
    st.stop()

ensure_state_db()
scored, tables, _latest_actions = load_all_data()

if scored.empty:
    st.warning("Aucune donnée scorée disponible.")
    st.stop()

scored = scored.copy()
if "action" not in scored.columns:
    scored["action"] = pd.NA
if "note" not in scored.columns:
    scored["note"] = pd.NA
if "updated_at" not in scored.columns:
    scored["updated_at"] = pd.NaT

scored["country_display"] = scored["country"].map(display_value)
scored["status_display"] = scored["status"].map(display_value)
scored["latest_action_label"] = scored["action"].map(lambda v: friendly_action(v if pd.notna(v) else None))
scored["latest_action_label"] = scored["latest_action_label"].fillna("—")
scored["why_remains"] = scored.apply(
    lambda row: "; ".join(risk_reason(row)) if risk_reason(row) else "Aucun signal fort",
    axis=1,
)
scored["last_activity_display"] = scored["last_seen_at_utc"].map(fmt_dt)
scored["signup_display"] = scored["signup_at_utc"].map(fmt_dt)
scored["badge"] = scored.apply(risk_badge, axis=1)

# Sidebar filters
st.sidebar.header("Filtres")

score_min, score_max = st.sidebar.slider(
    "Plage de score",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=0.5,
)

tier_order = ["critical", "high", "watch", "low"]
selected_tiers = st.sidebar.multiselect(
    "Niveau de risque",
    options=tier_order,
    default=tier_order,
)

country_options = sorted([v for v in scored["country_display"].dropna().astype(str).unique().tolist() if v != "—"])
selected_countries = st.sidebar.multiselect(
    "Pays",
    options=country_options,
    default=[],
)

status_options = sorted([v for v in scored["status_display"].dropna().astype(str).unique().tolist() if v != "—"])
selected_status = st.sidebar.multiselect(
    "Statut utilisateur (code brut)",
    options=status_options,
    default=[],
)

last_seen_min = scored["last_seen_at_utc"].min()
last_seen_max = scored["last_seen_at_utc"].max()
if pd.notna(last_seen_min) and pd.notna(last_seen_max):
    last_seen_range = st.sidebar.date_input(
        "Dernière activité entre",
        value=(last_seen_min.date(), last_seen_max.date()),
        min_value=last_seen_min.date(),
        max_value=last_seen_max.date(),
    )
else:
    last_seen_range = None

signup_min = scored["signup_at_utc"].min()
signup_max = scored["signup_at_utc"].max()
if pd.notna(signup_min) and pd.notna(signup_max):
    signup_range = st.sidebar.date_input(
        "Inscription entre",
        value=(signup_min.date(), signup_max.date()),
        min_value=signup_min.date(),
        max_value=signup_max.date(),
    )
else:
    signup_range = None

inactive_only = st.sidebar.checkbox("Seulement les inactifs 6 mois", value=False)

history_filter = st.sidebar.selectbox(
    "Historique",
    options=["Tous", "Avec historique", "Sans historique"],
    index=0,
)

action_filter = st.sidebar.selectbox(
    "Action actuelle",
    options=["Toutes", "Aucune action", "Watch", "Block", "Reset"],
    index=0,
)

search_query = st.sidebar.text_input(
    "Recherche email ou ID",
    value="",
    placeholder="Ex. 538 ou user_538",
)

# Apply filters
filtered = scored.loc[
    (scored["risk_score"] >= score_min)
    & (scored["risk_score"] <= score_max)
    & (scored["risk_tier"].isin(selected_tiers))
].copy()

if selected_countries:
    filtered = filtered[filtered["country_display"].isin(selected_countries)]
if selected_status:
    filtered = filtered[filtered["status_display"].isin(selected_status)]

if isinstance(last_seen_range, tuple) and len(last_seen_range) == 2:
    start_last_seen = pd.Timestamp(last_seen_range[0], tz="UTC")
    end_last_seen = pd.Timestamp(last_seen_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = filtered[
        filtered["last_seen_at_utc"].notna()
        & (filtered["last_seen_at_utc"] >= start_last_seen)
        & (filtered["last_seen_at_utc"] <= end_last_seen)
    ]

if isinstance(signup_range, tuple) and len(signup_range) == 2:
    start_signup = pd.Timestamp(signup_range[0], tz="UTC")
    end_signup = pd.Timestamp(signup_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = filtered[
        filtered["signup_at_utc"].notna()
        & (filtered["signup_at_utc"] >= start_signup)
        & (filtered["signup_at_utc"] <= end_signup)
    ]

if inactive_only:
    filtered = filtered[filtered["inactive_6m_flag"] == 1]
if history_filter == "Avec historique":
    filtered = filtered[filtered["has_history_flag"] == 1]
elif history_filter == "Sans historique":
    filtered = filtered[filtered["has_history_flag"] == 0]

if action_filter == "Watch":
    filtered = filtered[filtered["action"].astype(str).str.lower() == "watch"]
elif action_filter == "Block":
    filtered = filtered[filtered["action"].astype(str).str.lower() == "block"]
elif action_filter == "Reset":
    filtered = filtered[filtered["action"].astype(str).str.lower() == "reset"]
elif action_filter == "Aucune action":
    filtered = filtered[filtered["action"].isna() | (filtered["action"].astype(str).str.strip() == "")]

if search_query.strip():
    q = search_query.strip().lower()
    filtered = filtered[
        filtered["email"].astype(str).str.lower().str.contains(q, na=False)
        | filtered["user_id"].astype(str).str.contains(q, na=False)
    ]

filtered = filtered.sort_values(["risk_score", "user_id"], ascending=[False, True]).reset_index(drop=True)

# Top metrics
critical_count = int((filtered["risk_tier"] == "critical").sum()) if not filtered.empty else 0
watch_count = int((filtered["risk_tier"] == "watch").sum()) if not filtered.empty else 0
high_count = int((filtered["risk_tier"] == "high").sum()) if not filtered.empty else 0
avg_score = float(filtered["risk_score"].mean()) if not filtered.empty else 0.0
oldest_seen = fmt_dt(filtered["last_seen_at_utc"].min()) if not filtered.empty else "—"

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Subscribers dans la vue", len(filtered))
with metric_cols[1]:
    st.metric("Risque critique", critical_count)
with metric_cols[2]:
    st.metric("Score moyen", f"{avg_score:.2f}")
with metric_cols[3]:
    st.metric("Dernière activité la plus ancienne", oldest_seen)

st.caption(
    "La liste est triée du score le plus élevé au plus faible. "
    "Les valeurs manquantes sont affichées sous une forme lisible."
)

# -----------------------------------------------------------------------------
# Prioritized list
# -----------------------------------------------------------------------------

st.subheader("Subscribers prioritaires")

if filtered.empty:
    st.info("Aucun subscriber ne correspond aux filtres actuels.")
    st.stop()

list_cols = [
    "user_id",
    "email",
    "country_display",
    "risk_score",
    "risk_tier",
    "last_activity_display",
    "payment_failed_count",
    "complaint_total_count",
    "membership_exit_count",
    "latest_action_label",
    "why_remains",
]
list_view = filtered[list_cols].copy()
list_view = list_view.rename(
    columns={
        "user_id": "Subscriber ID",
        "email": "Email",
        "country_display": "Pays",
        "risk_score": "Score",
        "risk_tier": "Niveau",
        "last_activity_display": "Dernière activité",
        "payment_failed_count": "Échecs paiements",
        "complaint_total_count": "Réclamations",
        "membership_exit_count": "Sorties memberships",
        "latest_action_label": "Action",
        "why_remains": "Pourquoi il remonte",
    }
)
list_view["Score"] = list_view["Score"].map(lambda x: f"{float(x):.2f}")
list_view["Niveau"] = list_view["Niveau"].map(display_value)
list_view["Dernière activité"] = list_view["Dernière activité"].map(display_value)
list_view["Échecs paiements"] = list_view["Échecs paiements"].map(lambda x: fmt_num(x, 0))
list_view["Réclamations"] = list_view["Réclamations"].map(lambda x: fmt_num(x, 0))
list_view["Sorties memberships"] = list_view["Sorties memberships"].map(lambda x: fmt_num(x, 0))
list_view["Action"] = list_view["Action"].map(display_value)
list_view = clean_for_display(list_view)

st.dataframe(list_view.head(200), use_container_width=True, hide_index=True, height=460)

# -----------------------------------------------------------------------------
# Selection and detail section below the list
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Détail d’un subscriber")
st.caption("Choisissez une ligne pour voir un résumé rapide, l’historique et les actions disponibles.")

selection_pool = filtered.head(300).to_dict("records")
label_map = {int(row["user_id"]): selection_label(row) for row in selection_pool}
options = [int(row["user_id"]) for row in selection_pool]

if not options:
    st.warning("Aucun subscriber disponible pour la sélection.")
    st.stop()

if "selected_user_id" not in st.session_state or st.session_state.selected_user_id not in options:
    st.session_state.selected_user_id = options[0]

selected_user_id = st.selectbox(
    "Choisir un subscriber",
    options=options,
    format_func=lambda x: label_map.get(int(x), str(x)),
    key="selected_user_id",
)

selected_row = filtered.loc[filtered["user_id"] == int(selected_user_id)].head(1)
if selected_row.empty:
    selected_row = scored.loc[scored["user_id"] == int(selected_user_id)].head(1)

if selected_row.empty:
    st.warning("Impossible de charger ce subscriber.")
    st.stop()

row = selected_row.iloc[0]
current_action_value = row.get("action") if pd.notna(row.get("action")) else None
current_action_label = friendly_action(current_action_value)
notes = display_value(row.get("note"))

summary_tab, history_tab, actions_tab = st.tabs(["Résumé", "Historique", "Actions"])

with summary_tab:
    st.markdown("##### Résumé rapide")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Score", fmt_num(row.get("risk_score"), 2))
    with summary_cols[1]:
        st.metric("Niveau", display_value(row.get("risk_tier")))
    with summary_cols[2]:
        st.metric("Badge", risk_badge(row))

    summary_info = st.columns(2)
    with summary_info[0]:
        st.write(f"**Subscriber ID :** {display_value(row.get('user_id'))}")
        st.write(f"**Email :** {display_value(row.get('email'))}")
        st.write(f"**Pays :** {display_value(row.get('country'))}")
        st.write(f"**Statut brut :** {display_value(row.get('status'))}")
        st.write(f"**Dernière activité :** {fmt_dt(row.get('last_seen_at_utc'))}")
        st.write(f"**Inscription :** {fmt_dt(row.get('signup_at_utc'))}")
    with summary_info[1]:
        st.write(f"**Action actuelle :** {current_action_label}")
        st.write(f"**Commentaire :** {notes}")
        reasons = risk_reason(row)
        st.write("**Pourquoi il remonte :**")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("- Aucun signal fort")

    key_signals = pd.DataFrame(
        [
            ["Échecs de paiement", row.get("payment_failed_count", 0)],
            ["Réclamations", row.get("complaint_total_count", 0)],
            ["Sorties memberships", row.get("membership_exit_count", 0)],
            ["Inactivité 6 mois", "Oui" if int(row.get("inactive_6m_flag", 0)) == 1 else "Non"],
            ["Historique disponible", "Oui" if int(row.get("has_history_flag", 0)) == 1 else "Non"],
        ],
        columns=["Signal", "Valeur"],
    )
    st.dataframe(key_signals, use_container_width=True, hide_index=True)

with history_tab:
    st.markdown("##### Paiements")
    user_payments = tables["payments"].loc[tables["payments"]["user_id"] == int(selected_user_id)].copy()
    if user_payments.empty:
        st.info("Aucun paiement pour ce subscriber.")
    else:
        subscriptions_tbl = tables["subscriptions"][ ["id", "brand", "price_cents", "currency"] ].copy()
        payments_history = user_payments.merge(
            subscriptions_tbl,
            left_on="subscription_id",
            right_on="id",
            how="left",
            suffixes=("", "_sub"),
        )
        pay_cols = [c for c in [
            "created_at_utc",
            "captured_at_utc",
            "subscription_id",
            "brand",
            "status",
            "amount_cents",
            "fee_cents",
            "currency",
            "stripe_error_code",
        ] if c in payments_history.columns]
        st.dataframe(
            clean_for_display(
                payments_history.sort_values("created_at_utc", ascending=False)[pay_cols].head(50),
                datetime_cols=("created_at_utc", "captured_at_utc"),
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("##### Memberships")
    user_memberships = tables["memberships"].loc[tables["memberships"]["user_id"] == int(selected_user_id)].copy()
    if user_memberships.empty:
        st.info("Aucun membership pour ce subscriber.")
    else:
        subscriptions_tbl = tables["subscriptions"][ ["id", "brand", "price_cents", "currency"] ].copy()
        memberships_history = user_memberships.merge(
            subscriptions_tbl,
            left_on="subscription_id",
            right_on="id",
            how="left",
            suffixes=("", "_sub"),
        )
        mem_cols = [c for c in [
            "joined_at_utc",
            "left_at_utc",
            "subscription_id",
            "brand",
            "status",
            "reason",
        ] if c in memberships_history.columns]
        st.dataframe(
            clean_for_display(
                memberships_history.sort_values("joined_at_utc", ascending=False)[mem_cols].head(50),
                datetime_cols=("joined_at_utc", "left_at_utc"),
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("##### Réclamations")
    complaints_tbl = tables["complaints"].copy()
    complaints_reporter = complaints_tbl.loc[complaints_tbl["reporter_id"] == int(selected_user_id)].copy()
    complaints_reporter["role"] = "reporter"
    complaints_target = complaints_tbl.loc[complaints_tbl["target_id"] == int(selected_user_id)].copy()
    complaints_target["role"] = "target"
    complaints_history = pd.concat([complaints_reporter, complaints_target], ignore_index=True, sort=False)
    if complaints_history.empty:
        st.info("Aucune réclamation pour ce subscriber.")
    else:
        compl_cols = [c for c in [
            "created_at_utc",
            "resolved_at_utc",
            "role",
            "type",
            "status",
            "resolution",
            "subscription_id",
        ] if c in complaints_history.columns]
        st.dataframe(
            clean_for_display(
                complaints_history.sort_values("created_at_utc", ascending=False)[compl_cols].head(50),
                datetime_cols=("created_at_utc", "resolved_at_utc"),
            ),
            use_container_width=True,
            hide_index=True,
        )

with actions_tab:
    st.markdown("##### Action opérateur")
    action_info = st.columns(3)
    with action_info[0]:
        st.write(f"**Action actuelle :** {current_action_label}")
    with action_info[1]:
        st.write(f"**Commentaire :** {notes}")
    with action_info[2]:
        st.write(f"**Dernière mise à jour :** {fmt_dt(row.get('updated_at'))}")

    note_text = st.text_area(
        "Commentaire (optionnel)",
        value="",
        placeholder="Ex. paiement échoué plusieurs fois + réclamations récentes",
        help="Ajoutez un court motif pour garder une trace de la décision.",
    )

    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("Watch", use_container_width=True):
            persist_action(int(selected_user_id), "watch", note_text)
            st.success("Subscriber marqué comme à surveiller.")
            st.rerun()
    with btn_cols[1]:
        if st.button("Block", use_container_width=True):
            persist_action(int(selected_user_id), "block", note_text)
            st.success("Subscriber marqué comme bloqué.")
            st.rerun()
    with btn_cols[2]:
        if st.button("Reset", use_container_width=True):
            persist_action(int(selected_user_id), "reset", note_text)
            st.success("Action réinitialisée.")
            st.rerun()

    st.markdown("##### Journal des actions pour ce subscriber")
    user_history = load_action_state(str(STATE_DB)).loc[lambda d: d["user_id"] == int(selected_user_id)].copy()
    if user_history.empty:
        st.info("Aucune action enregistrée pour ce subscriber.")
    else:
        action_view = user_history[["updated_at", "action", "note"]].copy()
        action_view = action_view.rename(columns={"updated_at": "Date", "action": "Action", "note": "Commentaire"})
        action_view["Date"] = pd.to_datetime(action_view["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
        action_view["Action"] = action_view["Action"].map(friendly_action)
        st.dataframe(clean_for_display(action_view), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Journal des actions global")
actions_log = load_action_state(str(STATE_DB))
if actions_log.empty:
    st.info("Aucune action enregistrée pour le moment.")
else:
    global_actions = actions_log.head(50).copy()
    global_actions = global_actions.rename(columns={"updated_at": "Date", "action": "Action", "note": "Commentaire", "user_id": "Subscriber ID"})
    global_actions["Date"] = pd.to_datetime(global_actions["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
    global_actions["Action"] = global_actions["Action"].map(friendly_action)
    st.dataframe(clean_for_display(global_actions), use_container_width=True, hide_index=True)
