"""Streamlit control panel for Sentinel.

Three tabs:
1. Score a transaction (form → POST /predict → result card)
2. Live system state (polls /health, /model/info, /metrics)
3. About (project description + links)

The Streamlit app talks to FastAPI on http://localhost:8000 by default,
configurable via the SENTINEL_API_URL env var.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("SENTINEL_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Sentinel — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

import sentinel_theme as th
th.inject()

import sentinel_friendly as sf

if not sf.show_welcome():
    st.stop()


# ---------- Helpers ----------------------------------------------------------
def api_get(path: str, timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on GET {path}: {e}")
        return None


def api_post(path: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"API HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        return None
    except Exception as e:
        st.error(f"API error on POST {path}: {e}")
        return None


def metrics_text() -> str | None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5.0)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def parse_metric_total(metrics: str, metric_name: str) -> float:
    """Sum all label-permutations of a Prometheus counter."""
    if not metrics:
        return 0.0
    total = 0.0
    for line in metrics.splitlines():
        if line.startswith("#") or not line:
            continue
        if line.startswith(metric_name):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                continue
    return total


# ---------- Sidebar ----------------------------------------------------------
st.sidebar.title("🛡️ Sentinel")
st.sidebar.caption("Real-time fraud detection — demo UI")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{API_URL}`")
health = api_get("/health", timeout=2.0)
if health:
    st.sidebar.success("✅ Inference service reachable")
else:
    st.sidebar.error("❌ Inference service unreachable")
    sf.show_api_down_caveat()
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[GitHub repo](https://github.com/hugocorreia123/sentinel-fraud-mlops)"
)

# ---------- Tabs -------------------------------------------------------------
th.hero(
    "Real-time Fraud MLOps",
    "Sentinel",
    "A champion model decides every transaction in under 100 ms while a "
    "challenger shadows the same traffic. Approve or block — with the "
    "model, version and latency behind each call.",
    "PaySim synthetic mobile-money · production-shape by design",
)
tab_score, tab_system, tab_about, tab_help = st.tabs(
    ["🎯 Score a transaction", "📊 Live system state", "ℹ️ About", "❓ Help"]
)

# === TAB 1 — score a transaction =============================================
with tab_score:
    st.header("Score a transaction")
    st.caption(
        "Fill in a transaction — or click a preset below — and the "
        "champion model decides in real time. A challenger model scores "
        "the same call silently in shadow; its answers are compared, "
        "never served."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction")
        tx_type = st.selectbox(
            "Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
            help="PaySim transaction type",
        )
        amount = st.number_input("Amount (€)", min_value=0.0, value=5000.0, step=100.0)
        step = st.number_input("Step (hour since simulation start)", min_value=1, value=400, step=1)

    with col2:
        st.subheader("Accounts")
        name_orig = st.text_input("Originator account (e.g. C123456789)", value="C111111111")
        old_org = st.number_input("Originator old balance (€)", min_value=0.0, value=5000.0)
        new_org = st.number_input("Originator new balance (€)", min_value=0.0, value=0.0)
        name_dest = st.text_input("Destination account (e.g. C987654321)", value="C999999999")
        old_dest = st.number_input("Destination old balance (€)", min_value=0.0, value=0.0)
        new_dest = st.number_input("Destination new balance (€)", min_value=0.0, value=5000.0)

    st.markdown("**Ready-made examples**")
    st.caption("One click fills the whole form above with a complete "
               "example — pick **one**, then press **Score transaction**. "
               "Nothing is scored until you press the button, and you can "
               "still edit any field first.")
    p1, p2, p3 = st.columns(3)
    if p1.button("💸 Classic fraud — account drained to zero",
                 use_container_width=True,
                 help="A transfer that empties the sender's account in one "
                      "move — the signature PaySim fraud pattern. Expect "
                      "the model to BLOCK it."):
        st.session_state["preset"] = "fraud"
    if p2.button("🛍️ Everyday purchase — small payment",
                 use_container_width=True,
                 help="A €50 payment from a healthy balance — ordinary "
                      "behaviour. Expect an instant APPROVE."):
        st.session_state["preset"] = "payment"
    if p3.button("🏧 Everyday cash withdrawal",
                 use_container_width=True,
                 help="A routine €3,000 cash-out that leaves money in the "
                      "account — normal behaviour. Expect an APPROVE."):
        st.session_state["preset"] = "cashout"

    if st.session_state.get("preset") == "fraud":
        tx_type, amount, old_org, new_org, old_dest, new_dest = "TRANSFER", 250000.0, 250000.0, 0.0, 0.0, 0.0
    elif st.session_state.get("preset") == "payment":
        tx_type, amount, old_org, new_org, old_dest, new_dest = "PAYMENT", 50.0, 1000.0, 950.0, 0.0, 0.0
    elif st.session_state.get("preset") == "cashout":
        tx_type, amount, old_org, new_org, old_dest, new_dest = "CASH_OUT", 3000.0, 5000.0, 2000.0, 1000.0, 4000.0

    submitted = st.button("🚀 Score transaction", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "step": int(step),
            "type": tx_type,
            "amount": float(amount),
            "nameOrig": name_orig,
            "oldbalanceOrg": float(old_org),
            "newbalanceOrig": float(new_org),
            "nameDest": name_dest,
            "oldbalanceDest": float(old_dest),
            "newbalanceDest": float(new_dest),
        }
        with st.spinner("Scoring…"):
            result = api_post("/predict", payload)

        if result:
            decision = result["decision"]
            proba = result["fraud_probability"]
            threshold = result["threshold_used"]
            latency_ms = result["latency_ms"]
            model_name = result["model_name"]
            model_version = result["model_version"]
            request_id = result["request_id"]

            if decision == "BLOCK":
                st.error(f"🚨 BLOCKED — fraud probability {proba:.4f}")
            else:
                st.success(f"✅ APPROVED — fraud probability {proba:.4f}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud probability", f"{proba:.4f}")
            m2.metric("Threshold", f"{threshold:.4f}")
            m3.metric("Latency", f"{latency_ms:.1f} ms")
            m4.metric("Model", f"{model_name} v{model_version}")
            st.caption(f"**Means:** the model puts a {proba:.2%} chance "
                       f"this is fraud; anything above {threshold:.4f} "
                       f"gets blocked — a cutoff chosen by cost, not by "
                       f"default. Decided in {latency_ms:.1f} ms by "
                       f"{model_name} v{model_version}.")
            st.caption(f"Request ID: `{request_id}` — features used: {result.get('features_used', 'n/a')}")

            with st.expander("Request payload"):
                st.json(payload)
            with st.expander("Full response"):
                st.json(result)

# === TAB 2 — system state ====================================================
with tab_system:
    st.header("Live system state")
    st.caption("Polls the inference service. Refresh the page to update.")

    # --- Model info -------------------------------------------------------
    info = api_get("/model/info")
    health_info = api_get("/health")

    if info:
        st.subheader("🏆 Champion model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", info.get("name", "?"))
        c2.metric("Version", f"v{info.get('version', '?')}")
        c3.metric("Threshold", f"{info.get('threshold', 0):.4f}")
        c4.metric("Features", info.get("n_features", "?"))
        st.caption(
            f"Framework: `{info.get('framework', '?')}` · "
            f"Loaded at: {info.get('loaded_at', '?')}"
        )

    if health_info:
        st.markdown("---")
        st.subheader("🩺 Service health")
        h1, h2, h3 = st.columns(3)
        h1.metric(
            "Inference service",
            "✅ UP" if health_info.get("status") == "ok" else "❌ DOWN",
        )
        h2.metric(
            "Model loaded",
            "✅ YES" if health_info.get("model_loaded") else "❌ NO",
        )
        h3.metric(
            "Redis reachable",
            "✅ YES" if health_info.get("redis_reachable") else "❌ NO",
        )

    # --- Live counters from /metrics --------------------------------------
    st.markdown("---")
    st.subheader("📊 Service counters")
    metrics = metrics_text()
    if metrics:
        approved = parse_metric_total(metrics, 'sentinel_predictions_total{decision="APPROVE"')
        blocked = parse_metric_total(metrics, 'sentinel_predictions_total{decision="BLOCK"')
        total = approved + blocked

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total predictions", f"{int(total):,}")
        c2.metric("Approved", f"{int(approved):,}")
        c3.metric("Blocked", f"{int(blocked):,}")
        if total > 0:
            c4.metric("Block rate", f"{blocked/total*100:.2f}%")
    else:
        st.warning("Could not read /metrics — service may still be starting.")

    if st.button("Refresh", type="secondary"):
        st.rerun()

# === TAB 3 — about ===========================================================
with tab_about:
    st.header("About Sentinel")
    sf.show_how_it_works()
    st.markdown(
        """
**Sentinel** is a production-shape ML system for real-time fraud detection,
built end-to-end as a learning + portfolio project.

This Streamlit app is the **demo control panel** — it talks to the same
FastAPI inference service that the load tests and Grafana dashboards target.
The container hosting this UI also runs:

- The FastAPI inference service (on internal port 8000)
- Both registered models (champion: LightGBM, challenger: PyTorch MLP)
- An SQLite prediction log

Components that exist locally but are **not** in this hosted demo:

- Redis online feature store → the demo runs with velocity features defaulting to 0
- Prometheus + Grafana + Evidently → see GitHub for the full local stack
- MLflow registry → the demo loads model artifacts from disk

**Read more:**

- [GitHub repository](https://github.com/hugocorreia123/sentinel-fraud-mlops)
- [Phase 4 — shadow-mode bug story](https://github.com/hugocorreia123/sentinel-fraud-mlops/blob/main/docs/reports/phase4_shadow_ab_analysis.md)
- [Phase 5 — monitoring stack](https://github.com/hugocorreia123/sentinel-fraud-mlops/blob/main/docs/reports/phase5_monitoring.md)
- [Phase 6 — adversarial + load](https://github.com/hugocorreia123/sentinel-fraud-mlops/blob/main/docs/reports/phase6_robustness_and_load.md)

**Author:** Hugo Correia · Data Scientist / ML Engineer · Lisbon
"""
    )

# === TAB 4 — help ============================================================
with tab_help:
    sf.show_help()
