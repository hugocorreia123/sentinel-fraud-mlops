"""Sentinel — friendly UI layer for the control panel.

Welcome tour, plain-language explanations, honesty box, and a help tab.
Streamlit only — no other dependencies. Same pattern as Mandate's friendly
UI, in Sentinel's voice: written for the fraud-operations analyst.
"""

import streamlit as st

_TOUR_KEY = "sentinel_tour_done"


# ------------------------------------------------------------------ welcome
def show_welcome() -> bool:
    """One-time plain-language tour. Returns True once dismissed."""
    if st.session_state.get(_TOUR_KEY):
        return True

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.title("🛡️ Sentinel")
        st.subheader("A card swipe is either blocked in under 100 ms — or it isn't.")
        st.markdown(
            """
Fraud does not wait for a batch job. Sentinel is a **production-shape**
real-time fraud system: a champion model makes the call on every
transaction while a challenger scores the same traffic in shadow, with
latency, drift and decision rates monitored the way a real fraud team
would run it.

**What you can do here**

- **🎯 Score a transaction** — fill in a transaction (or click a preset,
  including a classic drain-to-zero fraud) and watch the live decision:
  APPROVE or BLOCK, with the probability, threshold and latency behind
  it.
- **📊 Live system state** — the champion model's identity, service
  health, and running approve/block counters straight from the
  service's own metrics.
- **❓ Help** — every term in plain language, plus exactly what this
  hosted demo runs and what it deliberately leaves out.

**What this will not do**

It scores one transaction at a time on a synthetic benchmark. It is the
*shape* of a production fraud stack — the point is the engineering
around the model, not a claim about your bank.

*Data: PaySim, a synthetic mobile-money simulation. No real financial
data anywhere in this system.*
"""
        )
        c1, c2 = st.columns(2)
        if c1.button("Start exploring", type="primary", use_container_width=True):
            st.session_state[_TOUR_KEY] = True
            st.rerun()
        if c2.button("Skip the intro", use_container_width=True):
            st.session_state[_TOUR_KEY] = True
            st.rerun()
    return False


def show_replay() -> None:
    """Small control to replay the intro tour."""
    if st.button("↻ Replay the intro", help="Show the welcome tour again"):
        st.session_state[_TOUR_KEY] = False
        st.rerun()


def show_api_down_caveat() -> None:
    """Friendly explanation when the inference service is unreachable."""
    st.warning(
        "**The inference service isn't reachable right now.** This panel is "
        "only the window — the decisions are made by a separate FastAPI "
        "service (that separation is the production shape). On the hosted "
        "Space it starts alongside the panel and can take a moment to come "
        "up; locally, start it with `docker compose up` or "
        "`uvicorn apps.inference.main:app`. Scoring will work as soon as "
        "the sidebar shows the service as reachable."
    )


# ------------------------------------------------------------------ method
def show_how_it_works() -> None:
    """Plain-language story for the top of the About tab."""
    st.markdown(
        """
#### How it works, in one breath

Every transaction goes to a FastAPI service where the **champion** model
(LightGBM) makes the real decision against a **cost-aware threshold** —
the cutoff that minimises money lost, not a generic 0.5. A **challenger**
model (a PyTorch network) scores the same traffic **in shadow**: its
answers are logged and compared, never served, until it earns promotion.
Latency, decision rates and input drift are watched continuously —
because the fraud rate in the data drifts 5× over time, and a model
nobody watches goes quietly stale.

**Does this affect what you see?** Yes: the decision card shows which
model and version answered, the threshold it used, and the latency it
took — the exact things an on-call fraud engineer would check first.
"""
    )
    st.markdown("---")


# ------------------------------------------------------------------ help
def show_help() -> None:
    """Help tab: orientation, mini-glossary, and the honesty box."""
    st.header("❓ Help")

    st.markdown(
        """
#### What am I looking at?

The control panel of a real-time fraud-detection system built
production-shape on the PaySim benchmark (6.36M synthetic mobile-money
transactions, ~0.13% fraud).

**Start here:** open **🎯 Score a transaction**, click the
**💸 Drain-to-zero fraud** preset, and score it. Then score the
**🛍️ Normal PAYMENT** preset and compare the two decision cards.

#### Words on the screen, in plain language

- **Champion / challenger** — the model in charge vs the candidate
  scoring the same traffic silently. The challenger is promoted only
  after proving itself in shadow.
- **Shadow mode** — the challenger's answers are logged and compared,
  never acted on.
- **Threshold** — the fraud-probability cutoff for blocking. Chosen by
  cost (missed fraud vs blocked customers), not by default.
- **p99 latency** — the time the *slowest 1%* of requests take. The SLO
  here is under 100 ms, held under load test.
- **Drift** — the live traffic slowly stops resembling the training
  data. PaySim's fraud rate climbs ~5× toward the end — drift detection
  exists because that happens in the real world too.
- **Velocity features** — how fast money moved through an account
  recently; computed from a Redis feature store in the full stack.
"""
    )

    st.markdown("#### The honesty box")
    st.markdown(
        """
- **This hosted demo is a subset of the full system.** In this container:
  the FastAPI service, both models, and a local prediction log. **Not**
  in this container: the Redis feature store (velocity features default
  to 0 here), Prometheus + Grafana dashboards, Evidently drift reports,
  and the MLflow registry — those run in the full local stack
  (`docker compose up` in the repo).
- **The shadow deployment earned its keep by failing first.** The
  challenger initially disagreed with itself between evaluation and
  serving — a BatchNorm eval-mode bug surfaced by shadow comparison,
  fixed by retraining without the offending features to 100% agreement.
  Shadow mode is in this project because it caught a real bug.
- **Synthetic data.** PaySim is a simulation calibrated on a real
  mobile-money operator; the engineering transfers, the exact fraud
  patterns do not.
- **One container, one instance.** The hosted demo's throughput is not
  the load-test number; the sub-100 ms p99 was verified under Locust
  load locally.
"""
    )

    st.markdown(
        """
#### Links

[GitHub repository](https://github.com/hugocorreia123/sentinel-fraud-mlops)
· Hugo Correia — [LinkedIn](https://www.linkedin.com/in/hugogncorreia)
"""
    )
    show_replay()
