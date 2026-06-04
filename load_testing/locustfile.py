"""Locust load test for the Sentinel inference API.

Run interactively with the web UI:
    uv run --group load_test locust -f load_testing/locustfile.py \
        --host http://127.0.0.1:8000

Run headless (CI-style) — 100 users for 30 seconds:
    uv run --group load_test locust -f load_testing/locustfile.py \
        --host http://127.0.0.1:8000 \
        --users 100 --spawn-rate 20 --run-time 30s --headless --csv=load_testing/report

Three behaviours are mixed to simulate realistic fraud traffic:
  - 90% clean PAYMENT transactions
  - 5% clean CASH_OUT transactions
  - 5% fraud-shaped TRANSFER (drain to zero)
"""
from __future__ import annotations

import random
import string

from locust import HttpUser, between, task


def _random_account_id(prefix: str = "C") -> str:
    """PaySim-style account ID: prefix + 8-10 random digits."""
    return prefix + "".join(random.choices(string.digits, k=random.randint(8, 10)))


def _clean_payment() -> dict:
    amount = round(random.uniform(5, 5000), 2)
    old_balance = round(amount + random.uniform(100, 100_000), 2)
    return {
        "step": random.randint(380, 700),
        "type": "PAYMENT",
        "amount": amount,
        "nameOrig": _random_account_id(),
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": round(old_balance - amount, 2),
        "nameDest": _random_account_id("M"),  # merchants in PaySim start with M
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


def _clean_cash_out() -> dict:
    amount = round(random.uniform(10, 10_000), 2)
    old_balance = round(amount + random.uniform(500, 200_000), 2)
    return {
        "step": random.randint(380, 700),
        "type": "CASH_OUT",
        "amount": amount,
        "nameOrig": _random_account_id(),
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": round(old_balance - amount, 2),
        "nameDest": _random_account_id(),
        "oldbalanceDest": round(random.uniform(0, 50_000), 2),
        "newbalanceDest": round(random.uniform(0, 50_000), 2),
    }


def _fraud_transfer() -> dict:
    """Fraud archetype: TRANSFER that drains the originator to zero."""
    amount = round(random.uniform(1000, 1_000_000), 2)
    return {
        "step": random.randint(380, 700),
        "type": "TRANSFER",
        "amount": amount,
        "nameOrig": _random_account_id(),
        "oldbalanceOrg": amount,
        "newbalanceOrig": 0.0,
        "nameDest": _random_account_id(),
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
    }


class FraudClient(HttpUser):
    """A simulated client hitting /predict with realistic traffic."""
    wait_time = between(0.001, 0.01)  # near-zero think time -> max throughput

    @task(18)  # ~90% of traffic
    def clean_payment(self):
        self.client.post("/predict", json=_clean_payment(), name="/predict [payment]")

    @task(1)  # ~5%
    def clean_cash_out(self):
        self.client.post("/predict", json=_clean_cash_out(), name="/predict [cash_out]")

    @task(1)  # ~5% fraud
    def fraud_transfer(self):
        self.client.post("/predict", json=_fraud_transfer(), name="/predict [fraud]")