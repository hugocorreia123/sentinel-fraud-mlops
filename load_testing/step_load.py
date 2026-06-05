"""Step-load test: ramp users from 50 → 500 in stages, watch for failure.

Each stage holds for 60s. Stop when failure rate or p99 latency explodes.
Run with:

    uv run --group load_test locust -f load_testing/step_load.py \\
      --host http://127.0.0.1:8000 --headless --only-summary
"""
from __future__ import annotations

from locust import LoadTestShape

from load_testing.locustfile import FraudClient  # reuse the same user behavior

assert FraudClient  # silence linter; the import registers the user class


class StepLoad(LoadTestShape):
    """Five 60-second stages: 50, 100, 200, 300, 500 concurrent users."""

    stages = [
        {"duration":  60, "users":  50, "spawn_rate": 10},
        {"duration": 120, "users": 100, "spawn_rate": 10},
        {"duration": 180, "users": 200, "spawn_rate": 20},
        {"duration": 240, "users": 300, "spawn_rate": 25},
        {"duration": 300, "users": 500, "spawn_rate": 50},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None