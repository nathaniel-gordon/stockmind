"""Synthetic multi-SKU weekly demand with trend, seasonality, promos + SKU economics."""
from __future__ import annotations

import numpy as np
import pandas as pd

SKUS = [
    # (sku, base_demand, season_amp, trend_per_wk, unit_cost, lead_time_wk, service_target)
    ("SKU-001 kettle", 120, 0.25, 0.15, 22.0, 2, 0.95),
    ("SKU-002 toaster", 95, 0.2, 0.05, 18.0, 3, 0.95),
    ("SKU-003 blender", 150, 0.3, 0.3, 35.0, 2, 0.97),
    ("SKU-004 air fryer", 210, 0.35, 0.8, 62.0, 4, 0.97),
    ("SKU-005 coffee maker", 180, 0.28, 0.2, 48.0, 2, 0.97),
    ("SKU-006 vacuum", 80, 0.15, 0.1, 95.0, 5, 0.93),
    ("SKU-007 iron", 60, 0.1, -0.05, 15.0, 3, 0.9),
    ("SKU-008 heater", 140, 0.9, 0.0, 40.0, 4, 0.95),
    ("SKU-009 fan", 130, 0.85, 0.05, 28.0, 4, 0.95),
    ("SKU-010 humidifier", 70, 0.5, 0.12, 33.0, 3, 0.93),
    ("SKU-011 scale", 55, 0.05, 0.02, 12.0, 2, 0.9),
    ("SKU-012 lamp", 100, 0.12, 0.06, 20.0, 2, 0.93),
]

HOLDING_RATE_ANNUAL = 0.25    # of unit cost
ORDER_COST = 140.0            # fixed cost per replenishment order


def generate_demand(weeks: int = 156, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sku, base, amp, trend, cost, lead, svc in SKUS:
        phase = rng.uniform(0, 2 * np.pi)
        for w in range(weeks):
            season = 1 + amp * np.sin(2 * np.pi * w / 52 + phase)
            promo = int(rng.random() < 0.08)
            mu = max((base + trend * w) * season * (1.35 if promo else 1.0), 1.0)
            demand = max(int(rng.normal(mu, 0.14 * mu)), 0)
            rows.append({"week": w, "sku": sku, "demand": demand, "promo": promo,
                         "unit_cost": cost, "lead_time_wk": lead, "service_target": svc})
    return pd.DataFrame(rows)
