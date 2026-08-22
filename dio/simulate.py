"""Replenishment simulator: replay demand under the (ROP, EOQ) policy and measure service."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_policy(forecast: pd.DataFrame, policies: pd.DataFrame, seed: int = 42,
                    demand_noise: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly simulation with in-transit pipeline; realized demand = forecast * lognoise."""
    rng = np.random.default_rng(seed)
    results, trajectories = [], []
    pol = policies.set_index("sku")
    for sku, g in forecast.groupby("sku"):
        g = g.sort_values("week")
        p = pol.loc[sku]
        on_hand = float(p["reorder_point"] + p["order_qty_eoq"] / 2)
        pipeline: list[tuple[int, float]] = []      # (arrival_week_index, qty)
        served = demanded = 0.0
        stockout_weeks = orders = 0
        for i, (_, row) in enumerate(g.iterrows()):
            arrived = sum(q for t, q in pipeline if t == i)
            pipeline = [(t, q) for t, q in pipeline if t > i]
            on_hand += arrived
            demand = max(float(rng.normal(row["forecast"], demand_noise * max(row["forecast"], 1))), 0)
            sold = min(on_hand, demand)
            served += sold
            demanded += demand
            if demand > on_hand:
                stockout_weeks += 1
            on_hand -= sold
            inv_position = on_hand + sum(q for _, q in pipeline)
            if inv_position <= p["reorder_point"]:
                pipeline.append((i + int(p["lead_wk"]), float(p["order_qty_eoq"])))
                orders += 1
            trajectories.append({"sku": sku, "week": int(row["week"]), "on_hand": round(on_hand, 0),
                                 "demand": round(demand, 0), "ordered": arrived > 0})
        results.append({"sku": sku, "fill_rate": round(served / max(demanded, 1), 4),
                        "service_target": float(p["service_target"]),
                        "stockout_weeks": stockout_weeks, "orders_placed": orders,
                        "avg_on_hand": round(float(np.mean([t['on_hand'] for t in trajectories
                                                            if t['sku'] == sku])), 0)})
    return pd.DataFrame(results), pd.DataFrame(trajectories)
