"""Inventory policy: newsvendor safety stock, EOQ, reorder points, cost projection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .datagen import HOLDING_RATE_ANNUAL, ORDER_COST


def standard_normal_loss(z: float) -> float:
    """E[max(Z - z, 0)] for standard normal (unit normal loss function)."""
    return float(stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))


def build_policies(forecast: pd.DataFrame) -> pd.DataFrame:
    """One (ROP, EOQ, SS) policy per SKU from its forward forecast."""
    rows = []
    for sku, g in forecast.groupby("sku"):
        mu_w = float(g["forecast"].mean())
        sigma_w = float(g["sigma"].iloc[0])
        lead = int(g["lead_time_wk"].iloc[0])
        svc = float(g["service_target"].iloc[0])
        cost = float(g["unit_cost"].iloc[0])

        mu_l = mu_w * lead
        sigma_l = sigma_w * np.sqrt(lead)
        z = float(stats.norm.ppf(svc))
        ss = z * sigma_l
        rop = mu_l + ss

        annual_demand = mu_w * 52
        holding_per_unit = cost * HOLDING_RATE_ANNUAL
        eoq = float(np.sqrt(2 * annual_demand * ORDER_COST / holding_per_unit))

        # expected units short per replenishment cycle (normal loss)
        exp_short_cycle = sigma_l * standard_normal_loss(z)
        cycles_per_year = annual_demand / eoq
        fill_rate_proj = 1 - (exp_short_cycle * cycles_per_year) / annual_demand

        avg_inventory = eoq / 2 + ss
        rows.append({"sku": sku, "mu_weekly": round(mu_w, 1), "sigma_weekly": round(sigma_w, 1),
                     "lead_wk": lead, "service_target": svc, "z": round(z, 2),
                     "safety_stock": int(np.ceil(ss)), "reorder_point": int(np.ceil(rop)),
                     "order_qty_eoq": int(np.ceil(eoq)),
                     "projected_fill_rate": round(float(fill_rate_proj), 4),
                     "avg_inventory_units": int(avg_inventory),
                     "annual_holding_cost": round(avg_inventory * holding_per_unit, 0),
                     "annual_order_cost": round(cycles_per_year * ORDER_COST, 0)})
    return pd.DataFrame(rows)
