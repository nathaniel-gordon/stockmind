"""Pooled demand forecaster: one model across SKUs with per-SKU residual dispersion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class DemandForecaster:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = HistGradientBoostingRegressor(random_state=seed, max_iter=250,
                                                   learning_rate=0.07)
        self.sku_codes: dict[str, int] = {}
        self.resid_std: dict[str, float] = {}
        self.metrics: dict = {}

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["sku_code"] = d["sku"].map(self.sku_codes)
        d["woy_sin"] = np.sin(2 * np.pi * (d["week"] % 52) / 52)
        d["woy_cos"] = np.cos(2 * np.pi * (d["week"] % 52) / 52)
        d["week_idx"] = d["week"]
        lag = d.groupby("sku")["demand"].shift(1) if "demand" in d else np.nan
        d["lag_1"] = lag
        d["lag_52"] = d.groupby("sku")["demand"].shift(52) if "demand" in d else np.nan
        d["roll_8"] = (d.groupby("sku")["demand"].shift(1).rolling(8).mean()
                       .reset_index(level=0, drop=True)) if "demand" in d else np.nan
        return d

    FEATURES = ["sku_code", "woy_sin", "woy_cos", "week_idx", "promo", "lag_1", "lag_52",
                "roll_8"]

    def fit(self, df: pd.DataFrame, holdout_weeks: int = 12) -> dict:
        self.sku_codes = {s: i for i, s in enumerate(sorted(df["sku"].unique()))}
        d = self._features(df).dropna(subset=["lag_52", "roll_8"])
        cut = d["week"].max() - holdout_weeks
        tr, te = d[d["week"] <= cut], d[d["week"] > cut]
        self.model.fit(tr[self.FEATURES], tr["demand"])
        pred = self.model.predict(te[self.FEATURES])
        wmape = float(np.abs(te["demand"] - pred).sum() / te["demand"].sum() * 100)
        naive = te["lag_52"].to_numpy()
        wmape_naive = float(np.abs(te["demand"] - naive).sum() / te["demand"].sum() * 100)
        # per-SKU residual std on holdout (drives safety stock)
        te = te.assign(err=te["demand"] - pred)
        self.resid_std = te.groupby("sku")["err"].std().fillna(te["err"].std()).to_dict()
        self.metrics = {"wmape_pct": round(wmape, 2), "wmape_seasonal_naive": round(wmape_naive, 2),
                        "holdout_weeks": holdout_weeks}
        return self.metrics

    def weekly_forecast(self, df: pd.DataFrame, horizon: int = 26) -> pd.DataFrame:
        """Iterative forecast per SKU for the next `horizon` weeks (no promos assumed)."""
        out = []
        for sku, g in df.groupby("sku"):
            g = g.sort_values("week")
            hist = g["demand"].tolist()
            weeks = g["week"].tolist()
            static = g.iloc[-1]
            for h in range(1, horizon + 1):
                w = weeks[-1] + 1
                row = {"sku": sku, "week": w, "promo": 0,
                       "sku_code": self.sku_codes.get(sku, 0),
                       "woy_sin": np.sin(2 * np.pi * (w % 52) / 52),
                       "woy_cos": np.cos(2 * np.pi * (w % 52) / 52),
                       "week_idx": w, "lag_1": hist[-1],
                       "lag_52": hist[-52] if len(hist) >= 52 else np.mean(hist),
                       "roll_8": float(np.mean(hist[-8:]))}
                pred = float(self.model.predict(pd.DataFrame([row])[self.FEATURES])[0])
                pred = max(pred, 0.0)
                out.append({"sku": sku, "week": w, "forecast": round(pred, 1),
                            "sigma": round(float(self.resid_std.get(sku, pred * 0.2)), 1),
                            "unit_cost": static["unit_cost"],
                            "lead_time_wk": int(static["lead_time_wk"]),
                            "service_target": float(static["service_target"])})
                hist.append(pred)
                weeks.append(w)
        return pd.DataFrame(out)
