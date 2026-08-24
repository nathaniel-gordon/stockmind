"""Smoke test: python tests/test_smoke.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dio.datagen import generate_demand
from dio.forecast import DemandForecaster
from dio.policy import build_policies, standard_normal_loss
from dio.simulate import simulate_policy


def main() -> None:
    assert abs(standard_normal_loss(0) - 0.3989) < 1e-3
    assert standard_normal_loss(2) < 0.01

    df = generate_demand(weeks=156, seed=8)
    fc = DemandForecaster(seed=8)
    m = fc.fit(df)
    assert m["wmape_pct"] < m["wmape_seasonal_naive"], m
    assert m["wmape_pct"] < 18, m

    fut = fc.weekly_forecast(df, horizon=26)
    assert len(fut) == 26 * df["sku"].nunique()
    assert (fut["forecast"] >= 0).all()

    pol = build_policies(fut)
    assert (pol["safety_stock"] > 0).all()
    assert (pol["reorder_point"] > pol["safety_stock"]).all()
    assert (pol["projected_fill_rate"] > 0.85).all()

    sim, traj = simulate_policy(fut, pol, seed=8)
    gap = (sim["fill_rate"] - sim["service_target"])
    assert gap.mean() > -0.03, f"policies badly miss service targets:\n{sim}"
    assert (sim["fill_rate"] > 0.85).all(), sim
    assert sim["orders_placed"].sum() > 0
    print(f"OK - wmape={m['wmape_pct']}% mean fill={sim['fill_rate'].mean():.3f} "
          f"(targets {sim['service_target'].mean():.3f})")


if __name__ == "__main__":
    main()


def test_smoke():
    main()
