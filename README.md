# StockMind — Forecast-Driven Inventory Optimization

> From demand signal to shelf replenishment plan, automatically. StockMind runs a pooled ML demand forecaster across SKUs, converts forecasts into operations-research inventory policies, and validates them through a 26-week replenishment simulation — policies that work in practice, not just on paper.

## What StockMind Does

- **Pooled demand forecasting** — single ML model across all SKUs with calendar + lag features
- **Baseline benchmarking** — validated against seasonal-naive with wMAPE; beats baseline or warns
- **Safety stock optimization** — newsvendor model with configurable service-level targets
- **EOQ & reorder points** — Economic Order Quantity with lead-time-adjusted ROP
- **Replenishment simulation** — 26-week replay with lead-time pipelines measuring actual fill rate

## Architecture

```
Historical Sales Data (SKU x Date x Quantity)
    └─> PooledForecaster    (LightGBM, calendar + lag features)
    └─> BaselineBenchmark   (seasonal-naive wMAPE comparison)
    └─> PolicyEngine        (newsvendor, EOQ, ROP)
    └─> ReplenishSim        (26-week replay, fill-rate measurement)
    └─> DAGRunner           (pipeline.json declarative execution)
```

## Quickstart

```bash
python -m dio demo                    # run full forecast -> policy -> simulation pipeline
python -m dio run --pipeline pipeline.json
```

## Test

```bash
python tests/test_smoke.py
```

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
