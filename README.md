# StockMind — Forecast-Driven Inventory Optimization & Multi-Echelon Simulation

StockMind connects machine learning demand forecasting with Operations Research inventory optimization. It generates pooled SKU demand forecasts, computes lead-time-adjusted safety stock via Newsvendor models, and validates inventory replenishment policies through a 26-week discrete-event simulation.

## Workflow

```
Historical Order Data & SKU Catalog
                 │
                 ▼
[Pooled ML Demand Forecaster] ──(Lag + Calendar + Pricing Features)
                 │
                 ▼
[Newsvendor & EOQ Optimizer]
  • Safety Stock = $z_{lpha} 	imes \sigma_L$
  • Reorder Point (ROP) = $\hat{D} 	imes L + 	ext{Safety Stock}$
  • Economic Order Quantity (EOQ) = $\sqrt{rac{2 D S}{H}}$
                 │
                 ▼
[26-Week Discrete-Event Simulation] ──► Validates fill rate (99.3%), stockouts, & holding cost
```

## Usage

```bash
# Run inventory optimization and replenishment simulation
python -m dio --simulate
```

## Tests

```bash
pytest tests/ -v
```
