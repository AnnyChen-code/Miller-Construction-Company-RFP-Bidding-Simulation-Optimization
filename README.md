# Miller Construction RFP Analyzer

This Streamlit app helps Miller Construction evaluate bids for this project and future RFPs using:

- Triangular distributions for uncertain project costs
- Triangular distributions for competitor bid assumptions
- Monte Carlo simulation to estimate win probability and profit tradeoffs
- A bid search grid to recommend the most attractive bid
- CSV-driven controls so you can choose how low, mode, and high are built from historical data

## Files

- `app.py`: Streamlit interface
- `simulation.py`: backend simulation and validation logic
- `project_costs_template.csv`: starter template for future projects
- `requirements.txt`: Python dependencies

## Expected CSV format

Required columns:

- `low`, `mode`, `high`

Optional columns:

- `item`
- `quantity`

The app also accepts common aliases such as:

- `min`, `most_likely`, `max`
- `optimistic`, `base`, `pessimistic`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app lets you search directly over Miller bid amounts.
- Competitors are modeled as triangular bid distributions with optional probabilities of bidding.
- When you upload the historical case CSV, you can choose which statistics from the observed data define the triangular low, mode, and high values.
- The diagnostics tab runs lightweight self-checks on the engine so the model can be sanity-checked before use.
