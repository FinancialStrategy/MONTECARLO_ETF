# Institutional Portfolio Analytics Platform

This project is a modular Streamlit Cloud-ready analytics platform featuring:
- Cholesky-based correlated Monte Carlo simulation
- True Black-Litterman posterior return/covariance integration
- Efficient frontier using posterior returns
- Tracking error optimization
- Active risk budgeting
- Benchmark-relative efficient frontier
- Rolling relative VaR / CVaR / ES
- Regime detection dashboard
- Downloadable Excel and branded PDF reports

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
- Upload the full folder to GitHub
- Set `app.py` as the entry point
- Use the included `requirements.txt`
