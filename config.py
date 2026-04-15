TRADING_DAYS = 252
ROLLING_WINDOW = 63
VAR_CONFIDENCE = 0.95
DEFAULT_RISK_FREE_RATE = 0.03
DEFAULT_TAU = 0.05
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_FORECAST_DAYS = 252
DEFAULT_REGIME_WINDOW = 63
APP_TITLE = "Institutional Portfolio Analytics Platform"
APP_SUBTITLE = (
    "Black-Litterman posterior optimization, benchmark-relative efficient frontier, "
    "active risk budgeting, rolling relative tail risk, and executive reporting."
)

INVESTMENT_UNIVERSE = {
    "SPY": {"name": "SPDR S&P 500 ETF", "category": "Large Cap", "expense_ratio": 0.0945},
    "QQQ": {"name": "Invesco QQQ Trust", "category": "Tech/Growth", "expense_ratio": 0.20},
    "DIA": {"name": "SPDR Dow Jones Industrial Average", "category": "Large Cap", "expense_ratio": 0.16},
    "IWM": {"name": "iShares Russell 2000 ETF", "category": "Small Cap", "expense_ratio": 0.19},
    "MDY": {"name": "SPDR S&P MidCap 400 ETF", "category": "Mid Cap", "expense_ratio": 0.23},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "category": "Total Market", "expense_ratio": 0.03},
    "SCHD": {"name": "Schwab US Dividend Equity ETF", "category": "Dividend", "expense_ratio": 0.06},
    "XLF": {"name": "Financial Select Sector SPDR", "category": "Sector", "expense_ratio": 0.10},
    "XLV": {"name": "Health Care Select Sector SPDR", "category": "Sector", "expense_ratio": 0.10},
    "XLK": {"name": "Technology Select Sector SPDR", "category": "Sector", "expense_ratio": 0.10},
    "XLE": {"name": "Energy Select Sector SPDR", "category": "Sector", "expense_ratio": 0.10},
    "XLI": {"name": "Industrial Select Sector SPDR", "category": "Sector", "expense_ratio": 0.10},
    "EFA": {"name": "iShares MSCI EAFE ETF", "category": "International Developed", "expense_ratio": 0.33},
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets", "expense_ratio": 0.68},
    "VXUS": {"name": "Vanguard Total International Stock ETF", "category": "International", "expense_ratio": 0.07},
    "AGG": {"name": "iShares Core US Aggregate Bond ETF", "category": "Bonds", "expense_ratio": 0.04},
    "BND": {"name": "Vanguard Total Bond Market ETF", "category": "Bonds", "expense_ratio": 0.03},
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "category": "Long Term Bonds", "expense_ratio": 0.15},
    "VNQ": {"name": "Vanguard Real Estate ETF", "category": "Real Estate", "expense_ratio": 0.12},
    "GLD": {"name": "SPDR Gold Shares", "category": "Commodity", "expense_ratio": 0.40},
}

BENCHMARK_ASSUMPTIONS = {
    "Cash": 0.00,
    "SPY": 0.10,
    "QQQ": 0.15,
}
