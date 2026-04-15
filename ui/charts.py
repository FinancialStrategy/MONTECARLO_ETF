# ui/charts.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def weight_bar_chart(df):
    fig = px.bar(df, x="Ticker", y="Weight (%)", color="Category", text="Weight (%)", title="Portfolio Weights")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=450)
    return fig

def category_pie_chart(df):
    cat = df.groupby("Category", as_index=False)["Weight (%)"].sum()
    fig = px.pie(cat, names="Category", values="Weight (%)", hole=0.45, title="Category Allocation")
    fig.update_layout(height=430)
    return fig

def monte_carlo_paths_chart(paths, initial_investment):
    fig = go.Figure()
    idx = np.random.choice(paths.shape[0], min(80, paths.shape[0]), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(y=paths[i], mode="lines", opacity=0.16, line=dict(width=0.7), showlegend=False))
    fig.add_hline(y=initial_investment, line_dash="dash")
    fig.add_trace(go.Scatter(y=paths.mean(axis=0), mode="lines", line=dict(width=3), name="Mean Path"))
    fig.update_layout(title="Monte Carlo Paths", height=480)
    return fig

def terminal_distribution_chart(final_values, initial_investment):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_values, nbinsx=60, name="Terminal Values"))
    fig.add_vline(x=initial_investment, line_dash="dash", annotation_text="Initial")
    fig.update_layout(title="Terminal Value Distribution", height=420)
    return fig

def regime_chart(regime_df):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Annualized Rolling Return", "Annualized Rolling Volatility"))
    fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df["return_ann"] * 100, mode="lines", name="Return"), row=1, col=1)
    fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df["vol_ann"] * 100, mode="lines", name="Vol"), row=2, col=1)
    fig.update_layout(height=650, title="Regime Detection Dashboard")
    return fig
