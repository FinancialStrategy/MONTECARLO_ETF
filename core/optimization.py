def get_weights_from_method(
    allocation_method: str,
    selected_etfs: list[str],
    optimizer: PortfolioOptimizer,
    benchmark_series: pd.Series | None,
    custom_weights: list[float] | None,
) -> np.ndarray:
    n = len(selected_etfs)

    if allocation_method == "Equal Weight":
        return np.repeat(1.0 / n, n)

    if allocation_method == "Optimized (Max Sharpe)":
        return normalize_weights(optimizer.optimize("max_sharpe"))

    if allocation_method == "Optimized (Min Volatility)":
        return normalize_weights(optimizer.optimize("min_volatility"))

    if allocation_method == "Tracking Error Optimization":
        if benchmark_series is None or benchmark_series.empty:
            st.warning("Benchmark data unavailable. Falling back to equal weights.")
            return np.repeat(1.0 / n, n)

        try:
            return normalize_weights(optimizer.optimize_tracking_error(benchmark_series))
        except Exception as e:
            st.warning(
                f"Tracking error optimization failed. Falling back to equal weights. Details: {e}"
            )
            return np.repeat(1.0 / n, n)

    if allocation_method == "Custom Weights":
        if custom_weights is None:
            st.warning("Custom weights unavailable. Falling back to equal weights.")
            return np.repeat(1.0 / n, n)
        return normalize_weights(np.array(custom_weights, dtype=float))

    st.warning("Unknown allocation method detected. Falling back to equal weights.")
    return np.repeat(1.0 / n, n)
