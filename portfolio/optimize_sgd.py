import numpy as np

def project_to_simplex(w):
    """Project vector onto probability simplex."""
    if len(w) == 0:
        raise ValueError("Input vector w is empty in project_to_simplex.")
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    nz = np.nonzero(u * np.arange(1, len(w)+1) > (cssv - 1))[0]
    if len(nz) == 0:
        raise ValueError("No valid projection found in project_to_simplex. Check input values.")
    rho = nz[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)

def sgd_optimize(price_df, lr=0.01, epochs=500, λ=0.1, batch_size=20, verbose=False):
    # 1) prep
    rets = price_df.pct_change().dropna().values
    if rets.shape[0] == 0:
        raise ValueError("Not enough price data for optimization. Please select a longer date range or different tickers.")
    mu_full = rets.mean(axis=0) * 252
    Sigma_full = np.cov(rets.T) * 252
    n = mu_full.shape[0]
    w = np.ones(n) / n
    losses = []

    m = rets.shape[0]
    for epoch in range(epochs):
        idx = np.random.permutation(m)
        for start in range(0, m, batch_size):
            batch = rets[idx[start:start+batch_size]]
            mu_b = batch.mean(axis=0) * 252
            Sigma_b = np.cov(batch.T) * 252
            grad = -(mu_b - λ * (Sigma_b @ w))
            w -= lr * grad
            w = project_to_simplex(w)

        obj = w.dot(mu_full) - λ/2 * w.dot(Sigma_full @ w)
        losses.append(-obj)
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, loss {losses[-1]:.4f}")

    return w, losses
