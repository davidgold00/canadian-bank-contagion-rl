from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def fit_regime_model(X, n_regimes=5):
    scaler=StandardScaler(); Z=scaler.fit_transform(X.fillna(0)); model=GaussianMixture(n_components=n_regimes, random_state=42).fit(Z); return scaler, model
