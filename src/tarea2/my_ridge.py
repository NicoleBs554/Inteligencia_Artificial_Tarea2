import numpy as np


class MyRidge:
    """Regresión Ridge con solución cerrada (solo usa numpy).

    theta = (Xc^T Xc + alpha * I)^(-1) Xc^T yc
    donde Xc y yc son X y y con la media restada (datos centrados).
    Así el intercepto no se penaliza.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # este es el lambda de las notas

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # 1. Medias calculadas SOLO con los datos recibidos en fit
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        # 2. Centrar X y y (restar las medias)
        Xc = X - self.X_mean_
        yc = y - self.y_mean_

        # 3. Resolver (Xc^T Xc + alpha I) theta = Xc^T yc
        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        b = Xc.T @ yc
        self.coef_ = np.linalg.solve(A, b)

        # 4. Recuperar el intercepto: theta0 = y_mean - x_mean^T theta
        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # Siempre se usan las medias guardadas en fit, nunca las de X nuevo
        return (X - self.X_mean_) @ self.coef_ + self.y_mean_

    def score(self, X, y):
        """Coeficiente de determinación R^2 (igual que sklearn)."""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    def rmse(self, X, y):
        """Raíz del error cuadrático medio."""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
