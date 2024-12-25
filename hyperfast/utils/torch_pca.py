import torch

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
    return u, v


class TorchPCA:
    def __init__(self, n_components=None, fit="full"):
        self.n_components = n_components
        self.fit = fit

    def _fit(self, X):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        n_samples, n_features = X.shape
        if n_components > min(X.shape):
            raise ValueError(
                f"n_components should be <= min(n_samples: {n_samples}, n_features: {n_features})"
            )

        self.mean_ = torch.mean(X, axis=0)
        X -= self.mean_

        if self.fit == "full":
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
        elif self.fit == "lowrank":
            U, S, Vt = torch.pca_lowrank(X)

        self.components_ = Vt[:n_components]
        self.n_components_ = n_components

        return U, S, Vt

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        assert self.mean_ is not None
        X -= self.mean_
        return torch.matmul(X, self.components_.T)

    def fit_transform(self, X):
        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]
        U *= S[: self.n_components_]
        return U

