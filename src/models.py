"""Models and their hyperparameter grids: name -> (constructor, grid searched on validation).

All models predict the 4 legs at once (4 independent binary outputs).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

N_KERNEL = 10_000  # training windows for kernel methods (an n x n kernel matrix: 70k windows would need 40 GB)


class Subsampled(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on `n_train` training windows taken evenly in time (all windows if fewer)."""

    def __init__(self, estimator, n_train=N_KERNEL):
        self.estimator, self.n_train = estimator, n_train

    def fit(self, X, y):
        keep = np.linspace(0, len(X) - 1, min(self.n_train, len(X))).astype(int)
        self.estimator_ = clone(self.estimator).fit(X[keep], y[keep])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class KernelRidgeClassifier(ClassifierMixin, BaseEstimator):
    """Kernel ridge (RBF) on targets -1/+1, one output per leg, contact if the output is > 0.

    The non-linear counterpart of RidgeClassifier: same loss and penalty, RBF kernel instead of a hyperplane.
    sklearn's KernelRidge has no bias term: the mean target is subtracted before fitting and added back when
    predicting, which gives it one (as RidgeClassifier's intercept). gamma = gamma_scale / n_features.
    """

    def __init__(self, alpha=1.0, gamma_scale=1.0):
        self.alpha, self.gamma_scale = alpha, gamma_scale

    def fit(self, X, y):
        t = 2.0 * y - 1
        self.offset_ = t.mean(0)
        self.krr_ = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma_scale / X.shape[1])
        self.krr_.fit(X.astype(np.float64), t - self.offset_)
        self.classes_ = np.arange(y.shape[1])  # multi-label: one output per leg (as RidgeClassifier)
        return self

    def predict(self, X):
        return (self.krr_.predict(X.astype(np.float64)) + self.offset_ > 0).astype(int)


class RBFSVMClassifier(ClassifierMixin, BaseEstimator):
    """Soft-margin SVM with RBF kernel, one per leg (4 independent binary SVMs), contact if the output is > 0.

    Same kernel as KernelRidgeClassifier, hinge loss instead of squared loss. gamma = gamma_scale / n_features.
    """

    def __init__(self, C=1.0, gamma_scale=1.0):
        self.C, self.gamma_scale = C, gamma_scale

    def fit(self, X, y):
        self.svms_ = [SVC(C=self.C, kernel="rbf", gamma=self.gamma_scale / X.shape[1], cache_size=1000).fit(X, y[:, k])
                      for k in range(y.shape[1])]
        self.classes_ = np.arange(y.shape[1])  # multi-label: one output per leg (as RidgeClassifier)
        return self

    def predict(self, X):
        return np.stack([svm.predict(X) for svm in self.svms_], 1)


MODELS = {
    # ignores the input, always predicts each leg's most frequent training label
    "majority": (lambda: DummyClassifier(strategy="most_frequent"), {}),

    # least squares on targets -1/+1 with an L2 penalty alpha*||w||^2, predicts contact if score > 0.
    # Inputs standardized first so the penalty treats every channel the same, whatever its unit.
    "ridge": (lambda: make_pipeline(StandardScaler(), RidgeClassifier()),
              {"ridgeclassifier__alpha": np.logspace(-3, 5, 9)}),

    # the same ridge, trained on the kernel methods' 10k windows: shows what the subsample costs
    "ridge_10k": (lambda: Subsampled(make_pipeline(StandardScaler(), RidgeClassifier())),
                  {"estimator__ridgeclassifier__alpha": np.logspace(-3, 5, 9)}),

    # same loss and penalty, RBF kernel (non-linear boundary), on the same 10k windows
    "krr": (lambda: Subsampled(make_pipeline(StandardScaler(), KernelRidgeClassifier())),
            {"estimator__kernelridgeclassifier__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
             "estimator__kernelridgeclassifier__gamma_scale": [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]}),

    # same hyperplane as ridge, hinge loss instead of squared loss (C weighs the loss, i.e. C ~ 1/alpha);
    # one binary SVM per leg; full training set, like ridge
    "svm_linear": (lambda: make_pipeline(StandardScaler(), MultiOutputClassifier(
                       LinearSVC(loss="hinge", dual=True, max_iter=1_000_000))),  # large C converges slowly
                   {"multioutputclassifier__estimator__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}),

    # the same linear SVM on the kernel methods' 10k windows: boundary comparison with svm_rbf on identical data
    "svm_linear_10k": (lambda: Subsampled(make_pipeline(StandardScaler(), MultiOutputClassifier(
                           LinearSVC(loss="hinge", dual=True, max_iter=1_000_000)))),
                       {"estimator__multioutputclassifier__estimator__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}),

    # same RBF kernel as krr, hinge loss; on the same 10k windows
    "svm_rbf": (lambda: Subsampled(make_pipeline(StandardScaler(), RBFSVMClassifier())),
                {"estimator__rbfsvmclassifier__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                 "estimator__rbfsvmclassifier__gamma_scale": [0.1, 0.3, 1.0, 3.0]}),
}
