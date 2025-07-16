"""
Custom transformers implementation to avoid scikit-learn dependency in base model.
This module provides a minimal implementation of TransformedTargetRegressor
that transforms the target variable before fitting and inverse transforms
predictions, without requiring scikit-learn. The rest of the interface is just
calling the underlying regressor methods directly.
"""

from typing import Any, Callable


class TransformedTargetRegressor:
    """
    Minimal implementation of a transformer for target values.

    This class wraps a regressor and applies a transform function to the target
    values (y) before fitting, and applies an inverse transform to the predictions.
    All other method calls are forwarded directly to the wrapped regressor.
    """

    def __init__(self, regressor: Any, func: Callable, inverse_func: Callable) -> None:
        self.regressor = regressor
        self.func = func
        self.inverse_func = inverse_func
        self._fitted = False

    def fit(self, X: Any, y: Any, **fit_params) -> "TransformedTargetRegressor":  # noqa: N803
        y_transformed = self.func(y)
        self.regressor.fit(X, y_transformed, **fit_params)
        self._fitted = True
        return self

    def predict(self, X: Any) -> Any:  # noqa: N803
        if not self._fitted:
            raise ValueError("This TransformedTargetRegressor is not fitted yet")

        predictions = self.regressor.predict(X)
        return self.inverse_func(predictions)

    def __getattr__(self, name: str) -> Any:
        # Only forward to regressor for non-special attributes to avoid recursion
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
        return getattr(self.regressor, name)

    def __call__(self, *args, **kwargs):
        # forward calls to the underlying regressor.
        if callable(self.regressor):
            return self.regressor.__call__(*args, **kwargs)

        raise TypeError("The underlying regressor is not callable")
