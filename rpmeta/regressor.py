"""
Custom transformers implementation to avoid scikit-learn dependency in base model.
This module provides a minimal implementation of TransformedTargetRegressor
that transforms the target variable before fitting and inverse transforms
predictions, without requiring scikit-learn. The rest of the interface is just
calling the underlying regressor methods directly.
"""

import logging
import os
import atexit
import tempfile

from typing import Any, Callable

import joblib

logger = logging.getLogger(__name__)


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
        self._mmap_file = None
        atexit.register(self._cleanup_mmap_file)

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
    
    def _cleanup_mmap_file(self) -> None:
        if not self._mmap_file or not os.path.exists(self._mmap_file):
            return
        
        try:
            logger.debug("Cleaning up memory-mapped file: %s", self._mmap_file)
            os.remove(self._mmap_file)
        except OSError as e:
            logger.error("Error cleaning up memory-mapped file: %s", e)

    def memory_mapped_regressor(self) -> None:
        """
        Creates a memory-mapped version of the regressor to save memory.
        If really the regressor is large enough to allocate this on disk, expect significant
        performance degradation.
        """
        self._cleanup_mmap_file()

        logger.debug("Creating a temporary file for memory-mapped regressor")
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            self._mmap_file = tmp_file.name

        logger.debug("Creating memory-mapped version of the regressor at %s", self._mmap_file)
        joblib.dump(self.regressor, self._mmap_file)
        self.regressor = joblib.load(self._mmap_file, mmap_mode='r')

    def __getattr__(self, name: str) -> Any:
        # Only forward to regressor for non-special attributes to avoid recursion
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

        logger.debug("Forwarding attribute '%s' to the underlying regressor", name)
        return getattr(self.regressor, name)

    def __call__(self, *args, **kwargs):
        # forward calls to the underlying regressor.
        if callable(self.regressor):
            logger.debug("Calling the underlying regressor with args: %s, kwargs: %s", args, kwargs)
            return self.regressor.__call__(*args, **kwargs)

        raise TypeError("The underlying regressor is not callable")
