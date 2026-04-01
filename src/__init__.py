"""
Yakub Promotion Analysis Package

This package provides tools for analyzing staff promotion patterns
and building predictive models for promotion eligibility at Yakub Trading Group.

Modules:
    data_processing: Functions for loading and cleaning data
    feature_engineering: Functions for creating new features
    models: Machine learning model training functions
    evaluation: Model evaluation and metrics functions
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_processing import load_and_clean_data, handle_missing_values
from .feature_engineering import create_features, encode_categorical
from .models import train_gradient_boosting, train_random_forest
from .evaluation import evaluate_model, plot_roc_curve

__all__ = [
    "load_and_clean_data",
    "handle_missing_values",
    "create_features",
    "encode_categorical",
    "train_gradient_boosting",
    "train_random_forest",
    "evaluate_model",
    "plot_roc_curve",
]
