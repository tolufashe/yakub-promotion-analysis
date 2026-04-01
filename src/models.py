"""
Machine Learning Models Module

This module contains functions for training and saving classification models
for the Yakub Trading Group promotion prediction task.

Functions:
    train_logistic_regression: Train logistic regression model
    train_decision_tree: Train decision tree classifier
    train_random_forest: Train random forest classifier
    train_gradient_boosting: Train gradient boosting classifier
    save_model: Save trained model to disk
    load_model: Load trained model from disk
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
import warnings

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Logistic Regression is a linear model that estimates probabilities using
    the logistic function. It's highly interpretable and serves as a good baseline.
    
    Args:
        X_train: Training features (scaled)
        y_train: Training target
        class_weight: 'balanced' to handle class imbalance
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Trained LogisticRegression model
        
    Example:
        >>> model = train_logistic_regression(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    print("Training Logistic Regression...")
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✓ Model trained successfully")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Class weight: {class_weight}")
    
    return model


def train_decision_tree(
    X_train: np.ndarray,
    y_train: pd.Series,
    max_depth: int = 5,
    class_weight: str = 'balanced',
    random_state: int = 42,
    **kwargs
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.
    
    Decision Trees split data based on feature thresholds, creating
    interpretable rules. Limited depth prevents overfitting.
    
    Args:
        X_train: Training features
        y_train: Training target
        max_depth: Maximum tree depth (limits overfitting)
        class_weight: 'balanced' to handle class imbalance
        random_state: Random seed
        **kwargs: Additional parameters for DecisionTreeClassifier
        
    Returns:
        Trained DecisionTreeClassifier
        
    Example:
        >>> model = train_decision_tree(X_train, y_train, max_depth=5)
        >>> predictions = model.predict(X_test)
    """
    print("Training Decision Tree...")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✓ Model trained successfully")
    print(f"  Max depth: {max_depth}")
    print(f"  Number of leaves: {model.get_n_leaves()}")
    
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 14,
    class_weight: str = 'balanced',
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Random Forest is an ensemble of decision trees using bagging.
    It reduces overfitting and provides feature importance scores.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree
        class_weight: 'balanced' to handle class imbalance
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 uses all cores)
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained RandomForestClassifier
        
    Example:
        >>> model = train_random_forest(X_train, y_train, n_estimators=100)
        >>> predictions = model.predict(X_test)
    """
    print("Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✓ Model trained successfully")
    print(f"  Trees: {n_estimators}")
    print(f"  Max depth: {max_depth}")
    
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.
    
    Gradient Boosting builds trees sequentially, with each tree correcting
    errors of the previous ones. Often achieves the highest accuracy.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting stages
        max_depth: Maximum depth of each tree
        learning_rate: Shrinks contribution of each tree
        random_state: Random seed
        **kwargs: Additional parameters for GradientBoostingClassifier
        
    Returns:
        Trained GradientBoostingClassifier
        
    Example:
        >>> model = train_gradient_boosting(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    print("Training Gradient Boosting...")
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✓ Model trained successfully")
    print(f"  Estimators: {n_estimators}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max depth: {max_depth}")
    
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    models_to_train: list = None
) -> Dict[str, Any]:
    """
    Train multiple models and return them in a dictionary.
    
    Args:
        X_train: Training features
        y_train: Training target
        models_to_train: List of model names to train (default: all)
        
    Returns:
        Dictionary of trained models
        
    Example:
        >>> models = train_all_models(X_train, y_train)
        >>> for name, model in models.items():
        ...     print(f"{name}: trained")
    """
    if models_to_train is None:
        models_to_train = ['logistic_regression', 'decision_tree', 
                          'random_forest', 'gradient_boosting']
    
    models = {}
    
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    if 'logistic_regression' in models_to_train:
        models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    
    if 'decision_tree' in models_to_train:
        models['Decision Tree'] = train_decision_tree(X_train, y_train)
    
    if 'random_forest' in models_to_train:
        models['Random Forest'] = train_random_forest(X_train, y_train)
    
    if 'gradient_boosting' in models_to_train:
        models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    print("\n" + "="*60)
    print(f"ALL {len(models)} MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    return models


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        
    Example:
        >>> save_model(model, 'models/gb_model.pkl')
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
        
    Example:
        >>> model = load_model('models/gb_model.pkl')
        >>> predictions = model.predict(X_test)
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def get_feature_importance(
    model: Any,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model (RandomForest or GradientBoosting)
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
        
    Example:
        >>> importance = get_feature_importance(rf_model, feature_names)
        >>> print(importance.head(10))
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance
    else:
        raise ValueError("Model does not have feature_importances_ attribute")


def get_model_params(model: Any) -> Dict[str, Any]:
    """
    Get model parameters for documentation.
    
    Args:
        model: Trained model
        
    Returns:
        Dictionary of model parameters
    """
    return model.get_params()


if __name__ == "__main__":
    print("Machine Learning Models Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - train_logistic_regression(X_train, y_train)")
    print("  - train_decision_tree(X_train, y_train)")
    print("  - train_random_forest(X_train, y_train)")
    print("  - train_gradient_boosting(X_train, y_train)")
    print("  - train_all_models(X_train, y_train)")
    print("  - save_model(model, filepath)")
    print("  - load_model(filepath)")
    print("  - get_feature_importance(model, feature_names)")
