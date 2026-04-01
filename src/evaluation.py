"""
Model Evaluation Module

This module contains functions for evaluating classification models
and generating performance reports for the Yakub Trading Group
promotion prediction task.

Functions:
    evaluate_model: Calculate comprehensive evaluation metrics
    plot_roc_curve: Plot ROC curves for one or more models
    plot_confusion_matrix: Visualize confusion matrix
    compare_models: Compare multiple models side by side
    check_overfitting: Detect overfitting by comparing train/test performance
    generate_classification_report: Detailed classification report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for a classification model.
    
    Metrics calculated:
    - Accuracy: Overall correctness
    - Precision: Positive predictive value
    - Recall (Sensitivity): True positive rate
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for positive class)
        model_name: Name of the model for display
        
    Returns:
        Dictionary of evaluation metrics
        
    Example:
        >>> metrics = evaluate_model(y_test, predictions, probabilities, "Random Forest")
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    """
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return metrics


def plot_roc_curve(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for one or more models.
    
    Args:
        models_dict: Dictionary of {model_name: (y_pred, y_prob)}
        y_true: True labels
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Example:
        >>> models = {
        ...     "Random Forest": (rf_pred, rf_prob),
        ...     "Gradient Boosting": (gb_pred, gb_prob)
        ... }
        >>> plot_roc_curve(models, y_test)
    """
    plt.figure(figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF7043', '#9C27B0', '#FFC107']
    
    for i, (name, (y_pred, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.plot(
            fpr, tpr,
            label=f'{name} (AUC = {auc:.3f})',
            color=colors[i % len(colors)],
            linewidth=2
        )
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Example:
        >>> plot_confusion_matrix(y_test, predictions, "Random Forest")
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Promoted', 'Promoted'],
        yticklabels=['Not Promoted', 'Promoted'],
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def compare_models(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    sort_by: str = 'roc_auc'
) -> pd.DataFrame:
    """
    Compare multiple models and return a comparison DataFrame.
    
    Args:
        models_dict: Dictionary of {model_name: (y_pred, y_prob)}
        y_true: True labels
        sort_by: Metric to sort by ('roc_auc', 'f1_score', 'accuracy')
        
    Returns:
        DataFrame with model comparison
        
    Example:
        >>> comparison = compare_models(models, y_test)
        >>> print(comparison)
    """
    results = []
    
    for name, (y_pred, y_prob) in models_dict.items():
        metrics = evaluate_model(y_true, y_pred, y_prob, name)
        results.append(metrics)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by specified metric
    df_results = df_results.sort_values(sort_by, ascending=False).reset_index(drop=True)
    
    # Select and rename columns for display
    display_cols = ['model_name', 'accuracy', 'roc_auc', 'f1_score', 'recall', 'precision']
    df_display = df_results[display_cols].copy()
    df_display.columns = ['Model', 'Accuracy', 'ROC-AUC', 'F1 Score', 'Recall', 'Precision']
    
    return df_display


def check_overfitting(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Detect overfitting by comparing training and testing performance.
    
    A large gap between train and test AUC (> 0.05) suggests overfitting.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        model_name: Name of the model
        
    Returns:
        Dictionary with train/test metrics and gap
        
    Example:
        >>> overfit_check = check_overfitting(model, X_train, y_train, X_test, y_test)
        >>> print(f"Gap: {overfit_check['auc_gap']:.4f}")
    """
    # Training predictions
    train_prob = model.predict_proba(X_train)[:, 1]
    train_pred = (train_prob >= 0.5).astype(int)
    
    # Test predictions
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)
    
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    
    # Calculate gaps
    auc_gap = train_auc - test_auc
    f1_gap = train_f1 - test_f1
    
    results = {
        'model_name': model_name,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'auc_gap': auc_gap,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'f1_gap': f1_gap,
        'overfitting_detected': auc_gap > 0.05
    }
    
    return results


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Formatted classification report string
        
    Example:
        >>> report = generate_classification_report(y_test, predictions, "RF")
        >>> print(report)
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Not Promoted', 'Promoted'],
        digits=4
    )
    
    output = f"\n{'='*60}\n"
    output += f"  CLASSIFICATION REPORT: {model_name}\n"
    output += f"{'='*60}\n"
    output += report
    
    return output


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart comparing models across multiple metrics.
    
    Args:
        comparison_df: DataFrame from compare_models()
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    metrics = ['Accuracy', 'ROC-AUC', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF7043', '#9C27B0']
    
    for i, row in comparison_df.iterrows():
        values = [row['Accuracy'], row['ROC-AUC'], row['F1 Score']]
        ax.bar(x + i*width, values, width, label=row['Model'], 
               color=colors[i % len(colors)], alpha=0.85)
    
    ax.set_xticks(x + width * (len(comparison_df) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def print_summary_report(
    comparison_df: pd.DataFrame,
    best_model_name: str
) -> None:
    """
    Print a formatted summary report.
    
    Args:
        comparison_df: DataFrame from compare_models()
        best_model_name: Name of the best performing model
    """
    print("\n" + "="*70)
    print("                    MODEL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 Model Comparison:")
    print(comparison_df.round(4).to_string(index=False))
    
    best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC:  {best_row['ROC-AUC']:.4f}")
    print(f"   F1 Score: {best_row['F1 Score']:.4f}")
    print(f"   Recall:   {best_row['Recall']:.4f}")
    print(f"   Accuracy: {best_row['Accuracy']:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - evaluate_model(y_true, y_pred, y_prob)")
    print("  - plot_roc_curve(models_dict, y_true)")
    print("  - plot_confusion_matrix(y_true, y_pred)")
    print("  - compare_models(models_dict, y_true)")
    print("  - check_overfitting(model, X_train, y_train, X_test, y_test)")
    print("  - generate_classification_report(y_true, y_pred)")
