#!/usr/bin/env python3
"""
Main script for Yakub Promotion Analysis.

This script demonstrates the complete data science pipeline:
1. Load and clean data
2. Engineer features
3. Train models
4. Evaluate and compare models
5. Generate insights

Usage:
    python main.py

Or import as a module:
    from main import run_analysis
    results = run_analysis('data/raw/Promotion Dataset.csv')
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import load_and_clean_data, handle_missing_values, get_data_summary
from src.feature_engineering import create_features, prepare_features
from src.models import train_all_models, get_feature_importance
from src.evaluation import (
    compare_models,
    check_overfitting,
    plot_roc_curve,
    plot_confusion_matrix,
    print_summary_report
)


def run_analysis(data_path: str = 'data/raw/Promotion Dataset.csv') -> dict:
    """
    Run the complete analysis pipeline.
    
    Args:
        data_path: Path to the raw data CSV file
        
    Returns:
        Dictionary containing all results
    """
    print("\n" + "="*70)
    print("     YAKUB TRADING GROUP - PROMOTION ANALYSIS")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load and Clean Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading and Cleaning Data")
    print("="*70)
    
    df = load_and_clean_data(data_path)
    df = handle_missing_values(df)
    
    summary = get_data_summary(df)
    print(f"\nDataset Summary:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Total features: {summary['total_features']}")
    print(f"  Promotion rate: {summary['promotion_rate']:.2%}")
    
    # ========================================================================
    # STEP 2: Feature Engineering and Data Split
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Feature Engineering and Data Split")
    print("="*70)
    
    X_train, X_test, y_train, y_test, scaler = prepare_features(df)
    
    # Get feature names for importance analysis
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # ========================================================================
    # STEP 3: Train Models
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Training Models")
    print("="*70)
    
    models = train_all_models(X_train, y_train)
    
    # ========================================================================
    # STEP 4: Evaluate Models
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Evaluating Models")
    print("="*70)
    
    # Prepare predictions dictionary
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[name] = (y_pred, y_prob)
    
    # Compare models
    comparison = compare_models(predictions, y_test.values)
    
    # Identify best model
    best_model_name = comparison.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print_summary_report(comparison, best_model_name)
    
    # ========================================================================
    # STEP 5: Overfitting Check
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Overfitting Check")
    print("="*70)
    
    for name, model in models.items():
        overfit = check_overfitting(
            model, X_train, y_train.values,
            X_test, y_test.values, name
        )
        
        status = "⚠️ OVERFITTING" if overfit['overfitting_detected'] else "✅ OK"
        print(f"\n{name}:")
        print(f"  Train AUC: {overfit['train_auc']:.4f}")
        print(f"  Test AUC:  {overfit['test_auc']:.4f}")
        print(f"  Gap:       {overfit['auc_gap']:.4f} {status}")
    
    # ========================================================================
    # STEP 6: Feature Importance
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Feature Importance Analysis")
    print("="*70)
    
    if hasattr(best_model, 'feature_importances_'):
        importance = get_feature_importance(best_model, feature_names)
        print("\nTop 10 Most Important Features:")
        for i, row in importance.head(10).iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: Generating Visualizations")
    print("="*70)
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # ROC Curves
    print("\nGenerating ROC curves...")
    plot_roc_curve(predictions, y_test.values, save_path='reports/figures/roc_curves.png')
    
    # Confusion Matrix for best model
    print("\nGenerating confusion matrix for best model...")
    best_pred, _ = predictions[best_model_name]
    plot_confusion_matrix(
        y_test.values, best_pred, best_model_name,
        save_path='reports/figures/confusion_matrix.png'
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC-AUC: {comparison.iloc[0]['ROC-AUC']:.4f}")
    print(f"\nVisualizations saved to: reports/figures/")
    print("="*70)
    
    # Return results
    results = {
        'models': models,
        'predictions': predictions,
        'comparison': comparison,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler
    }
    
    return results


def main():
    """Main entry point."""
    # Check if data file exists
    data_path = 'data/raw/Promotion Dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("\nPlease ensure the data file is in the correct location:")
        print("  data/raw/Promotion Dataset.csv")
        sys.exit(1)
    
    # Run analysis
    try:
        results = run_analysis(data_path)
        print("\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
