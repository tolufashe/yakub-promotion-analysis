"""
Data Processing Module

This module contains functions for loading, cleaning, and preprocessing
the Yakub Trading Group promotion dataset.

Functions:
    load_and_clean_data: Load and perform initial cleaning
    handle_missing_values: Handle missing data appropriately
    detect_outliers: Identify potential outliers
    split_data: Split data into train/test sets with stratification
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the promotion dataset and perform initial cleaning.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Cleaned DataFrame
        
    Example:
        >>> df = load_and_clean_data('data/raw/Promotion Dataset.csv')
        >>> print(f"Loaded {len(df)} records")
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Remove duplicates if any
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"Missing values found in: {list(missing_cols.index)}")
    
    return df


def analyze_missing_qualification(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze whether missing qualification data is related to promotion outcomes.
    
    Uses chi-square test to determine if missingness is random or systematic.
    
    Args:
        df: DataFrame with Qualification column
        
    Returns:
        Dictionary with test results and recommendation
        
    Note:
        If p-value < 0.05, missing qualification is associated with promotion
        and should be encoded as a separate category rather than dropped.
    """
    # Create missing indicator
    df_temp = df.copy()
    df_temp['Qual_Missing'] = df_temp['Qualification'].isnull()
    
    # Calculate promotion rates
    promo_by_missing = df_temp.groupby('Qual_Missing')['Promoted_or_Not'].mean()
    
    # Chi-square test
    contingency = pd.crosstab(df_temp['Qual_Missing'], df_temp['Promoted_or_Not'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    results = {
        'promo_with_qual': promo_by_missing.get(False, 0),
        'promo_without_qual': promo_by_missing.get(True, 0),
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'recommendation': 'encode_as_category' if p_value < 0.05 else 'can_drop_or_impute'
    }
    
    return results


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Qualification: Encode missing as 'Unknown' category (preserves signal)
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Handle Qualification missing values
    if df_clean['Qualification'].isnull().sum() > 0:
        # Analyze if missingness is related to promotion
        analysis = analyze_missing_qualification(df_clean)
        
        if analysis['significant']:
            print(f"Missing Qualification IS associated with promotion (p={analysis['p_value']:.2e})")
            print(f"  With Qualification: {analysis['promo_with_qual']:.2%}")
            print(f"  Without Qualification: {analysis['promo_without_qual']:.2%}")
            print("  → Encoding missing as 'Unknown' category")
        
        # Fill missing with 'Unknown'
        df_clean['Qualification'] = df_clean['Qualification'].fillna('Unknown')
    
    return df_clean


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numeric column.
    
    Args:
        df: DataFrame
        column: Column name to check
        method: 'iqr' (Interquartile Range) or 'zscore'
        
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")


def split_data(
    df: pd.DataFrame,
    target_col: str = 'Promoted_or_Not',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with stratification.
    
    Stratification ensures both sets have the same proportion of promoted employees,
    which is important given the class imbalance (~8.5% promotion rate).
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
        
    Example:
        >>> X_train, X_test, y_train, y_test = split_data(df)
        >>> print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Data split complete:")
    print(f"  Training: {len(X_train):,} samples ({len(X_train)/len(df):.1%})")
    print(f"  Testing:  {len(X_test):,} samples ({len(X_test)/len(df):.1%})")
    print(f"  Promotion rate (train): {y_train.mean():.2%}")
    print(f"  Promotion rate (test):  {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'promotion_rate': df['Promoted_or_Not'].mean(),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - load_and_clean_data(filepath)")
    print("  - handle_missing_values(df)")
    print("  - analyze_missing_qualification(df)")
    print("  - detect_outliers(df, column, method)")
    print("  - split_data(df, target_col, test_size)")
    print("  - get_data_summary(df)")
