"""
Feature Engineering Module

This module contains functions for creating new features and encoding
categorical variables for the Yakub Trading Group promotion analysis.

IMPORTANT: To prevent data leakage, target encoding is performed AFTER
the train-test split, using only training data to calculate target means.

Functions:
    create_features: Create new engineered features (Age, Years_at_Company)
    target_encode_state: Perform target encoding for State_Of_Origin
    encode_categorical: Encode categorical variables using appropriate methods
    prepare_features: Full pipeline for feature preparation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features from existing columns.
    
    Features created:
    - Age: From Year_of_birth
    - Years_at_Company: From Year_of_recruitment
    
    Args:
        df: DataFrame with Year_of_birth and Year_of_recruitment columns
        
    Returns:
        DataFrame with new features added
        
    Note:
        This function does NOT create State_Score (target encoding).
        Target encoding must be done AFTER train-test split to prevent data leakage.
    """
    df_new = df.copy()
    
    # Get current year for age calculation
    current_year = datetime.now().year
    
    # Feature 1: Age (more interpretable than birth year)
    df_new['Age'] = current_year - df_new['Year_of_birth']
    
    # Feature 2: Years_at_Company (tenure)
    latest_year = df_new['Year_of_recruitment'].max()
    df_new['Years_at_Company'] = latest_year - df_new['Year_of_recruitment']
    
    # Drop raw year columns (replaced by engineered features)
    columns_to_drop = ['Year_of_birth', 'Year_of_recruitment', 'EmployeeNo']
    df_new = df_new.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Features created: Age, Years_at_Company")
    print(f"Columns dropped: {columns_to_drop}")
    
    return df_new


def target_encode_state(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    state_col: str = 'State_Of_Origin',
    smooth: bool = True,
    min_samples: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform target encoding for State_Of_Origin column.
    
    CRITICAL: This function prevents data leakage by using ONLY training data
    to calculate target means. The same mapping is then applied to test data.
    
    Target encoding replaces categorical values with the mean of the target
    variable for that category. This helps detect geographic bias.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target (used to calculate means)
        state_col: Name of the state column
        smooth: Whether to apply smoothing for rare categories
        min_samples: Minimum samples for a category to use its own mean
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
        
    Example:
        >>> X_train, X_test = target_encode_state(X_train, X_test, y_train)
        >>> print(X_train['State_Score'].head())
        
    Note:
        States not seen in training will be encoded with the global mean.
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # Calculate global mean from training data only
    global_mean = y_train.mean()
    
    # Calculate state means from training data only
    train_with_target = X_train_enc.copy()
    train_with_target['target'] = y_train.values
    
    if smooth:
        # Smoothing: weighted average between category mean and global mean
        # This prevents overfitting for rare categories
        state_stats = train_with_target.groupby(state_col).agg({
            'target': ['sum', 'count']
        })
        state_stats.columns = ['sum', 'count']
        
        # Smoothed mean = (sum + min_samples * global_mean) / (count + min_samples)
        state_means = (
            (state_stats['sum'] + min_samples * global_mean) / 
            (state_stats['count'] + min_samples)
        )
    else:
        state_means = train_with_target.groupby(state_col)['target'].mean()
    
    # Create mapping dictionary
    state_mapping = state_means.to_dict()
    
    # Apply encoding to training data
    X_train_enc['State_Score'] = X_train_enc[state_col].map(state_mapping)
    
    # Apply encoding to test data (using training mapping)
    X_test_enc['State_Score'] = X_test_enc[state_col].map(state_mapping)
    
    # Fill states not seen in training with global mean
    X_test_enc['State_Score'] = X_test_enc['State_Score'].fillna(global_mean)
    
    # Drop original state column
    X_train_enc = X_train_enc.drop(columns=[state_col])
    X_test_enc = X_test_enc.drop(columns=[state_col])
    
    print(f"Target encoding complete:")
    print(f"  States in training: {len(state_mapping)}")
    print(f"  Global mean: {global_mean:.4f}")
    print(f"  State means range: {state_means.min():.4f} - {state_means.max():.4f}")
    
    return X_train_enc, X_test_enc


def encode_categorical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Encode categorical variables using appropriate methods.
    
    Encoding strategies:
    - Binary (Yes/No, Male/Female): Map to 0/1
    - Ordinal (Qualification): Map to ordered integers
    - Nominal (Division, Channel, Marital Status): One-hot encoding
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded, encoding_info)
        
    Example:
        >>> X_train, X_test, info = encode_categorical(X_train, X_test)
        >>> print(f"Final shape: {X_train.shape}")
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    encoding_info = {}
    
    # ----- 1. Binary Encoding -----
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = [
        'Gender',
        'Foreign_schooled',
        'Past_Disciplinary_Action',
        'Previous_IntraDepartmental_Movement'
    ]
    
    for col in binary_cols:
        if col in X_train_enc.columns:
            X_train_enc[col] = X_train_enc[col].map(binary_map)
            X_test_enc[col] = X_test_enc[col].map(binary_map)
    
    encoding_info['binary_encoded'] = binary_cols
    
    # ----- 2. Ordinal Encoding for Qualification -----
    qual_map = {
        'Non University Education': 1,
        'Unknown': 2,
        'First Degree or HND': 3,
        'MSc  MBA and PhD': 4
    }
    
    if 'Qualification' in X_train_enc.columns:
        X_train_enc['Qualification'] = X_train_enc['Qualification'].map(qual_map)
        X_test_enc['Qualification'] = X_test_enc['Qualification'].map(qual_map)
        encoding_info['qualification_map'] = qual_map
    
    # ----- 3. One-Hot Encoding for Nominal Categories -----
    nominal_cols = ['Division', 'Channel_of_Recruitment', 'Marital_Status']
    nominal_cols = [col for col in nominal_cols if col in X_train_enc.columns]
    
    if nominal_cols:
        # Combine train and test for consistent encoding
        combined = pd.concat([X_train_enc, X_test_enc], axis=0)
        combined_encoded = pd.get_dummies(combined, columns=nominal_cols, drop_first=True)
        
        # Split back
        X_train_enc = combined_encoded.iloc[:len(X_train_enc)]
        X_test_enc = combined_encoded.iloc[len(X_train_enc):]
        
        encoding_info['one_hot_encoded'] = nominal_cols
        encoding_info['new_columns'] = len(combined_encoded.columns) - len(X_train.columns)
    
    # Convert boolean columns to integers
    bool_cols = X_train_enc.select_dtypes(include='bool').columns
    X_train_enc[bool_cols] = X_train_enc[bool_cols].astype(int)
    X_test_enc[bool_cols] = X_test_enc[bool_cols].astype(int)
    
    print(f"Encoding complete:")
    print(f"  Binary encoded: {len(binary_cols)} columns")
    print(f"  Ordinal encoded: Qualification")
    print(f"  One-hot encoded: {len(nominal_cols)} columns")
    print(f"  Final shape: {X_train_enc.shape}")
    
    return X_train_enc, X_test_enc, encoding_info


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    StandardScaler transforms features to have mean=0 and std=1.
    This is important for algorithms like Logistic Regression that are
    sensitive to feature scales.
    
    Args:
        X_train: Training features
        X_test: Testing features
        scaler: Pre-fitted scaler (if None, a new one will be created)
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
        
    Note:
        The scaler is fitted ONLY on training data to prevent data leakage.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    print(f"  Training mean (should be ~0): {X_train_scaled.mean():.6f}")
    print(f"  Training std (should be ~1): {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Promoted_or_Not',
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True
) -> Tuple:
    """
    Full pipeline for feature preparation.
    
    This function orchestrates the entire feature engineering pipeline:
    1. Create new features (Age, Years_at_Company)
    2. Split data into train/test
    3. Target encode State_Of_Origin (after split to prevent leakage)
    4. Encode categorical variables
    5. Scale features (optional)
    
    Args:
        df: Cleaned DataFrame
        target_col: Name of target column
        test_size: Proportion for testing
        random_state: Random seed
        scale: Whether to scale features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
        
    Example:
        >>> X_train, X_test, y_train, y_test, scaler = prepare_features(df)
        >>> model.fit(X_train, y_train)
    """
    from .data_processing import split_data
    
    # Step 1: Create new features
    print("\n" + "="*60)
    print("STEP 1: Creating new features")
    print("="*60)
    df = create_features(df)
    
    # Step 2: Split data
    print("\n" + "="*60)
    print("STEP 2: Splitting data")
    print("="*60)
    X_train, X_test, y_train, y_test = split_data(
        df, target_col, test_size, random_state
    )
    
    # Step 3: Target encoding (AFTER split to prevent data leakage)
    print("\n" + "="*60)
    print("STEP 3: Target encoding State_Of_Origin")
    print("="*60)
    if 'State_Of_Origin' in X_train.columns:
        X_train, X_test = target_encode_state(X_train, X_test, y_train)
    
    # Step 4: Encode categorical variables
    print("\n" + "="*60)
    print("STEP 4: Encoding categorical variables")
    print("="*60)
    X_train, X_test, encoding_info = encode_categorical(X_train, X_test)
    
    # Step 5: Scale features (optional)
    scaler = None
    if scale:
        print("\n" + "="*60)
        print("STEP 5: Scaling features")
        print("="*60)
        X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print("\n" + "="*60)
    print("FEATURE PREPARATION COMPLETE")
    print("="*60)
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - create_features(df)")
    print("  - target_encode_state(X_train, X_test, y_train)")
    print("  - encode_categorical(X_train, X_test)")
    print("  - scale_features(X_train, X_test)")
    print("  - prepare_features(df) [full pipeline]")
    print("\nIMPORTANT: Always split data BEFORE target encoding to prevent data leakage!")
