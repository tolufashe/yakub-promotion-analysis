"""
Unit tests for data_processing module.

Run with: pytest tests/test_data_processing.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    handle_missing_values,
    analyze_missing_qualification,
    detect_outliers,
    split_data,
    get_data_summary
)


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""
    
    def test_no_missing_values(self):
        """Test with no missing values."""
        df = pd.DataFrame({
            'Qualification': ['First Degree', 'MSc', 'PhD'],
            'Promoted_or_Not': [1, 0, 1]
        })
        
        result = handle_missing_values(df)
        
        assert result['Qualification'].isnull().sum() == 0
        assert len(result) == len(df)
    
    def test_missing_qualification_encoded(self):
        """Test that missing qualifications are encoded as 'Unknown'."""
        df = pd.DataFrame({
            'Qualification': ['First Degree', None, 'PhD', None],
            'Promoted_or_Not': [1, 0, 1, 0]
        })
        
        result = handle_missing_values(df)
        
        assert result['Qualification'].isnull().sum() == 0
        assert 'Unknown' in result['Qualification'].values
        assert result['Qualification'].value_counts()['Unknown'] == 2


class TestAnalyzeMissingQualification:
    """Tests for analyze_missing_qualification function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        df = pd.DataFrame({
            'Qualification': ['First Degree', None, 'PhD'],
            'Promoted_or_Not': [1, 0, 1]
        })
        
        result = analyze_missing_qualification(df)
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'significant' in result
    
    def test_detects_association(self):
        """Test that function detects association between missingness and promotion."""
        # Create data where missing qualification is associated with no promotion
        df = pd.DataFrame({
            'Qualification': ['First Degree'] * 10 + [None] * 10,
            'Promoted_or_Not': [1] * 8 + [0] * 2 + [0] * 10  # Missing = no promotion
        })
        
        result = analyze_missing_qualification(df)
        
        # Should detect significant association
        assert result['significant'] == True
        assert result['recommendation'] == 'encode_as_category'


class TestDetectOutliers:
    """Tests for detect_outliers function."""
    
    def test_iqr_method(self):
        """Test IQR outlier detection."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        outliers = detect_outliers(df, 'score', method='iqr')
        
        assert outliers.sum() == 1  # One outlier
        assert outliers.iloc[-1] == True  # Last value is outlier
    
    def test_zscore_method(self):
        """Test Z-score outlier detection."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        outliers = detect_outliers(df, 'score', method='zscore')
        
        assert outliers.sum() == 1  # One outlier
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        df = pd.DataFrame({'score': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            detect_outliers(df, 'score', method='invalid')


class TestSplitData:
    """Tests for split_data function."""
    
    def test_split_proportions(self):
        """Test that split proportions are correct."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100),
            'Promoted_or_Not': [0] * 90 + [1] * 10  # 10% promotion rate
        })
        
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        
        # Check proportions
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_stratification(self):
        """Test that stratification preserves class distribution."""
        df = pd.DataFrame({
            'feature1': range(1000),
            'feature2': range(1000),
            'Promoted_or_Not': [0] * 900 + [1] * 100  # 10% promotion rate
        })
        
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        
        # Check that promotion rates are similar
        original_rate = df['Promoted_or_Not'].mean()
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        
        assert abs(train_rate - original_rate) < 0.02  # Within 2%
        assert abs(test_rate - original_rate) < 0.02
    
    def test_target_not_in_features(self):
        """Test that target column is not in feature sets."""
        df = pd.DataFrame({
            'feature1': range(100),
            'Promoted_or_Not': [0] * 90 + [1] * 10
        })
        
        X_train, X_test, y_train, y_test = split_data(df)
        
        assert 'Promoted_or_Not' not in X_train.columns
        assert 'Promoted_or_Not' not in X_test.columns


class TestGetDataSummary:
    """Tests for get_data_summary function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': ['A', 'B'] * 50,
            'Promoted_or_Not': [0] * 90 + [1] * 10
        })
        
        summary = get_data_summary(df)
        
        assert isinstance(summary, dict)
        assert 'total_records' in summary
        assert 'promotion_rate' in summary
    
    def test_correct_values(self):
        """Test that summary values are correct."""
        df = pd.DataFrame({
            'num1': range(100),
            'num2': range(100),
            'cat': ['A', 'B'] * 50,
            'Promoted_or_Not': [0] * 90 + [1] * 10
        })
        
        summary = get_data_summary(df)
        
        assert summary['total_records'] == 100
        assert summary['total_features'] == 4
        assert summary['promotion_rate'] == 0.1
        assert summary['numeric_features'] == 3
        assert summary['categorical_features'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
