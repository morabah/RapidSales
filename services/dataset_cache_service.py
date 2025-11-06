"""
Dataset Cache Service
Handles dataset change detection and cache invalidation using hash-based fingerprinting.
"""

import hashlib
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime


class DatasetCacheService:
    """Service for detecting dataset changes and managing cache invalidation."""
    
    REQUIRED_COLUMNS = {
        'date': ['visit_date', 'date', 'transaction_date'],
        'amount': ['value', 'amount', 'revenue'],
        'salesman': ['salesman_name', 'salesman', 'rep']
    }
    
    @staticmethod
    def compute_dataset_hash(df: pd.DataFrame, columns: Dict[str, str]) -> str:
        """
        Compute a stable hash for dataset fingerprinting.
        
        Hash includes: sorted column names, row count, min/max dates.
        This invalidates caches when file is replaced or structure changes.
        
        Args:
            df: DataFrame
            columns: Column mapping dictionary
            
        Returns:
            SHA256 hash string (hex)
        """
        if df is None or df.empty:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Collect fingerprint components
        fingerprint_parts = []
        
        # 1. Sorted column names (schema signature)
        sorted_cols = sorted(df.columns.tolist())
        fingerprint_parts.append(f"cols:{','.join(sorted_cols)}")
        
        # 2. Row count
        fingerprint_parts.append(f"rows:{len(df)}")
        
        # 3. Min/max dates (if date column exists)
        date_col = columns.get('date')
        if date_col and date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if not dates.empty:
                    min_date = dates.min().isoformat()
                    max_date = dates.max().isoformat()
                    fingerprint_parts.append(f"dates:{min_date}:{max_date}")
            except Exception:
                pass
        
        # 4. Column mapping (detected columns)
        col_signature = ','.join(f"{k}:{v}" for k, v in sorted(columns.items()) if v)
        fingerprint_parts.append(f"mapping:{col_signature}")
        
        # Combine and hash
        fingerprint = '|'.join(fingerprint_parts)
        return hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()
    
    @staticmethod
    def validate_required_columns(columns: Dict[str, str], df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """
        Validate that required columns are present in the dataset.
        
        Args:
            columns: Column mapping dictionary
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Dataset is empty"
        
        missing = []
        
        # Check required mappings exist
        if not columns.get('date'):
            missing.append('date')
        if not columns.get('amount'):
            missing.append('amount')
        if not columns.get('salesman'):
            missing.append('salesman')
        
        if missing:
            return False, f"Required column mappings missing: {', '.join(missing)}"
        
        # Check columns exist in DataFrame
        date_col = columns.get('date')
        amount_col = columns.get('amount')
        salesman_col = columns.get('salesman')
        
        missing_in_df = []
        if date_col and date_col not in df.columns:
            missing_in_df.append(f"date ('{date_col}')")
        if amount_col and amount_col not in df.columns:
            missing_in_df.append(f"amount ('{amount_col}')")
        if salesman_col and salesman_col not in df.columns:
            missing_in_df.append(f"salesman ('{salesman_col}')")
        
        if missing_in_df:
            return False, f"Required columns not found in data: {', '.join(missing_in_df)}"
        
        return True, None
    
    @staticmethod
    def get_cache_key(operation: str, dataset_hash: str, **kwargs) -> str:
        """
        Generate a cache key for an operation.
        
        Args:
            operation: Operation name (e.g., 'normalized_view', 'monthly_cube', 'sql_plan')
            dataset_hash: Dataset hash
            **kwargs: Additional parameters for cache key
            
        Returns:
            Cache key string
        """
        key_parts = [operation, dataset_hash]
        if kwargs:
            sorted_params = sorted(f"{k}:{v}" for k, v in kwargs.items())
            key_parts.extend(sorted_params)
        return ':'.join(key_parts)
    
    @staticmethod
    def should_invalidate_cache(cached_hash: str, current_hash: str) -> bool:
        """
        Check if cache should be invalidated based on hash comparison.
        
        Args:
            cached_hash: Previously cached dataset hash
            current_hash: Current dataset hash
            
        Returns:
            True if cache should be invalidated
        """
        return cached_hash != current_hash

