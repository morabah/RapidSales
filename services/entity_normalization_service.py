"""
Entity Normalization Service
Handles entity name hygiene: whitespace collapse, alias mapping, and null handling.
"""

import re
from typing import Optional, Dict, Set
import pandas as pd


class EntityNormalizationService:
    """Service for normalizing entity names (salesmen, brands, customers)."""
    
    @staticmethod
    def normalize_entity_name(name: Optional[str], alias_map: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Normalize entity name: collapse whitespace, apply title case, apply alias mapping.
        
        Args:
            name: Entity name (can be None)
            alias_map: Optional dictionary mapping original names to canonical names
            
        Returns:
            Normalized name (or None if input is None/empty)
        """
        if name is None:
            return None
        
        # Convert to string and strip
        name_str = str(name).strip()
        
        if not name_str:
            return None
        
        # Collapse repeated whitespace (spaces, tabs, newlines)
        name_str = re.sub(r'\s+', ' ', name_str)
        
        # Apply title case (e.g., "ahmad soliman" → "Ahmad Soliman")
        name_str = name_str.title()
        
        # Apply alias mapping if provided
        if alias_map and name_str in alias_map:
            return alias_map[name_str]
        
        return name_str
    
    @staticmethod
    def normalize_dataframe_column(
        df: pd.DataFrame,
        column: str,
        alias_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Normalize a column in a DataFrame using entity normalization.
        
        Args:
            df: DataFrame to normalize
            column: Column name to normalize
            alias_map: Optional alias mapping
            
        Returns:
            DataFrame with normalized column (copy)
        """
        if column not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy[column] = df_copy[column].apply(
            lambda x: EntityNormalizationService.normalize_entity_name(x, alias_map)
        )
        return df_copy
    
    @staticmethod
    def load_alias_map_from_file(file_path: str) -> Dict[str, str]:
        """
        Load alias mapping from CSV or JSON file.
        
        CSV format: original_name,canonical_name
        JSON format: {"original_name": "canonical_name", ...}
        
        Args:
            file_path: Path to alias file
            
        Returns:
            Dictionary mapping original names to canonical names
        """
        alias_map = {}
        
        try:
            if file_path.endswith('.csv'):
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        original = row.get('original_name', '').strip()
                        canonical = row.get('canonical_name', '').strip()
                        if original and canonical:
                            alias_map[original] = canonical
            elif file_path.endswith('.json'):
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    alias_map = json.load(f)
        except Exception:
            # If file doesn't exist or is invalid, return empty map
            pass
        
        return alias_map
    
    @staticmethod
    def build_alias_map_from_dataframe(
        df: pd.DataFrame,
        entity_column: str,
        stable_key_column: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build alias map from DataFrame based on frequency or stable key.
        
        If stable_key_column is provided, maps all variants to the most common name per key.
        Otherwise, maps all variants to the most common variant.
        
        Args:
            df: DataFrame with entity names
            entity_column: Column containing entity names
            stable_key_column: Optional column with stable identifier (e.g., employee_id)
            
        Returns:
            Dictionary mapping original names to canonical names
        """
        if entity_column not in df.columns:
            return {}
        
        alias_map = {}
        
        if stable_key_column and stable_key_column in df.columns:
            # Group by stable key, map all variants to most common name per key
            for key, group in df.groupby(stable_key_column):
                entity_counts = group[entity_column].value_counts()
                if not entity_counts.empty:
                    canonical = entity_counts.index[0]  # Most common name
                    for variant in entity_counts.index:
                        if variant != canonical:
                            alias_map[variant] = canonical
        else:
            # Map all variants to most common overall
            entity_counts = df[entity_column].value_counts()
            if not entity_counts.empty:
                canonical = entity_counts.index[0]  # Most common name
                for variant in entity_counts.index:
                    if variant != canonical:
                        alias_map[variant] = canonical
        
        return alias_map

