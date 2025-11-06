"""
Canonical Time Resolver
Centralized time window resolution for consistent date bounds across all paths.
"""

from datetime import datetime
from typing import Optional, Tuple
from dateutil.relativedelta import relativedelta
import pandas as pd

from .query_plan_service import TimeMode, TimeFilter


class TimeResolver:
    """Canonical time resolver - single source of truth for all time windows."""
    
    @staticmethod
    def resolve_time_window(
        time_filter: Optional[TimeFilter],
        ref_date: datetime,
        dataset_max_date: Optional[datetime] = None
    ) -> Tuple[Optional[datetime], Optional[datetime], str]:
        """
        Resolve time window to exact bounds with end-exclusive convention.
        
        Args:
            time_filter: TimeFilter specification
            ref_date: Reference date (usually dataset max date)
            dataset_max_date: Explicit dataset max date (if different from ref_date)
            
        Returns:
            Tuple of (start_date, end_date_exclusive, label)
            - start_date: Inclusive start date
            - end_date_exclusive: Exclusive end date (use < end_date_exclusive in filters)
            - label: Human-readable period label with exact bounds
        """
        if not time_filter:
            return None, None, "All time"
        
        # Use dataset max date as reference (critical for accuracy)
        anchor_date = dataset_max_date if dataset_max_date else ref_date
        
        start_date = None
        end_date_exclusive = None
        label = ""
        
        if time_filter.mode == TimeMode.LAST_COMPLETED_QUARTER:
            # Last completed quarter: full quarter before current quarter
            q = (anchor_date.month - 1) // 3 + 1  # Current quarter (1-4)
            curr_q_start = datetime(anchor_date.year, (q - 1) * 3 + 1, 1)
            
            # Last quarter number
            last_q = q - 1 if q > 1 else 4
            last_q_year = anchor_date.year if q > 1 else anchor_date.year - 1
            
            # Start of last quarter
            start_date = datetime(last_q_year, (last_q - 1) * 3 + 1, 1)
            end_date_exclusive = curr_q_start  # Exclusive end
            
            q_label = f"Q{last_q}-{last_q_year}"
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"{q_label}: {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.THIS_QUARTER:
            q = (anchor_date.month - 1) // 3 + 1
            start_date = datetime(anchor_date.year, (q - 1) * 3 + 1, 1)
            end_date_exclusive = start_date + relativedelta(months=3)
            
            q_label = f"Q{q}-{anchor_date.year}"
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"{q_label}: {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.LAST_MONTH:
            # Last completed month
            end_date_exclusive = anchor_date.replace(day=1)
            start_date = (end_date_exclusive - relativedelta(months=1))
            
            month_name = start_date.strftime('%B %Y')
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"Last month ({month_name}): {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.THIS_MONTH:
            start_date = anchor_date.replace(day=1)
            end_date_exclusive = start_date + relativedelta(months=1)
            
            month_name = start_date.strftime('%B %Y')
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"This month ({month_name}): {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.LAST_N_MONTHS and time_filter.n_months:
            # Calendar months anchored to dataset max date
            end_date_exclusive = anchor_date.replace(day=1) + relativedelta(months=1)
            start_date = end_date_exclusive - relativedelta(months=time_filter.n_months)
            
            n = time_filter.n_months
            start_month = start_date.strftime('%b %Y')
            end_month = (end_date_exclusive - relativedelta(days=1)).strftime('%b %Y')
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"Last {n} months ({start_month} → {end_month}): {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.SPECIFIC_QUARTER and time_filter.quarter and time_filter.year:
            quarter_start_month = (time_filter.quarter - 1) * 3 + 1
            start_date = datetime(time_filter.year, quarter_start_month, 1)
            end_date_exclusive = start_date + relativedelta(months=3)
            
            q_label = f"Q{time_filter.quarter}-{time_filter.year}"
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"{q_label}: {start_str} → {end_str}"
            
        elif time_filter.mode == TimeMode.SPECIFIC_MONTH and time_filter.month and time_filter.year:
            start_date = datetime(time_filter.year, time_filter.month, 1)
            end_date_exclusive = start_date + relativedelta(months=1)
            
            month_name = start_date.strftime('%B %Y')
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date_exclusive - relativedelta(days=1)).strftime('%Y-%m-%d')
            label = f"{month_name}: {start_str} → {end_str}"
        
        return start_date, end_date_exclusive, label
    
    @staticmethod
    def filter_dataframe(
        df: pd.DataFrame,
        date_col: str,
        start_date: Optional[datetime],
        end_date_exclusive: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Filter DataFrame using end-exclusive bounds.
        
        Args:
            df: DataFrame to filter
            date_col: Name of date column
            start_date: Inclusive start date
            end_date_exclusive: Exclusive end date
            
        Returns:
            Filtered DataFrame
        """
        if start_date is None or end_date_exclusive is None:
            return df
        
        if date_col not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Use end-exclusive bounds: >= start AND < end
        mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] < end_date_exclusive)
        return df_copy[mask].copy()
    
    @staticmethod
    def get_sql_predicate(
        date_col: str,
        start_date: Optional[datetime],
        end_date_exclusive: Optional[datetime]
    ) -> str:
        """
        Generate SQL predicate with end-exclusive bounds and DATE casting.
        
        Args:
            date_col: Name of date column
            start_date: Inclusive start date
            end_date_exclusive: Exclusive end date
            
        Returns:
            SQL WHERE clause (e.g., "DATE(VISITDATE) >= DATE '2025-01-01' AND DATE(VISITDATE) < DATE '2025-10-01'")
        """
        if start_date is None or end_date_exclusive is None:
            return "1=1"  # Always true
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date_exclusive.strftime('%Y-%m-%d')
        
        # Use DATE casting for timestamp columns and end-exclusive bounds
        return f"DATE({date_col}) >= DATE '{start_str}' AND DATE({date_col}) < DATE '{end_str}'"
    
    @staticmethod
    def last_completed_quarter_bounds(ref: datetime) -> Tuple[datetime, datetime]:
        """
        Get bounds for last completed calendar quarter.
        
        Args:
            ref: Reference date (usually max date in dataset)
            
        Returns:
            Tuple of (start, end_exclusive) datetime for last completed quarter
        """
        q = (ref.month - 1) // 3 + 1  # Current quarter (1-4)
        curr_q_start = datetime(ref.year, (q - 1) * 3 + 1, 1)
        
        # Last quarter number
        last_q = q - 1 if q > 1 else 4
        last_q_year = ref.year if q > 1 else ref.year - 1
        
        # Start of last quarter
        last_q_start = datetime(last_q_year, (last_q - 1) * 3 + 1, 1)
        last_q_end_exclusive = curr_q_start
        
        return last_q_start, last_q_end_exclusive
    
    @staticmethod
    def previous_quarter_bounds(ref: datetime) -> Tuple[datetime, datetime]:
        """
        Get bounds for previous quarter (the quarter before last completed quarter).
        
        Args:
            ref: Reference date (usually max date in dataset)
            
        Returns:
            Tuple of (start, end_exclusive) datetime for previous quarter
        """
        # Get last completed quarter
        last_q_start, last_q_end_exclusive = TimeResolver.last_completed_quarter_bounds(ref)
        
        # Previous quarter is one quarter before last completed quarter
        prev_q_start = last_q_start - relativedelta(months=3)
        prev_q_end_exclusive = last_q_start
        
        return prev_q_start, prev_q_end_exclusive

