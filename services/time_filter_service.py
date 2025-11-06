"""
Time Filter Service
Handles time period calculations with proper calendar logic.

DEPRECATED: This service is being phased out in favor of TimeResolver.
Kept only as a fallback for edge cases. New code should use TimeResolver instead.
"""

from datetime import datetime, timedelta
from typing import Optional
from dateutil.relativedelta import relativedelta
import pandas as pd

from .query_plan_service import TimeMode, TimeFilter


class TimeFilterService:
    """Service for filtering DataFrames by time periods."""
    
    @staticmethod
    def last_completed_quarter_bounds(ref: datetime) -> tuple[datetime, datetime]:
        """
        Get bounds for last completed calendar quarter.
        
        Quarters:
        - Q1: Jan-Mar
        - Q2: Apr-Jun
        - Q3: Jul-Sep
        - Q4: Oct-Dec
        
        Args:
            ref: Reference date (usually max date in dataset)
            
        Returns:
            Tuple of (start, end_exclusive) datetime for last completed quarter
            end_exclusive is the start of the current quarter (not inclusive)
        """
        q = (ref.month - 1) // 3 + 1  # Current quarter (1-4)
        
        # Start of current quarter
        curr_q_start = datetime(ref.year, (q - 1) * 3 + 1, 1)
        
        # Last quarter number
        last_q = q - 1 if q > 1 else 4
        last_q_year = ref.year if q > 1 else ref.year - 1
        
        # Start of last quarter
        last_q_start = datetime(last_q_year, (last_q - 1) * 3 + 1, 1)
        
        # End of last quarter is exclusive (start of current quarter)
        last_q_end_exclusive = curr_q_start
        
        return last_q_start, last_q_end_exclusive
    
    @staticmethod
    def previous_quarter_bounds(ref: datetime) -> tuple[datetime, datetime]:
        """
        Get bounds for previous quarter (the quarter before last completed quarter).
        
        Args:
            ref: Reference date (usually max date in dataset)
            
        Returns:
            Tuple of (start, end_exclusive) datetime for previous quarter
        """
        # Get last completed quarter
        last_q_start, last_q_end_exclusive = TimeFilterService.last_completed_quarter_bounds(ref)
        
        # Previous quarter is one quarter before last completed quarter
        prev_q_start = last_q_start - relativedelta(months=3)
        prev_q_end_exclusive = last_q_start
        
        return prev_q_start, prev_q_end_exclusive
    
    @staticmethod
    def this_quarter_bounds(ref: datetime) -> tuple[datetime, datetime]:
        """Get bounds for current quarter."""
        q = (ref.month - 1) // 3 + 1
        quarter_start = datetime(ref.year, (q - 1) * 3 + 1, 1)
        quarter_end = quarter_start + relativedelta(months=3) - timedelta(days=1)
        return quarter_start, quarter_end
    
    @staticmethod
    def filter_by_time_period(
        df: pd.DataFrame,
        date_col: str,
        time_filter: TimeFilter,
        ref_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame by time period.
        
        Args:
            df: DataFrame to filter
            date_col: Name of date column
            time_filter: TimeFilter specification
            ref_date: Reference date (defaults to today or max date in dataset)
            
        Returns:
            Filtered DataFrame
        """
        if time_filter is None or date_col not in df.columns:
            return df
        
        # Use max date in dataset as reference if not provided
        if ref_date is None:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if not dates.empty:
                    ref_date = dates.max().to_pydatetime()
                else:
                    ref_date = datetime.now()
            except Exception:
                ref_date = datetime.now()
        
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        start_date = None
        end_date = None
        
        if time_filter.mode == TimeMode.LAST_COMPLETED_QUARTER:
            start_date, end_date_exclusive = TimeFilterService.last_completed_quarter_bounds(ref_date)
            # Use exclusive end for last completed quarter
            end_date = end_date_exclusive - timedelta(days=1)
        
        elif time_filter.mode == TimeMode.THIS_QUARTER:
            start_date, end_date = TimeFilterService.this_quarter_bounds(ref_date)
        
        elif time_filter.mode == TimeMode.LAST_MONTH:
            end_date = ref_date.replace(day=1) - timedelta(days=1)
            start_date = end_date.replace(day=1)
        
        elif time_filter.mode == TimeMode.THIS_MONTH:
            start_date = ref_date.replace(day=1)
            end_date = start_date + relativedelta(months=1) - timedelta(days=1)
        
        elif time_filter.mode == TimeMode.LAST_N_MONTHS and time_filter.n_months:
            # CALENDAR MONTHS (anchored to dataset max date): Last N complete calendar months
            # Anchor to dataset max date (ref_date), not "today"
            # Example: If max_date is 2025-09-28 and N=3:
            #   end_exclusive = 2025-10-01 (start of next month after max_date)
            #   start = 2025-07-01 (start of first month in range)
            # Returns: Jul 2025, Aug 2025, Sep 2025 (3 complete months, including zeros)
            end_exclusive = ref_date.replace(day=1) + relativedelta(months=1)  # Exclusive: start of next month
            start_date = end_exclusive - relativedelta(months=time_filter.n_months)
            # Store end_exclusive for later use in filtering
            end_date = end_exclusive
        
        elif time_filter.mode == TimeMode.SPECIFIC_QUARTER and time_filter.quarter and time_filter.year:
            quarter_start_month = (time_filter.quarter - 1) * 3 + 1
            start_date = datetime(time_filter.year, quarter_start_month, 1)
            end_date = start_date + relativedelta(months=3) - timedelta(days=1)
        
        elif time_filter.mode == TimeMode.SPECIFIC_MONTH and time_filter.month and time_filter.year:
            start_date = datetime(time_filter.year, time_filter.month, 1)
            end_date = start_date + relativedelta(months=1) - timedelta(days=1)
        
        if start_date and end_date:
            # For LAST_N_MONTHS, use exclusive end (start of next month)
            if time_filter.mode == TimeMode.LAST_N_MONTHS:
                mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] < end_date)
            else:
                # For other periods, use inclusive end
                mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)
            
            filtered = df_copy[mask].copy()
            
            # Handle empty results
            if len(filtered) == 0:
                # Return empty DataFrame with same structure
                return filtered
            
            return filtered
        
        return df  # No filter applied

