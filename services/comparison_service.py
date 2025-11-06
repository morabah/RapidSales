"""
Comparison Service
Centralized logic for MoM/YoY percentage change calculations with zero-baseline rules.
"""

from typing import Optional, Tuple


class ComparisonService:
    """Service for calculating percentage changes with consistent zero-baseline rules."""
    
    @staticmethod
    def calculate_percentage_change(
        current: float,
        previous: float,
        format_result: bool = True
    ) -> Tuple[Optional[str], bool]:
        """
        Calculate percentage change with centralized zero-baseline rules.
        
        Rules:
        - prev == 0 and curr == 0 → 0.00% (valid)
        - prev == 0 and curr > 0 → N/A (undefined, cannot divide by zero)
        - else → (curr - prev) / prev * 100
        
        Args:
            current: Current period value
            previous: Previous period value (baseline)
            format_result: If True, returns formatted string; if False, returns float or None
            
        Returns:
            Tuple of (percentage_change, is_valid)
            - percentage_change: Formatted string (e.g., "5.23%") or "N/A" or None, or float if format_result=False
            - is_valid: True if percentage is calculable, False if N/A
        """
        # Rule 1: Both zero → 0% change
        if previous == 0 and current == 0:
            if format_result:
                return "0.00%", True
            else:
                return 0.0, True
        
        # Rule 2: Previous is zero but current is not → N/A (undefined)
        if previous == 0:
            if format_result:
                return "N/A", False
            else:
                return None, False
        
        # Rule 3: Normal case → calculate percentage
        pct_change = ((current - previous) / previous) * 100
        
        if format_result:
            return f"{pct_change:.2f}%", True
        else:
            return pct_change, True
    
    @staticmethod
    def calculate_absolute_change(current: float, previous: float) -> float:
        """
        Calculate absolute change (current - previous).
        
        Args:
            current: Current period value
            previous: Previous period value
            
        Returns:
            Absolute change (float)
        """
        return current - previous

