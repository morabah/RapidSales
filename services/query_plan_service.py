"""
Query Plan Service
Parses user questions into structured query plans for safe, deterministic execution.
"""

import re
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class TimeMode(Enum):
    """Time period modes."""
    ALL_TIME = "all_time"
    LAST_COMPLETED_QUARTER = "last_completed_quarter"
    THIS_QUARTER = "this_quarter"
    LAST_MONTH = "last_month"
    THIS_MONTH = "this_month"
    LAST_N_MONTHS = "last_n_months"
    SPECIFIC_QUARTER = "specific_quarter"
    SPECIFIC_MONTH = "specific_month"
    CUSTOM_RANGE = "custom_range"


class SortDirection(Enum):
    """Sort direction."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class TimeFilter:
    """Time filter specification."""
    mode: TimeMode
    n_months: Optional[int] = None
    quarter: Optional[int] = None
    year: Optional[int] = None
    month: Optional[int] = None


@dataclass
class QueryPlan:
    """Structured query plan for safe execution."""
    metric: str = "sales_value"  # sales_value, units, avg_price, etc.
    aggregation: str = "sum"  # sum, avg, count, etc.
    dimensions: list[str] = field(default_factory=list)  # ["salesman"], ["customer"], ["product"], etc.
    time_filter: Optional[TimeFilter] = None
    filters: list[dict] = field(default_factory=list)  # Additional filters
    sort: list[dict] = field(default_factory=list)  # [{"by": "sales_value", "dir": "desc"}]
    limit: int = 50
    include_bottom: bool = False  # For worst/bottom questions
    include_top: bool = True  # For top/best questions


class QueryPlanService:
    """Service for parsing questions into query plans."""
    
    @staticmethod
    def parse_question(question: str) -> QueryPlan:
        """
        Parse user question into structured QueryPlan.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryPlan with structured query specification
        """
        ql = (question or "").lower()
        
        # Initialize plan
        plan = QueryPlan()
        
        # Detect dimensions (salesman, customer, product)
        plan.dimensions = QueryPlanService._detect_dimensions(ql)
        
        # Detect time filter
        plan.time_filter = QueryPlanService._detect_time_filter(ql)
        
        # Detect sort direction and limit
        plan.include_bottom = QueryPlanService._has_phrase(ql, r'\b(worst|bottom|lowest|poor|bad|underperform|weak)\b')
        plan.include_top = QueryPlanService._has_phrase(ql, r'\b(top|best|highest|leading|great|excellent|strong)\b')
        
        # If no specific intent, include both
        if not plan.include_bottom and not plan.include_top:
            plan.include_top = True
            plan.include_bottom = True
        
        # Detect limit (e.g., "top 5", "bottom 10")
        limit_match = re.search(r'\b(top|bottom|worst|best)\s+(\d+)\b', ql)
        if limit_match:
            plan.limit = int(limit_match.group(2))
        else:
            plan.limit = 10  # Default
        
        # Set sort direction
        if plan.include_bottom and not plan.include_top:
            plan.sort = [{"by": "sales_value", "dir": "asc"}]  # Ascending for worst
        else:
            plan.sort = [{"by": "sales_value", "dir": "desc"}]  # Descending for top
        
        return plan
    
    @staticmethod
    def _has_phrase(text: str, pattern: str) -> bool:
        """
        Check if text contains phrase with word boundaries.
        
        Args:
            text: Text to search
            pattern: Regex pattern with word boundaries
            
        Returns:
            True if phrase found
        """
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    
    @staticmethod
    def _detect_dimensions(question: str) -> list[str]:
        """Detect which dimensions (salesman, customer, product) are relevant."""
        dimensions = []
        
        if QueryPlanService._has_phrase(question, r'\b(salesman|sales rep|rep|agent|salesperson)\b'):
            dimensions.append("salesman")
        
        if QueryPlanService._has_phrase(question, r'\b(customer|client|account|buyer)\b'):
            dimensions.append("customer")
        
        if QueryPlanService._has_phrase(question, r'\b(product|item|sku|product code|brand)\b'):
            dimensions.append("product")
        
        return dimensions if dimensions else ["salesman"]  # Default to salesman
    
    @staticmethod
    def _detect_time_filter(question: str) -> Optional[TimeFilter]:
        """Detect time filter from question using proper phrase matching."""
        # Last completed quarter
        if QueryPlanService._has_phrase(question, r'\blast\s+completed\s+quarter\b'):
            return TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
        
        # Last quarter (should be last completed)
        if QueryPlanService._has_phrase(question, r'\blast\s+quarter\b'):
            return TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
        
        # This quarter
        if QueryPlanService._has_phrase(question, r'\bthis\s+quarter\b'):
            return TimeFilter(mode=TimeMode.THIS_QUARTER)
        
        # Last month
        if QueryPlanService._has_phrase(question, r'\blast\s+month\b'):
            return TimeFilter(mode=TimeMode.LAST_MONTH)
        
        # This month
        if QueryPlanService._has_phrase(question, r'\bthis\s+month\b'):
            return TimeFilter(mode=TimeMode.THIS_MONTH)
        
        # Last N months
        match = re.search(r'\blast\s+(\d+)\s+months?\b', question, re.IGNORECASE)
        if match:
            return TimeFilter(mode=TimeMode.LAST_N_MONTHS, n_months=int(match.group(1)))
        
        # Specific quarter (Q3 2024)
        match = re.search(r'\bq([1-4])\s+(\d{4})\b', question, re.IGNORECASE)
        if match:
            return TimeFilter(
                mode=TimeMode.SPECIFIC_QUARTER,
                quarter=int(match.group(1)),
                year=int(match.group(2))
            )
        
        # Specific month (July 2024)
        month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        for i, month in enumerate(month_names, 1):
            match = re.search(rf'\b{month}\s+(\d{{4}})\b', question, re.IGNORECASE)
            if match:
                return TimeFilter(
                    mode=TimeMode.SPECIFIC_MONTH,
                    month=i,
                    year=int(match.group(1))
                )
        
        return None  # No time filter detected


