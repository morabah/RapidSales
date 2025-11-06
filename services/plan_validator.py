"""
Plan Validator
Validates QueryPlans before execution to prevent bad plans from executing.
"""

from .query_plan_service import QueryPlan


# Allowed values for validation
ALLOWED_METRICS = {"sales_value", "total_revenue", "revenue", "units", "avg_price", "avg_order_value", "transaction_count"}
ALLOWED_AGGS = {"sum", "avg", "count", "max", "min"}
ALLOWED_DIMS = {"salesman", "customer", "product", "brand"}  # Base dimension names, will check column mapping

# Allowed SQL functions (for SQL path validation)
ALLOWED_SQL_FUNCTIONS = {
    "SUM", "AVG", "COUNT", "MAX", "MIN", "ROUND", "DATE", "DATE_TRUNC", 
    "EXTRACT", "YEAR", "MONTH", "DAY", "INTERVAL", "CAST"
}


def validate_plan(plan: QueryPlan, df_columns: set[str], column_mapping: dict = None) -> tuple[bool, str]:
    """
    Validate a QueryPlan before execution.
    
    Args:
        plan: QueryPlan to validate
        df_columns: Set of actual column names in the DataFrame
        column_mapping: Optional column mapping dictionary (e.g., {'salesman': 'Sales Rep'})
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Validate metric
    if plan.metric not in ALLOWED_METRICS:
        return False, f"Unsupported metric: {plan.metric}. Allowed: {ALLOWED_METRICS}"
    
    # Validate aggregation
    if plan.aggregation not in ALLOWED_AGGS:
        return False, f"Unsupported aggregation: {plan.aggregation}. Allowed: {ALLOWED_AGGS}"
    
    # Validate dimensions exist in DataFrame
    if column_mapping:
        # Check if mapped columns exist
        for dim in plan.dimensions or []:
            mapped_col = column_mapping.get(dim)
            if mapped_col and mapped_col not in df_columns:
                return False, f"Unknown dimension column: {dim} -> {mapped_col}"
    else:
        # Check base dimension names (fallback)
        for dim in plan.dimensions or []:
            if dim not in ALLOWED_DIMS:
                # Allow if it's a direct column name
                if dim not in df_columns:
                    return False, f"Unknown dimension: {dim}"
    
    # Ensure date column exists when time filter is applied
    if plan.time_filter:
        date_col = column_mapping.get('date') if column_mapping else 'date'
        if date_col not in df_columns:
            # Try common date column names
            date_cols = [col for col in df_columns if 'date' in col.lower() or 'time' in col.lower()]
            if not date_cols:
                return False, f"Missing date column for time filter. Available columns: {list(df_columns)[:10]}"
    
    # Clamp limit to prevent excessive data
    if plan.limit is None or plan.limit > 1000:
        plan.limit = 1000
    
    # Validate limit is reasonable
    if plan.limit < 1:
        plan.limit = 10  # Minimum reasonable limit
    
    # Validate that we have at least one dimension or metric
    if not plan.dimensions and plan.metric == "sales_value":
        # This is OK - can be a total query
        pass
    
    return True, ""


def validate_sql_columns(sql: str, allowed_columns: set[str]) -> tuple[bool, str]:
    """
    Validate that SQL only references allowed columns.
    
    Args:
        sql: SQL query string
        allowed_columns: Set of allowed column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Extract column references from SQL (simple check)
    # Look for column names in SELECT, WHERE, GROUP BY, ORDER BY
    sql_upper = sql.upper()
    
    # Check for SELECT * without LIMIT or GROUP BY (potentially unsafe)
    if "SELECT *" in sql_upper and "LIMIT" not in sql_upper and "GROUP BY" not in sql_upper:
        return False, "SELECT * without LIMIT or GROUP BY is not allowed"
    
    # Note: Full column validation would require SQL parsing
    # For now, this is a basic check
    return True, ""

