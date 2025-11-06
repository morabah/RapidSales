"""
AI Service
Handles AI queries using Gemini API with fallback to local analysis.
"""

import json
import time
import re
import google.generativeai as genai
import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .error_handling_service import ErrorHandlingService, ErrorCategory
from .config_service import ConfigService
from .query_plan_service import QueryPlanService, QueryPlan, TimeMode
from .time_resolver import TimeResolver
# TimeFilterService kept only as ultimate fallback (deprecated)
from .time_filter_service import TimeFilterService
from .intent_parser_service import IntentParserService
from .plan_validator import validate_plan
from .data_formatting_service import render_placeholders, has_unfilled_placeholders
from .sql_generation_service import SQLGenerationService
from prompts.ai_prompts import get_bi_analyst_prompt


class AIService:
    """Service for AI-powered business intelligence queries."""
    
    def __init__(self):
        self.use_gemini = self._check_gemini_availability()
        self.intent_parser = IntentParserService()
        self.sql_service = SQLGenerationService()
    
    def _check_gemini_availability(self) -> bool:
        """Check if Gemini API is available."""
        api_key = ConfigService.get_gemini_api_key()
        if api_key:
            try:
                genai.configure(api_key=api_key)
                return True
            except Exception:
                pass
        return False
    
    def query(
        self,
        question: str,
        data_summary: dict,
        df=None,
        insights=None
    ) -> str | dict:
        """
        Query AI with business question using SQL generation (primary) or RAG (fallback).
        
        Primary path: Text-to-SQL using DuckDB (handles any phrasing)
        Fallback path: RAG with pattern matching (legacy compatibility)
        
        Args:
            question: Business question to answer
            data_summary: Prepared data summary dictionary
            df: Optional DataFrame for local fallback
            insights: Optional insights dictionary
            
        Returns:
            AI response as markdown string, or dict with 'text' and 'mom_data' for month-over-month queries
        """
        try:
            # Try SQL/deterministic path first (even without Gemini)
            if df is not None and not df.empty:
                try:
                    # Ensure columns is a dictionary
                    columns = insights.get('columns', {}) if isinstance(insights.get('columns', {}), dict) else {}
                    if not columns:
                        columns = data_summary.get('column_mapping', {}) if isinstance(data_summary.get('column_mapping', {}), dict) else {}
                    
                    result = self.sql_service.process_query(question, df, columns)
                    
                    # Debug: log what we got
                    print(f"[AIService] SQL service result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                    
                    # Ensure result is a dictionary
                    if not isinstance(result, dict):
                        ErrorHandlingService.log_error(
                            f"SQL service returned non-dict result: {type(result).__name__}",
                            category=ErrorCategory.DATA_PROCESSING
                        )
                        # Fall through to RAG path
                    elif "data" in result:
                        # SQL path succeeded - format and return
                        sql_result_df = result["data"]
                        sql_used = result.get("sql", "")
                        observability = result.get("observability", {})
                        
                        # Guardrail: Don't fall back to narrative for computable queries with results
                        if not sql_result_df.empty:
                            # Format result for display with period label and month total if available
                            # Extract month_total from result metadata if available
                            month_total = None
                            if hasattr(sql_result_df, 'attrs') and 'month_total' in sql_result_df.attrs:
                                month_total = sql_result_df.attrs['month_total']
                            else:
                                # Fallback: get from observability
                                month_total = observability.get("month_total")
                            
                            formatted_result = self._format_sql_result(
                                sql_result_df, 
                                question, 
                                sql_used,
                                period_label=observability.get("period_label"),
                                month_total=month_total,
                                executor=observability.get("executor"),
                                peak_period=observability.get("peak_period"),
                                peak_value=observability.get("peak_value"),
                                period_bounds=observability.get("period_bounds")
                            )
                            
                            # Add CSV/Copy functionality if it's a table result
                            if isinstance(sql_result_df, pd.DataFrame) and not sql_result_df.empty:
                                # Add CSV download button (handled in app.py)
                                pass
                            
                            return formatted_result
                        # If empty, let error handling below provide context-specific message
                    elif "needs_clarification" in result:
                        return f"## Clarification Needed\n\n{result['needs_clarification']}"
                    elif "error" in result:
                        # Check if error is due to empty result with required columns present
                        # If so, provide context-specific error message
                        error_msg = result.get('error', '')
                        
                        # Check if we have required columns for months-with-single-salesman query
                        # Use LLM/heuristic intent understanding
                        plan = self.sql_service._detect_deterministic_pattern(question, columns, df if df is not None else pd.DataFrame())
                        if plan and plan.get('intent') == 'constraint_timeseries':
                            cols = insights.get('column_mapping', {})
                            date_col = cols.get('date')
                            value_col = cols.get('amount')
                            salesman_col = cols.get('salesman')
                            
                            if date_col and value_col and salesman_col:
                                # Check if global date filter might be excluding all data
                                global_filter_enabled = st.session_state.get('global_date_filter_enabled', False)
                                if global_filter_enabled:
                                    global_range = st.session_state.get('global_date_filter_range')
                                    if global_range:
                                        return f"## No Results\n\nNo month found where exactly one salesman recorded revenue. The global date filter ({global_range[0]} → {global_range[1]}) may be excluding all relevant months.\n\n💡 **Clear the global date filter** and try again."
                                    else:
                                        return "## No Results\n\nNo month found where exactly one salesman recorded revenue. The global date filter may be excluding all relevant months. **Clear the filter** and try again."
                                else:
                                    return "## No Results\n\nNo month found where exactly one salesman recorded revenue."
                        
                        # SQL path failed, fall through to RAG path
                        ErrorHandlingService.log_error(
                            f"SQL generation error: {result['error']}",
                            category=ErrorCategory.DATA_PROCESSING
                        )
                except Exception as e:
                    # SQL path failed, fall through to RAG path
                    import traceback
                    error_trace = traceback.format_exc()
                    ErrorHandlingService.log_error(
                        e,
                        category=ErrorCategory.DATA_PROCESSING
                    )
                    # Also print to console for debugging
                    print(f"[AIService SQL Error] {e}")
                    print(f"[AIService SQL Traceback]\n{error_trace}")
            
            # Fallback to RAG path (legacy pattern-based)
            if self.use_gemini:
                return self._query_gemini(question, data_summary, df, insights)
            else:
                return self._query_local_fallback(question, df, insights)
        except Exception as e:
            error_info = ErrorHandlingService.process_error(
                e,
                context='query_ai',
                category=ErrorCategory.API if self.use_gemini else ErrorCategory.DATA,
                details={'question': question[:100]}  # Truncate for logging
            )
            ErrorHandlingService.log_error(error_info)
            # Return user-friendly error message
            return f"## Error\n\n{ErrorHandlingService.display_error(error_info)}"
    
    def _format_sql_result(
        self,
        df: pd.DataFrame,
        question: str,
        sql: str,
        period_label: str | None = None,
        month_total: float | None = None,
        executor: str | None = None,
        peak_period: str | None = None,
        peak_value: float | None = None,
        period_bounds: tuple[str, str] | None = None
    ) -> str:
        """
        Format SQL result DataFrame into readable markdown.
        Includes global filter badge, period label, and CSV/Copy options.
        
        Args:
            df: Result DataFrame from SQL query
            question: Original question (for context)
            sql: SQL query that was executed
            period_label: Optional period label with exact bounds (e.g., "Q2-2025 (2025-04-01 → 2025-06-30)")
            month_total: Optional total for share/ranking queries
            executor: Name of executor used (for contextual messaging)
            peak_period: Optional period identifier flagged as the peak (timeseries metadata)
            peak_value: Optional numeric value associated with peak_period
            period_bounds: Optional tuple of (start, end_exclusive) bounds
        
        Returns:
            Formatted markdown string
        """
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            return f"## Query Error\n\nInvalid result type: {type(df).__name__}"
        
        if df.empty:
            period_info = f" ({period_label})" if period_label else ""
            return f"## Query Results{period_info}\n\nNo data found for this query."
        
        output = ["## Query Results\n"]
        
        # Kill-switch banner: show if LLM is degraded
        if st.session_state.get('llm_degraded', False):
            output.append(f"<span style='background: rgba(245,158,11,0.2); padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; color: #FDE68A;'>⚠️ AI parsing temporarily degraded. Using heuristic fallback.</span>\n\n")
        
        # Add global date filter badge if active (tiny chip)
        if st.session_state.get('global_date_filter_enabled', False):
            global_range = st.session_state.get('global_date_filter_range')
            if global_range:
                output.append(f"<span style='background: rgba(245,158,11,0.15); padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; color: #FDE68A;'>📅 Global filter active</span>\n\n")
        
        # Add period label with exact bounds if available
        if period_label:
            output.append(f"**Period:** {period_label}\n\n")
        elif period_bounds:
            output.append(f"**Period:** {period_bounds[0]} → {period_bounds[1]}\n\n")
        
        # Memory cap: check if result needs truncation for display
        if len(df) > 1000:
            output.append(f"*Showing first 1000 rows (total: {len(df)}). Download CSV for full results.*\n\n")
            df = df.head(1000)
        
        # Format dates and currency before displaying
        df_formatted = df.copy()
        
        # Format date columns (remove timestamp, show just date)
        for col in df_formatted.columns:
            if 'date' in col.lower() or 'start' in col.lower() or 'end' in col.lower():
                if pd.api.types.is_datetime64_any_dtype(df_formatted[col]):
                    df_formatted[col] = df_formatted[col].dt.date
                elif pd.api.types.is_object_dtype(df_formatted[col]):
                    # Try to parse and format
                    try:
                        dates = pd.to_datetime(df_formatted[col], errors='coerce')
                        df_formatted[col] = dates.dt.date
                    except:
                        pass
        
        # Format currency columns using CurrencyFormattingService (render layer: 2 decimals, symbol from settings)
        from .currency_formatting_service import CurrencyFormattingService
        currency_settings = CurrencyFormattingService.get_currency_settings()
        
        # Detect currency/revenue columns (format at render layer with 2 decimals, symbol from settings)
        currency_keywords = ['revenue', 'total', 'amount', 'value', 'sales', 'change', 'delta', 'aov']
        # Transaction columns should remain as integers (not formatted as currency)
        transaction_keywords = ['transactions', 'count']
        # Percent/share columns should be formatted with "%" symbol
        percent_keywords = ['pct_share', 'share', 'percentage', 'pct']
        
        for col in df_formatted.columns:
            col_lower = col.lower()
            # Format percent columns with "%" symbol (render layer, not SQL)
            if any(keyword in col_lower for keyword in percent_keywords) and pd.api.types.is_numeric_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else x
                )
            # Format currency columns (round to 2 decimals, apply currency symbol)
            elif any(keyword in col_lower for keyword in currency_keywords) and pd.api.types.is_numeric_dtype(df_formatted[col]):
                # Format as currency (rounding happens in format_currency)
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: CurrencyFormattingService.format_currency(x) if pd.notna(x) else x
                )
            # Transaction counts: format as integers with thousand separators (no currency symbol)
            elif any(keyword in col_lower for keyword in transaction_keywords) and pd.api.types.is_numeric_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else x
                )
        
        # Add execution details
        if executor:
            output.append(f"*Executor used: {executor}*\n\n")
        # Add SQL provenance (for debugging - can be toggled in UI)
        output.append(f"*SQL executed: `{sql}`*\n\n")
        
        # Format as table
        output.append("**Results:**\n")
        try:
            output.append(df_formatted.to_markdown(index=False))
        except ImportError as e:
            # Fallback: format as simple text table if tabulate is not available
            ErrorHandlingService.log_error(
                f"Tabulate not available, using fallback formatting: {e}",
                category=ErrorCategory.DATA_PROCESSING
            )
            # Simple text table format
            output.append("```")
            output.append(df.to_string(index=False))
            output.append("```")
        except Exception as e:
            ErrorHandlingService.log_error(
                e,
                category=ErrorCategory.DATA_PROCESSING
            )
            # Fallback: simple text representation
            output.append("```")
            output.append(df.to_string(index=False))
            output.append("```")
        
        # For ranking queries: show month total explicitly and period label
        if executor == "ranking" and month_total is not None:
            from .currency_formatting_service import CurrencyFormattingService
            formatted_month_total = CurrencyFormattingService.format_currency(month_total)
            output.append(f"\n**Month total:** {formatted_month_total}")
        
        # Add period label under the table if available
        if period_label:
            output.append(f"\n*{period_label}*")
        
        # If it's a simple aggregate query (e.g., total revenue for a quarter), format nicely
        question_lower = question.lower()
        
        # Check if result has total_revenue and date columns (quarter query)
        if 'total_revenue' in df.columns and any('quarter' in col.lower() or 'start' in col.lower() or 'end' in col.lower() for col in df.columns):
            if len(df) == 1:
                row = df.iloc[0]
                total_rev = row.get('total_revenue', 0)
                
                # Format revenue with currency
                formatted_rev = CurrencyFormattingService.format_currency(total_rev)
                
                # Get date range
                start_col = next((c for c in df.columns if 'start' in c.lower() or 'quarter_start' in c.lower()), None)
                end_col = next((c for c in df.columns if 'end' in c.lower() or 'quarter_end' in c.lower()), None)
                
                if start_col and end_col:
                    start_date = row.get(start_col)
                    end_date = row.get(end_col)
                    
                    # Format dates (remove timestamp if present)
                    if pd.api.types.is_datetime64_any_dtype(type(start_date)) or isinstance(start_date, pd.Timestamp):
                        start_date = start_date.date() if hasattr(start_date, 'date') else str(start_date).split()[0]
                    if pd.api.types.is_datetime64_any_dtype(type(end_date)) or isinstance(end_date, pd.Timestamp):
                        end_date = end_date.date() if hasattr(end_date, 'date') else str(end_date).split()[0]
                    
                    output.append(f"\n**Total Revenue:** {formatted_rev}")
                    output.append(f"\n**Period:** {start_date} to {end_date}")
        
        # If it's a monthly timeseries breakdown, add summary (structural gating by executor)
        elif executor == "timeseries":
            # Find total/revenue column
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if 'is_peak' in numeric_cols:
                numeric_cols = [c for c in numeric_cols if c != 'is_peak']
            if len(numeric_cols) > 0:
                total_col = numeric_cols[0]
                total = df[total_col].sum()
                
                formatted_total = CurrencyFormattingService.format_currency(total)
                output.append(f"\n**Total:** {formatted_total}")
                if peak_period and peak_value is not None:
                    formatted_peak = CurrencyFormattingService.format_currency(peak_value)
                    output.append(f"\n**Peak:** {peak_period} — {formatted_peak}")
                elif 'month' in df.columns or any('month' in str(c).lower() for c in df.columns):
                    month_col = next((c for c in df.columns if 'month' in str(c).lower()), None)
                    if month_col:
                        max_idx = df[total_col].idxmax()
                        if pd.notna(max_idx):
                            max_row = df.loc[max_idx]
                            formatted_max = CurrencyFormattingService.format_currency(max_row[total_col])
                            output.append(f"\n**Highest Month:** {max_row[month_col]} — {formatted_max}")

        # For constraint_timeseries, append count of months (structural, not keyword-based)
        if executor == "constraint_timeseries":
            try:
                output.append(f"\n**Count of months:** {len(df)}")
            except Exception:
                pass
        
        return '\n'.join(output)
    
    def _prepare_relevant_context(
        self,
        question: str,
        df: pd.DataFrame,
        insights: dict,
        data_summary: dict
    ) -> dict:
        """
        Prepare relevant data context based on question intent (RAG approach).
        
        Uses QueryPlan for safe, deterministic data extraction.
        All numbers are computed locally - LLM only receives aggregates.
        """
        import pandas as pd
        
        # Initialize context early to avoid UnboundLocalError
        context = {}
        
        # 1) Parse question into structured plan (LLM-based if available, else pattern-based)
        try:
            plan = self.intent_parser.parse_question(question)
        except Exception:
            # Fallback to pattern-based parsing
            plan = QueryPlanService.parse_question(question)
        
        # 2) Validate plan before execution
        cols = data_summary.get('column_mapping', {}) or {}
        df_columns = set(df.columns) if df is not None else set()
        is_valid, error_msg = validate_plan(plan, df_columns, cols)
        
        if not is_valid:
            context['error'] = error_msg
            context['needs_clarification'] = True
            return context
        
        # Get currency code for consistent formatting
        currency_code = st.session_state.get('currency_code', ConfigService.DEFAULT_CURRENCY_CODE)
        currency_symbol = ConfigService.CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        cols = data_summary.get('column_mapping', {})
        date_col = cols.get('date')
        
        # CRITICAL: Apply time filter FIRST - use canonical TimeResolver for consistent bounds
        # Also apply global date filter if enabled (intersection)
        filtered_df = df
        time_filter_applied_data = None
        filtered_date_range = None
        period_label_with_bounds = None
        global_filter_applied = False
        
        # Apply global date filter if enabled (intersection with question-level filter)
        global_filter_range = st.session_state.get('global_date_filter_range')
        global_filter_enabled = st.session_state.get('global_date_filter_enabled', False)
        
        if global_filter_enabled and global_filter_range and df is not None:
            date_col = cols.get('date')
            if date_col and date_col in df.columns:
                from datetime import datetime as dt
                global_start = pd.Timestamp(global_filter_range[0])
                global_end = pd.Timestamp(global_filter_range[1]) + pd.Timedelta(days=1)  # End-exclusive
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                mask = (df_copy[date_col] >= global_start) & (df_copy[date_col] < global_end)
                filtered_df = df_copy[mask].copy()
                global_filter_applied = True
        
        # Apply question-level time filter (intersection with global filter)
        if plan.time_filter and filtered_df is not None:
            date_col = cols.get('date')
            if date_col and date_col in df.columns:
                # Get dataset max date as reference (not "today")
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if not dates.empty:
                    max_date = dates.max().to_pydatetime()
                    
                    # Use canonical TimeResolver for consistent bounds
                    start_date, end_date_exclusive, period_label_with_bounds = TimeResolver.resolve_time_window(
                        plan.time_filter,
                        max_date,
                        dataset_max_date=max_date
                    )
                    
                    # Filter DataFrame using TimeResolver
                    if start_date and end_date_exclusive:
                        filtered_df = TimeResolver.filter_dataframe(df, date_col, start_date, end_date_exclusive)
                        
                        if not filtered_df.empty:
                            # Get actual date range from filtered data
                            filtered_dates = pd.to_datetime(filtered_df[date_col], errors='coerce').dropna()
                            if not filtered_dates.empty:
                                start_date_actual = filtered_dates.min().date()
                                end_date_actual = filtered_dates.max().date()
                                
                                time_filter_applied_data = {
                                    'mode': plan.time_filter.mode.value,
                                    'records_after_filter': len(filtered_df),
                                    'n_months': plan.time_filter.n_months if plan.time_filter.n_months else None,
                                    'date_range': {
                                        'start': str(start_date_actual),
                                        'end': str(end_date_actual)
                                    },
                                    'period_label': period_label_with_bounds
                                }
                                filtered_date_range = {
                                    'start': str(start_date_actual),
                                    'end': str(end_date_actual)
                                }
                    else:
                        # Fallback: manually resolve time window if TimeResolver returns None
                        # This should rarely happen, but provides safety
                        try:
                            start_date, end_date_exclusive, _ = TimeResolver.resolve_time_window(
                                plan.time_filter,
                                max_date,
                                dataset_max_date=max_date
                            )
                            if start_date and end_date_exclusive:
                                filtered_df = TimeResolver.filter_dataframe(df, date_col, start_date, end_date_exclusive)
                            else:
                                filtered_df = df  # No filter applied
                        except Exception:
                            # Ultimate fallback: use TimeFilterService (deprecated, but kept for safety)
                            filtered_df = TimeFilterService.filter_by_time_period(df, date_col, plan.time_filter, max_date)
        
        # CRITICAL: All computations MUST use filtered_df (not df)
        # Compute totals from FILTERED data
        amount_col = cols.get('amount')
        if amount_col and filtered_df is not None and not filtered_df.empty:
            vals = pd.to_numeric(filtered_df[amount_col], errors='coerce').fillna(0)
            total_revenue = float(vals.sum())
            avg_order_value = float(vals.mean()) if len(vals) > 0 else 0.0
        else:
            # Fallback to original data if no filter or no amount column
            total_revenue = data_summary.get('total_revenue', 0)
            avg_order_value = data_summary.get('avg_order_value', 0)
        
        # Get date range from filtered data or original
        if filtered_df is not None and not filtered_df.empty and date_col:
            dates = pd.to_datetime(filtered_df[date_col], errors='coerce').dropna()
            if not dates.empty:
                start_date = dates.min().date()
                end_date = dates.max().date()
            else:
                # Fallback to original dataset range
                orig_dates = pd.to_datetime(df[date_col], errors='coerce').dropna() if df is not None and date_col in df.columns else pd.Series()
                if not orig_dates.empty:
                    start_date = orig_dates.min().date()
                    end_date = orig_dates.max().date()
                else:
                    date_range = data_summary.get('date_range', {})
                    start_date = date_range.get('start') if isinstance(date_range.get('start'), str) else None
                    end_date = date_range.get('end') if isinstance(date_range.get('end'), str) else None
        else:
            # Use original dataset date range
            date_range = data_summary.get('date_range', {})
            start_date = date_range.get('start') if isinstance(date_range.get('start'), str) else None
            end_date = date_range.get('end') if isinstance(date_range.get('end'), str) else None
        
        # Build context dictionary
        context = {
            'total_records': len(filtered_df) if filtered_df is not None else data_summary.get('total_records', 0),
            'date_range': filtered_date_range if filtered_date_range else {
                'start': str(start_date) if start_date else '',
                'end': str(end_date) if end_date else ''
            },
            'columns': cols,
            'total_revenue': total_revenue,  # From filtered_df
            'avg_order_value': avg_order_value,  # From filtered_df
            'currency': {
                'code': currency_code,
                'symbol': currency_symbol
            },
            'query_plan': {
                'dimensions': plan.dimensions,
                'include_top': plan.include_top,
                'include_bottom': plan.include_bottom,
                'limit': plan.limit
            }
        }
        
        # Add time filter info if available
        if time_filter_applied_data:
            context['time_filter_applied'] = time_filter_applied_data
        
        # SALESMAN QUERIES - Use authoritative InsightService method
        # CRITICAL: Use filtered_df (period-filtered) not df (all-time)
        if 'salesman' in plan.dimensions:
            # Use InsightService for authoritative, tie-safe computation of full ranking
            from .insight_service import InsightService
            
            insight_service = InsightService()
            cols_for_perf = insights.get('columns', {}) or cols
            
            try:
                # Get authoritative performance data from FILTERED DataFrame (not all-time)
                k = min(plan.limit, 25)  # Cap at 25 for token efficiency
                # Use filtered_df if available and not empty, otherwise use df
                df_for_perf = filtered_df if filtered_df is not None and not filtered_df.empty else df
                perf = insight_service.get_salesmen_performance(
                    df_for_perf,
                    cols_for_perf,
                    k=k,
                    cap=50
                )
                
                # Always include rep_count
                context['all_salesmen_count'] = perf.get('rep_count', 0)
                
                # Map question intent to data
                asked_bottom = plan.include_bottom
                asked_top = plan.include_top
                
                if asked_bottom:
                    context['bottom_salesmen'] = perf.get('bottom_k', {})
                if asked_top:
                    context['top_salesmen'] = perf.get('top_k', {})
                
                # Include full ranking if available (only when rep_count <= 50)
                if 'all' in perf:
                    context['all_salesmen'] = perf['all']
                
            except Exception as e:
                # Fallback to summary data
                if 'top_salesmen' in data_summary:
                    context['top_salesmen'] = data_summary['top_salesmen']
        
        # CUSTOMER QUERIES - Use QueryPlan with FILTERED DataFrame
        if 'customer' in plan.dimensions:
            customer_col = cols.get('customer')
            amount_col = cols.get('amount')
            
            # CRITICAL: Use filtered_df (period-filtered), not df (all-time)
            df_for_customer = filtered_df if filtered_df is not None and not filtered_df.empty else df
            
            if customer_col and amount_col and df_for_customer is not None and not df_for_customer.empty:
                try:
                    vals = pd.to_numeric(df_for_customer[amount_col], errors='coerce').fillna(0)
                    customer_data = df_for_customer.groupby(customer_col, dropna=False)[amount_col].agg([
                        ('total_sales', 'sum'),
                        ('num_orders', 'count')
                    ]).round(ConfigService.DEFAULT_DECIMAL_PLACES)
                    
                    ascending = not plan.include_top or (plan.include_bottom and not plan.include_top)
                    customer_data = customer_data.fillna(0)
                    
                    # Handle empty results
                    if len(customer_data) == 0:
                        return context
                    
                    # Sort with explicit tie-breaker: sales value, then name
                    customer_data_reset = customer_data.reset_index()
                    customer_data_sorted = customer_data_reset.sort_values(
                        ['total_sales', customer_col],
                        ascending=[ascending, True]
                    )
                    customer_data = customer_data_sorted.set_index(customer_col)
                    
                    # Cap at 25 for token efficiency
                    max_items = min(plan.limit, 25)
                    total_count = len(customer_data)
                    
                    if plan.include_bottom and not plan.include_top:
                        context['bottom_customers'] = customer_data.tail(min(max_items, total_count)).to_dict('index')
                        context['all_customers_count'] = total_count
                    elif plan.include_top and not plan.include_bottom:
                        context['top_customers'] = customer_data.head(min(max_items, total_count)).to_dict('index')
                        context['all_customers_count'] = total_count
                    else:
                        context['top_customers'] = customer_data.head(min(max_items, total_count)).to_dict('index')
                        context['all_customers_count'] = total_count
                        if total_count > max_items:
                            context['bottom_customers'] = customer_data.tail(min(max_items, total_count)).to_dict('index')
                    
                    if 'all' in question.lower() and total_count <= 50:
                        context['all_customers'] = customer_data.head(50).to_dict('index')
                except Exception:
                    if 'top_customers' in data_summary:
                        context['top_customers'] = data_summary.get('top_customers', [])
        
        # PRODUCT/BRAND QUERIES - Use QueryPlan with FILTERED DataFrame
        if 'product' in plan.dimensions:
            # Try to find brand/product column
            product_col = cols.get('product') or cols.get('brand')  # Support both 'product' and 'brand'
            amount_col = cols.get('amount')
            
            # CRITICAL: Use filtered_df (period-filtered), not df (all-time)
            df_for_product = filtered_df if filtered_df is not None and not filtered_df.empty else df
            
            if product_col and amount_col and df_for_product is not None and not df_for_product.empty:
                try:
                    vals = pd.to_numeric(df_for_product[amount_col], errors='coerce').fillna(0)
                    product_data = df_for_product.groupby(product_col, dropna=False)[amount_col].agg([
                        ('total_sales', 'sum'),
                        ('num_orders', 'count')
                    ]).round(ConfigService.DEFAULT_DECIMAL_PLACES)
                    
                    ascending = not plan.include_top or (plan.include_bottom and not plan.include_top)
                    product_data = product_data.fillna(0)
                    
                    # Handle empty results
                    if len(product_data) == 0:
                        return context
                    
                    # Sort with explicit tie-breaker: sales value, then name
                    product_data_reset = product_data.reset_index()
                    product_data_sorted = product_data_reset.sort_values(
                        ['total_sales', product_col],
                        ascending=[ascending, True]
                    )
                    product_data = product_data_sorted.set_index(product_col)
                    
                    # Cap at plan.limit (e.g., "top 3" = 3 items)
                    max_items = min(plan.limit, 25)
                    total_count = len(product_data)
                    
                    if plan.include_bottom and not plan.include_top:
                        context['bottom_products'] = product_data.tail(min(max_items, total_count)).to_dict('index')
                        context['all_products_count'] = total_count
                    elif plan.include_top and not plan.include_bottom:
                        context['top_products'] = product_data.head(min(max_items, total_count)).to_dict('index')
                        context['all_products_count'] = total_count
                    else:
                        context['top_products'] = product_data.head(min(max_items, total_count)).to_dict('index')
                        context['all_products_count'] = total_count
                        if total_count > max_items:
                            context['bottom_products'] = product_data.tail(min(max_items, total_count)).to_dict('index')
                except Exception:
                    if 'top_products' in data_summary:
                        context['top_products'] = data_summary.get('top_products', [])
        
        # Handle comparative queries (e.g., "compare last quarter to previous quarter")
        question_lower = question.lower()
        if any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'previous', 'improved', 'improvement', 'change']):
            # Check if question asks for quarter comparison
            if 'quarter' in question_lower and ('last' in question_lower or 'completed' in question_lower):
                try:
                    # Get last completed quarter and previous quarter data
                    from .insight_service import InsightService
                    
                    insight_service = InsightService()
                    cols_for_perf = insights.get('columns', {}) or cols
                    salesman_col = cols_for_perf.get('salesman')
                    amount_col = cols_for_perf.get('amount')
                    date_col = cols_for_perf.get('date')
                    
                    if salesman_col and amount_col and date_col and df is not None:
                        # Get dataset max date (CRITICAL: use dataset max, not "today")
                        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                        if not dates.empty:
                            max_date = dates.max().to_pydatetime()
                            
                            # Last completed quarter (proper quarter bounds) - use TimeResolver
                            last_q_start, last_q_end_exclusive = TimeResolver.last_completed_quarter_bounds(max_date)
                            last_q_end_inclusive = last_q_end_exclusive - relativedelta(days=1)
                            
                            # Filter DataFrame for last completed quarter using TimeResolver
                            last_q_df = TimeResolver.filter_dataframe(df, date_col, last_q_start, last_q_end_exclusive)
                            
                            # Previous quarter (before last completed) - proper quarter bounds
                            prev_q_start, prev_q_end_exclusive = TimeResolver.previous_quarter_bounds(max_date)
                            prev_q_end_inclusive = prev_q_end_exclusive - relativedelta(days=1)
                            
                            # Filter DataFrame for previous quarter using TimeResolver
                            prev_q_df = TimeResolver.filter_dataframe(df, date_col, prev_q_start, prev_q_end_exclusive)
                            
                            # Compute salesmen performance for both quarters (using filtered DataFrames)
                            if not last_q_df.empty:
                                last_q_perf = insight_service.get_salesmen_performance(
                                    last_q_df, cols_for_perf, k=100, cap=1000
                                )
                            else:
                                last_q_perf = {'all': {}}
                            
                            if not prev_q_df.empty:
                                prev_q_perf = insight_service.get_salesmen_performance(
                                    prev_q_df, cols_for_perf, k=100, cap=1000
                                )
                            else:
                                prev_q_perf = {'all': {}}
                            
                            # Calculate percentage changes with zero-baseline handling
                            comparison_data = {}
                            last_q_all = last_q_perf.get('all', {})
                            prev_q_all = prev_q_perf.get('all', {})
                            
                            # Get all unique salesmen from both quarters
                            all_salesmen = set(last_q_all.keys()) | set(prev_q_all.keys())
                            
                            for salesman in all_salesmen:
                                last_q_total = last_q_all.get(salesman, {}).get('total', 0.0)
                                prev_q_total = prev_q_all.get(salesman, {}).get('total', 0.0)
                                
                                # Calculate absolute change
                                abs_change = last_q_total - prev_q_total
                                
                                # Calculate percentage change with zero-baseline rule
                                if prev_q_total > 0:
                                    # Normal case: previous quarter > 0
                                    pct_change = (abs_change / prev_q_total) * 100
                                    pct_change_str = f"{pct_change:.1f}%"
                                    pct_change_valid = True
                                elif last_q_total > 0 and prev_q_total == 0:
                                    # Growth from zero: report absolute delta, % = N/A
                                    pct_change = float('inf')
                                    pct_change_str = "N/A (prev: 0.00)"
                                    pct_change_valid = False
                                else:
                                    # Both zero: no change
                                    pct_change = 0.0
                                    pct_change_str = "0.0%"
                                    pct_change_valid = True
                                
                                comparison_data[salesman] = {
                                    'last_quarter': round(last_q_total, 2),
                                    'previous_quarter': round(prev_q_total, 2),
                                    'change': round(abs_change, 2),
                                    'pct_change': pct_change,
                                    'pct_change_str': pct_change_str,
                                    'pct_change_valid': pct_change_valid
                                }
                            
                            # Sort by absolute change first (most improved = largest absolute change)
                            # Then by percentage change if previous > 0
                            sorted_comparison = sorted(
                                comparison_data.items(),
                                key=lambda x: (
                                    x[1]['change'] if x[1]['pct_change_valid'] else x[1]['change'],  # Absolute change primary
                                    x[1]['pct_change'] if x[1]['pct_change'] != float('inf') and x[1]['pct_change_valid'] else -999999
                                ),
                                reverse=True
                            )
                            
                            # Format for context
                            most_improved = None
                            if sorted_comparison:
                                top_item = sorted_comparison[0]
                                most_improved = {
                                    'salesman': top_item[0],
                                    'previous_quarter': top_item[1]['previous_quarter'],
                                    'last_quarter': top_item[1]['last_quarter'],
                                    'change': top_item[1]['change'],
                                    'pct_change_str': top_item[1]['pct_change_str'],
                                    'pct_change_valid': top_item[1]['pct_change_valid']
                                }
                            
                            context['quarter_comparison'] = {
                                'last_quarter': {
                                    'label': f"Q{(last_q_start.month - 1) // 3 + 1}-{last_q_start.year}",
                                    'start': str(last_q_start.date()),
                                    'end': str(last_q_end_inclusive.date())
                                },
                                'previous_quarter': {
                                    'label': f"Q{(prev_q_start.month - 1) // 3 + 1}-{prev_q_start.year}",
                                    'start': str(prev_q_start.date()),
                                    'end': str(prev_q_end_inclusive.date())
                                },
                                'comparison': {k: {
                                    'previous': v['previous_quarter'],
                                    'current': v['last_quarter'],
                                    'change': v['change'],
                                    'pct_change': v['pct_change_str'],
                                    'pct_valid': v['pct_change_valid']
                                } for k, v in dict(sorted_comparison[:50]).items()},
                                'most_improved': most_improved
                            }
                except Exception as e:
                    # If comparison fails, continue with regular context
                    import traceback
                    # Log error but don't break the flow
                    pass
        
        # Handle month-over-month comparison queries first (e.g., "month-over-month change between July, August, September")
        # BUT: Only if NOT a ranking query (top-N queries should be handled by SQL path)
        question_lower = question.lower()
        
        # Check if this is a ranking query (top-N) - if so, skip MoM detection
        is_ranking_query = any(word in question_lower for word in ['top ', 'best ', 'highest ', 'top-', 'top '])
        
        # More flexible MOM detection: check for month names AND comparison keywords
        month_keywords = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        has_months = any(month in question_lower for month in month_keywords)
        has_comparison_keywords = any(phrase in question_lower for phrase in [
            'month-over-month', 'month over month', 'mom', 'change between', 
            'compare', 'sequential', 'absolute and percentage change',
            'vs ', 'versus', 'compared to', 'compared with',
            'aug→jul', 'sep→aug', 'aug to jul', 'sep to aug'
        ])
        # Also detect if multiple months are mentioned (likely MOM query)
        month_count = sum(1 for month in month_keywords if month in question_lower)
        # MoM query requires: months present AND (comparison keywords OR multiple months) AND NOT a ranking query
        is_mom_query = has_months and (has_comparison_keywords or month_count >= 2) and not is_ranking_query
        
        # Handle month-over-month comparison queries FIRST (before monthly breakdown)
        if is_mom_query:
            try:
                import re
                from datetime import datetime
                
                cols_for_mom = insights.get('columns', {}) or cols
                date_col = cols_for_mom.get('date')
                amount_col = cols_for_mom.get('amount')
                
                if date_col and amount_col and df is not None and not df.empty:
                    # Extract months and year from question
                    month_names = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                    }
                    
                    # Extract year from question or use dataset max year
                    year_match = re.search(r'20\d{2}', question)
                    target_year = int(year_match.group(0)) if year_match else None
                    if not target_year:
                        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                        if not dates.empty:
                            target_year = dates.max().year
                    
                    # Extract months mentioned in question
                    months_found = []
                    for month_name, month_num in month_names.items():
                        if month_name in question_lower:
                            months_found.append((month_num, month_name))
                    
                    # Sort months by number
                    months_found.sort(key=lambda x: x[0])
                    
                    if months_found and target_year:
                        # Calculate totals for each month
                        df_copy = df.copy()
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                        
                        month_data = {}
                        for month_num, month_name in months_found:
                            month_df = df_copy[
                                (df_copy[date_col].dt.year == target_year) &
                                (df_copy[date_col].dt.month == month_num)
                            ]
                            total = float(month_df[amount_col].sum()) if not month_df.empty else 0.0
                            count = len(month_df)
                            # Use full month name for consistency (e.g., "July" not "Jul")
                            full_month_names = {
                                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                9: 'September', 10: 'October', 11: 'November', 12: 'December'
                            }
                            month_data[month_num] = {
                                'month_name': full_month_names.get(month_num, month_name.title()),
                                'month_num': month_num,
                                'total': total,
                                'count': count
                            }
                        
                        # Calculate month-over-month changes
                        # Compare each month TO the previous month (only adjacent months, no self-comparisons)
                        # Example: [Jul, Aug, Sep] → Aug→Jul, Sep→Aug (only 2 comparisons, not 3)
                        comparisons = []
                        for i in range(1, len(months_found)):  # Start from index 1 (second month)
                            curr_month_num = months_found[i][0]  # Current month (e.g., Aug)
                            prev_month_num = months_found[i - 1][0]  # Previous month (e.g., Jul)
                            
                            # CRITICAL: Skip self-comparisons (should never happen, but guard against it)
                            if curr_month_num == prev_month_num:
                                continue
                            
                            curr_data = month_data[curr_month_num]
                            prev_data = month_data[prev_month_num]
                            
                            curr_total = curr_data['total']  # Current month total (e.g., Aug)
                            prev_total = prev_data['total']  # Previous month total (e.g., Jul)
                            
                            # Absolute change: current - previous
                            abs_change = curr_total - prev_total
                            
                            # Percentage change using centralized comparison service
                            from .comparison_service import ComparisonService
                            pct_change_str, pct_valid = ComparisonService.calculate_percentage_change(
                                curr_total, prev_total, format_result=True
                            )
                            
                            comparisons.append({
                                'from_month': prev_data['month_name'],  # Previous month (e.g., Jul)
                                'from_month_num': prev_month_num,
                                'to_month': curr_data['month_name'],  # Current month (e.g., Aug)
                                'to_month_num': curr_month_num,
                                'from_total': prev_total,
                                'to_total': curr_total,
                                'abs_change': abs_change,
                                'pct_change': pct_change_str,  # None for N/A, otherwise formatted string
                                'pct_valid': pct_valid
                            })
                        
                        context['month_over_month'] = {
                            'year': target_year,
                            'months': {m['month_num']: m for m in month_data.values()},
                            'comparisons': comparisons,
                            'month_list': [m['month_name'] for m in sorted(month_data.values(), key=lambda x: x['month_num'])]
                        }
                        
                        # Store mom_data for structured return
                        mom_data = context['month_over_month']
                        
                        # CRITICAL: Return early to prevent monthly_breakdown from overwriting
                        # Don't continue to monthly breakdown processing
                        return context
            except Exception as e:
                # If month-over-month calculation fails, continue with regular context
                import traceback
                pass
        
        # Handle monthly breakdown queries (e.g., "monthly totals for 2025" or "last N months")
        # Only process if month_over_month was NOT set
        if 'month_over_month' not in context:
            is_monthly_query = any(word in question_lower for word in ['monthly', 'month', 'list months', 'show months'])
            is_last_n_months = plan.time_filter and plan.time_filter.mode == TimeMode.LAST_N_MONTHS and plan.time_filter.n_months
            is_year_specific = any(word in question_lower for word in ['2025', '2024', 'year', 'totals'])
            
            if is_monthly_query or is_last_n_months:
                # Check if question asks for monthly breakdown (year-specific OR last N months)
                if is_year_specific or is_last_n_months:
                    try:
                        cols_for_monthly = insights.get('columns', {}) or cols
                        date_col = cols_for_monthly.get('date')
                        amount_col = cols_for_monthly.get('amount')
                        
                        if date_col and amount_col and df is not None and not df.empty:
                            # Extract year from question or use dataset year
                            import re
                            year_match = re.search(r'20\d{2}', question)
                            target_year = int(year_match.group(0)) if year_match else None
                            
                            if not target_year:
                                # Default to dataset max year
                                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                                if not dates.empty:
                                    target_year = dates.max().year
                            
                            if target_year:
                                # Filter DataFrame for target year (CRITICAL: filter first)
                                df_copy = df.copy()
                                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                                
                                year_df = df_copy[
                                    (df_copy[date_col].dt.year == target_year) &
                                    (df_copy[date_col].notna())
                                ].copy()
                                
                                if not year_df.empty:
                                    # Group by month and sum (clean approach)
                                    # Create month period column and aggregate
                                    monthly_series = (
                                        year_df.assign(_YM_=year_df[date_col].dt.to_period('M'))
                                        .groupby('_YM_')[amount_col]
                                        .sum()
                                    )
                                    
                                    # Ensure all 12 months exist (zeros included)
                                    all_months = pd.period_range(
                                        start=f"{target_year}-01",
                                        end=f"{target_year}-12",
                                        freq='M'
                                    )
                                    
                                    # Reindex to include all months (fill zeros)
                                    monthly_series = monthly_series.reindex(all_months, fill_value=0)
                                    
                                    # Identify highest month
                                    best_month_period = monthly_series.idxmax()  # e.g., Period('2025-07', 'M')
                                    best_value = float(monthly_series.max())
                                    best_month_label = best_month_period.strftime('%B %Y')
                                    
                                    # Also get transaction counts per month
                                    monthly_counts = (
                                        year_df.assign(_YM_=year_df[date_col].dt.to_period('M'))
                                        .groupby('_YM_')[amount_col]
                                        .count()
                                        .reindex(all_months, fill_value=0)
                                    )
                                    
                                    # Convert to dictionary format
                                    monthly_data = {}
                                    for month_period in all_months:
                                        month_label = month_period.strftime('%B %Y')  # e.g., "January 2025"
                                        total = float(monthly_series.loc[month_period])
                                        count = int(monthly_counts.loc[month_period])
                                        
                                        monthly_data[month_label] = {
                                            'total': total,
                                            'count': count,
                                            'month_period': str(month_period)
                                        }
                                    
                                    context['monthly_breakdown'] = {
                                        'year': target_year,
                                        'monthly_data': monthly_data,
                                        'highest_month': best_month_label,
                                        'highest_total': best_value,
                                        'highest_month_period': str(best_month_period),
                                        'total_months': len(all_months),
                                        'months_with_sales': sum(1 for v in monthly_data.values() if v['total'] > 0),
                                        'months_with_zero': sum(1 for v in monthly_data.values() if v['total'] == 0)
                                    }
                                else:
                                    # Even if no data, create empty breakdown for all 12 months
                                    all_months = pd.period_range(
                                        start=f"{target_year}-01",
                                        end=f"{target_year}-12",
                                        freq='M'
                                    )
                                    
                                    monthly_data = {}
                                    for month_period in all_months:
                                        month_label = month_period.strftime('%B %Y')
                                        monthly_data[month_label] = {
                                            'total': 0.0,
                                            'count': 0,
                                            'month_period': str(month_period)
                                        }
                                    
                                    context['monthly_breakdown'] = {
                                        'year': target_year,
                                        'monthly_data': monthly_data,
                                        'highest_month': 'N/A',
                                        'highest_total': 0.0,
                                        'total_months': len(all_months),
                                        'months_with_sales': 0,
                                        'months_with_zero': 12
                                    }
                    except Exception as e:
                        # If monthly breakdown fails, continue with regular context
                        pass
                
                # Handle "last N months" queries (e.g., "last six calendar months")
                if is_last_n_months:
                    try:
                        n_months = plan.time_filter.n_months
                        date_col = cols.get('date')
                        amount_col = cols.get('amount')
                        
                        if date_col and amount_col and df is not None:
                            # Get dataset max date to calculate proper month range
                            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                            if not dates.empty:
                                max_date = dates.max().to_pydatetime()
                                
                                # Calculate month range from time filter bounds (not from filtered data)
                                # This ensures all N months are included even if some have zero sales
                                from dateutil.relativedelta import relativedelta
                                end_exclusive = max_date.replace(day=1) + relativedelta(months=1)  # Start of next month
                                start_date = end_exclusive - relativedelta(months=n_months)  # Start of first month
                                
                                # Convert to periods
                                start_month = pd.Period(start_date, freq='M')
                                end_month = pd.Period(end_exclusive - relativedelta(days=1), freq='M')  # Last day of last month
                                
                                # Create all months in range (including zeros)
                                all_months = pd.period_range(start=start_month, end=end_month, freq='M')
                                
                                # Use filtered_df (already filtered by time filter service) for aggregation
                                df_for_monthly = filtered_df.copy() if filtered_df is not None and not filtered_df.empty else pd.DataFrame()
                                if not df_for_monthly.empty:
                                    df_for_monthly[date_col] = pd.to_datetime(df_for_monthly[date_col], errors='coerce')
                                    
                                    # Calculate monthly breakdown from filtered data
                                    monthly_series = (
                                        df_for_monthly.assign(_YM_=df_for_monthly[date_col].dt.to_period('M'))
                                        .groupby('_YM_')[amount_col]
                                        .sum()
                                    )
                                    
                                    # Also get transaction counts
                                    monthly_counts = (
                                        df_for_monthly.assign(_YM_=df_for_monthly[date_col].dt.to_period('M'))
                                        .groupby('_YM_')[amount_col]
                                        .count()
                                    )
                                else:
                                    # No data in period - create empty series
                                    monthly_series = pd.Series(dtype=float)
                                    monthly_counts = pd.Series(dtype=int)
                                
                                # Reindex to include all months (fill zeros)
                                monthly_series = monthly_series.reindex(all_months, fill_value=0)
                                monthly_counts = monthly_counts.reindex(all_months, fill_value=0)
                                
                                # Identify highest month (skip if all zeros)
                                if monthly_series.sum() > 0:
                                    best_month_period = monthly_series.idxmax()
                                    best_value = float(monthly_series.max())
                                    best_month_label = best_month_period.strftime('%B %Y')
                                else:
                                    best_month_period = None
                                    best_value = 0.0
                                    best_month_label = 'N/A'
                                
                                # Convert to dictionary format
                                monthly_data = {}
                                for month_period in all_months:
                                    month_label = month_period.strftime('%B %Y')
                                    total = float(monthly_series.loc[month_period])
                                    count = int(monthly_counts.loc[month_period])
                                    
                                    monthly_data[month_label] = {
                                        'total': total,
                                        'count': count,
                                        'month_period': str(month_period)
                                    }
                                
                                # Calculate period label (e.g., "Apr–Sep 2025")
                                start_month_name = start_month.strftime('%b')
                                end_month_name = end_month.strftime('%b %Y')
                                period_label = f"{start_month_name}–{end_month_name}"
                                
                                context['monthly_breakdown'] = {
                                    'period_label': period_label,
                                    'n_months': n_months,
                                    'monthly_data': monthly_data,
                                    'highest_month': best_month_label,
                                    'highest_total': best_value,
                                    'highest_month_period': str(best_month_period) if best_month_period else '',
                                    'total_months': len(all_months),
                                    'months_with_sales': sum(1 for v in monthly_data.values() if v['total'] > 0),
                                    'months_with_zero': sum(1 for v in monthly_data.values() if v['total'] == 0),
                                    'start_month': str(start_month),
                                    'end_month': str(end_month)
                                }
                    except Exception as e:
                        # If monthly breakdown fails, continue with regular context
                        import traceback
                        pass
        
        return context
    
    @staticmethod
    def _compact_for_json(data: dict) -> dict:
        """
        Compact data for efficient JSON serialization.
        
        - Rounds numbers to sensible decimals
        - Limits large collections
        - Removes unnecessary fields
        """
        import math
        
        compact = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # Round numeric values in dicts
                compact[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        # Nested dict (e.g., salesmen data)
                        compact[k][sub_k] = {
                            k2: round(v2, 2) if isinstance(v2, (int, float)) and not math.isnan(v2) else v2
                            for k2, v2 in sub_v.items()
                        }
                    else:
                        compact[k][sub_k] = round(sub_v, 2) if isinstance(sub_v, (int, float)) and not math.isnan(sub_v) else sub_v
            elif isinstance(v, list):
                # Limit list size
                compact[k] = v[:50] if len(v) > 50 else v
            elif isinstance(v, (int, float)):
                # Round numeric values
                compact[k] = round(v, 2) if not math.isnan(v) else 0
            else:
                compact[k] = v
        
        return compact
    
    def _compact_summary(self, data: dict, question: str = "") -> dict:
        """
        Create compact summary for Gemini API.
        
        Intelligently includes data based on question context:
        - For "worst/bottom/lowest" questions: includes bottom performers
        - For "top/best/highest" questions: includes top performers
        - For general questions: includes both top and bottom
        """
        s = {}
        for k in [
            'total_records', 'columns', 'column_mapping', 'date_range',
            'total_revenue', 'avg_order_value', 'top_salesmen', 'all_salesmen',
            'top_customers', 'all_customers', 'top_products', 'all_products',
            'churn_risk_customers', 'sample_data', 'quarter_comparison'
        ]:
            if k in data:
                s[k] = data[k]
        
        # Analyze question to determine what data is needed
        ql = (question or "").lower()
        needs_bottom_data = any(w in ql for w in [
            'worst', 'bottom', 'lowest', 'poor', 'bad', 'underperform',
            'weak', 'declin', 'churn', 'at-risk', 'losing', 'drop'
        ])
        needs_top_data = any(w in ql for w in [
            'top', 'best', 'highest', 'great', 'excellent', 'strong'
        ])
        
        # Include full data if question needs bottom performers
        # Otherwise use smart truncation
        max_items = ConfigService.MAX_TOP_ITEMS
        
        # Truncate large collections
        if 'columns' in s and isinstance(s['columns'], list):
            s['columns'] = s['columns'][:ConfigService.MAX_COLUMNS_SUMMARY]
        
        # For salesmen: use all_salesmen if available, otherwise fallback to top_salesmen
        salesmen_dict = None
        if 'all_salesmen' in s and isinstance(s['all_salesmen'], dict):
            salesmen_dict = s['all_salesmen']
            # Remove all_salesmen from summary to avoid duplication
            del s['all_salesmen']
        elif 'top_salesmen' in s and isinstance(s['top_salesmen'], dict):
            salesmen_dict = s['top_salesmen']
        
        if salesmen_dict:
            sorted_items = sorted(salesmen_dict.items(), key=lambda x: x[1], reverse=True)
            
            if needs_bottom_data:
                # Include ALL salesmen for worst/bottom questions
                s['top_salesmen'] = dict(sorted_items)  # Full data
                # Also add bottom salesmen explicitly for clarity
                if len(sorted_items) > max_items:
                    s['bottom_salesmen'] = dict(sorted_items[-max_items:])
            elif needs_top_data:
                # Only top performers needed
                s['top_salesmen'] = dict(sorted_items[:max_items])
            else:
                # Include both top and bottom for general questions
                s['top_salesmen'] = dict(sorted_items[:max_items])
                if len(sorted_items) > max_items:
                    s['bottom_salesmen'] = dict(sorted_items[-max_items:])
        
        # For customers: similar logic
        if 'top_customers' in s and isinstance(s['top_customers'], list):
            customers_list = s['top_customers']
            if needs_bottom_data:
                # Include all customers for worst/bottom questions
                s['top_customers'] = customers_list  # Full data
                # Also add bottom customers
                if len(customers_list) > max_items:
                    sorted_customers = sorted(
                        customers_list, 
                        key=lambda x: x.get('revenue', 0) if isinstance(x, dict) else 0
                    )
                    s['bottom_customers'] = sorted_customers[:max_items]
            elif needs_top_data:
                s['top_customers'] = customers_list[:max_items]
            else:
                # Include both for general questions
                s['top_customers'] = customers_list[:max_items]
                if len(customers_list) > max_items:
                    sorted_customers = sorted(
                        customers_list,
                        key=lambda x: x.get('revenue', 0) if isinstance(x, dict) else 0
                    )
                    s['bottom_customers'] = sorted_customers[:max_items]
        
        # For products: similar logic
        if 'top_products' in s and isinstance(s['top_products'], list):
            products_list = s['top_products']
            if needs_bottom_data:
                s['top_products'] = products_list  # Full data
                if len(products_list) > max_items:
                    sorted_products = sorted(
                        products_list,
                        key=lambda x: x.get('revenue', 0) if isinstance(x, dict) else 0
                    )
                    s['bottom_products'] = sorted_products[:max_items]
            elif needs_top_data:
                s['top_products'] = products_list[:max_items]
            else:
                s['top_products'] = products_list[:max_items]
                if len(products_list) > max_items:
                    sorted_products = sorted(
                        products_list,
                        key=lambda x: x.get('revenue', 0) if isinstance(x, dict) else 0
                    )
                    s['bottom_products'] = sorted_products[:max_items]
        
        if 'sample_data' in s and isinstance(s['sample_data'], list):
            s['sample_data'] = s['sample_data'][:ConfigService.MAX_SAMPLE_DATA_ROWS]
        
        return s
    
    def _query_gemini(self, question: str, data_summary: dict, df=None, insights=None) -> str:
        """
        Query Gemini API with retry logic using RAG approach.
        
        Uses question-aware data preparation to send only relevant context.
        """
        # Use RAG approach: prepare relevant context based on question
        if df is not None and insights is not None:
            context = self._prepare_relevant_context(question, df, insights, data_summary)
        else:
            # Fallback to compact summary if DataFrame not available
            context = self._compact_summary(data_summary or {}, question)
        # Compact data for token efficiency
        # Round numbers before serialization
        context = self._compact_for_json(context)
        
        try:
            json_data = json.dumps(
                context, default=str, separators=(',', ':')
            )
        except Exception:
            json_data = json.dumps(
                list((context or {}).keys()), default=str, separators=(',', ':')
            )
        
        model_names = ConfigService.GEMINI_MODELS
        last_err = None
        
        # Build context from computed metrics (never from LLM)
        ctx = self._build_revenue_context(context, question)
        
        # Add monthly breakdown placeholders if available (year-scoped or period-scoped)
        if 'monthly_breakdown' in context:
            mb = context['monthly_breakdown']
            monthly_data = mb.get('monthly_data', {})
            
            # Add all monthly totals as placeholders
            for month_label, data in monthly_data.items():
                month_key = month_label.replace(' ', '_').lower()  # "May 2025" -> "may_2025"
                ctx[f'monthly_{month_key}_total'] = f"{data.get('total', 0):,.2f}"
                ctx[f'monthly_{month_key}_count'] = str(data.get('count', 0))
            
            # Add summary placeholders
            ctx['highest_month'] = mb.get('highest_month', 'N/A')
            ctx['highest_month_total'] = f"{mb.get('highest_total', 0):,.2f}"
            ctx['monthly_year'] = str(mb.get('year', ''))
            ctx['monthly_period_label'] = mb.get('period_label', '')
            ctx['monthly_n_months'] = str(mb.get('n_months', ''))
            ctx['months_with_zero'] = str(mb.get('months_with_zero', 0))
            
            # Calculate period total from monthly data
            period_total = sum(data.get('total', 0) for data in monthly_data.values())
            period_transactions = sum(data.get('count', 0) for data in monthly_data.values())
            ctx['year_total'] = f"{period_total:,.2f}"  # Keep key name for compatibility
            ctx['year_transactions'] = str(period_transactions)
            ctx['period_total'] = f"{period_total:,.2f}"
            ctx['period_transactions'] = str(period_transactions)
        
        # Add quarter comparison placeholders if available
        if 'quarter_comparison' in context:
            qc = context['quarter_comparison']
            ctx['quarter_comparison'] = json.dumps(qc, default=str, indent=2)
            
            if qc.get('most_improved'):
                mi = qc['most_improved']
                ctx['most_improved_salesman'] = mi.get('salesman', 'N/A')
                ctx['most_improved_prev'] = f"{mi.get('previous_quarter', 0):,.2f}"
                ctx['most_improved_current'] = f"{mi.get('last_quarter', 0):,.2f}"
                ctx['most_improved_change'] = f"{mi.get('change', 0):,.2f}"
                ctx['most_improved_pct'] = mi.get('pct_change_str', 'N/A')
            
            if qc.get('last_quarter'):
                ctx['last_quarter_label'] = qc['last_quarter'].get('label', '')
                ctx['last_quarter_range'] = f"{qc['last_quarter'].get('start', '')} → {qc['last_quarter'].get('end', '')}"
            
            if qc.get('previous_quarter'):
                ctx['previous_quarter_label'] = qc['previous_quarter'].get('label', '')
                ctx['previous_quarter_range'] = f"{qc['previous_quarter'].get('start', '')} → {qc['previous_quarter'].get('end', '')}"
        
        # CRITICAL: If month-over-month comparison exists, render it directly
        if 'month_over_month' in context:
            rendered_text = self._render_deterministic_fallback(context, ctx)
            # Return structured data for CSV export
            return {
                'text': rendered_text,
                'mom_data': context['month_over_month']
            }
        
        # CRITICAL: If monthly breakdown exists, render it directly (year-scoped, no all-time KPIs)
        if 'monthly_breakdown' in context:
            return self._render_deterministic_fallback(context, ctx)
        
        # CRITICAL: If quarter comparison exists, render it directly
        if 'quarter_comparison' in context:
            return self._render_deterministic_fallback(context, ctx)
        
        for mname in model_names:
            try:
                model = genai.GenerativeModel(mname)
                for attempt in range(ConfigService.GEMINI_RETRY_ATTEMPTS):
                    try:
                        # Get template from LLM (with placeholders)
                        prompt = get_bi_analyst_prompt(question, json_data)
                        resp = model.generate_content(prompt)
                        template = resp.text
                        
                        # Inject numbers from context (computed locally)
                        filled = render_placeholders(template, ctx)
                        
                        # Verify no placeholders remain
                        if has_unfilled_placeholders(filled):
                            # Fallback to deterministic markdown
                            filled = self._render_deterministic_fallback(context, ctx)
                        
                        return filled
                    except Exception as e:
                        last_err = str(e)
                        if any(x in last_err for x in [
                            'incomplete envelope', 'reset by peer',
                            'invalid_argument', 'connection reset'
                        ]):
                            delay = ConfigService.GEMINI_RETRY_DELAY_BASE * (2 ** attempt)
                            time.sleep(delay)
                            continue
                        break
            except Exception:
                continue
        
        # Fallback to local if Gemini fails
        return (
            "## Error\n"
            f"Gemini API unavailable: {last_err or 'Unknown error'}\n\n"
            "Please check your API key or try again later."
        )
    
    def _query_local_fallback(
        self,
        question: str,
        df=None,
        insights=None
    ) -> str:
        """Local fallback when AI is unavailable."""
        # Import here to avoid circular dependency
        from .insight_service import InsightService
        
        insight_service = InsightService()
        table = insight_service.create_data_table(df, question, insights)
        
        if table is not None and not table.empty:
            top_row = table.iloc[0].to_dict()
            return (
                "## Executive Summary\n"
                "- Answered locally from the uploaded sheet (no external AI).\n\n"
                "## Analysis\n"
                f"- Top row: `{top_row}`\n\n"
                "## Recommended Actions\n"
                "1. Investigate why winners win (mix, price, route).\n"
                "2. Coach bottom performers using items and customers from the top group."
            )
        
        return (
            "I couldn't find the needed columns yet. "
            "Please check that amount/value and date/salesman exist."
        )
    
    def _render_layout(self, layout_json: str, context: dict) -> str:
        """
        Render layout JSON by replacing placeholders with actual numbers from context.
        
        Guards against placeholder leakage and ensures all numbers are injected locally.
        
        Args:
            layout_json: JSON string with placeholders like {{top_name}}, {{top_value}}
            context: Context dictionary with computed numbers
            
        Returns:
            Rendered markdown string with no remaining placeholders
        """
        import json
        import re
        
        try:
            # Parse layout JSON
            if layout_json.strip().startswith('```'):
                # Remove markdown code blocks
                layout_json = re.sub(r'```json?\s*', '', layout_json).strip()
                layout_json = layout_json.rstrip('```').strip()
            
            layout = json.loads(layout_json)
            
            # Extract values from context (all computed locally)
            currency_symbol = context.get('currency', {}).get('symbol', '$')
            rep_count = context.get('all_salesmen_count', 0)
            
            # Get period label for display
            period_label = self._get_period_label(context)
            
            # Get top/bottom performers (all numbers pre-computed)
            top_salesmen = context.get('top_salesmen', {})
            bottom_salesmen = context.get('bottom_salesmen', {})
            
            top_name = "N/A"
            top_value = "0.00"
            bot_name = "N/A"
            bot_value = "0.00"
            
            if top_salesmen:
                top_item = next(iter(top_salesmen.items()))
                top_name = top_item[0]
                # Format number locally - never let LLM do math
                top_value = f"{top_item[1].get('total', 0):,.2f}"
            
            if bottom_salesmen:
                bot_item = next(iter(bottom_salesmen.items()))
                bot_name = bot_item[0]
                # Format number locally - never let LLM do math
                bot_value = f"{bot_item[1].get('total', 0):,.2f}"
            
            # Build replacements dictionary (all numbers computed locally)
            replacements = {
                '{{title}}': f"Sales Performance Analysis {period_label}",
                '{{top_name}}': top_name,
                '{{top_value}}': top_value,
                '{{bot_name}}': bot_name,
                '{{bot_value}}': bot_value,
                '{{currency}}': currency_symbol,
                '{{rep_count}}': str(rep_count),
                '{{time_period}}': period_label,
            }
            
            # Render sections
            output = []
            for section in layout.get('sections', []):
                section_type = section.get('type')
                if section_type == 'headline':
                    text = section.get('text', '{{title}}')
                    for placeholder, value in replacements.items():
                        text = text.replace(placeholder, value)
                    output.append(f"## {text}\n")
                elif section_type == 'bullets':
                    items = section.get('items', [])
                    for item in items:
                        for placeholder, value in replacements.items():
                            item = item.replace(placeholder, value)
                        output.append(f"- {item}")
                elif section_type == 'insights':
                    items = section.get('items', [])
                    output.append("\n## Key Insights")
                    for item in items:
                        for placeholder, value in replacements.items():
                            item = item.replace(placeholder, value)
                        output.append(f"- {item}")
                elif section_type == 'actions':
                    items = section.get('items', [])
                    output.append("\n## Recommended Actions")
                    for item in items:
                        for placeholder, value in replacements.items():
                            item = item.replace(placeholder, value)
                        output.append(f"- {item}")
            
            result = '\n'.join(output)
            
            # Placeholder gate: assert no unresolved placeholders
            try:
                from .data_formatting_service import assert_no_placeholders
                assert_no_placeholders(result)
            except ValueError:
                # Placeholders leaked - fall back to deterministic markdown
                if ctx is None:
                    ctx = self._build_revenue_context(context, "")
                return self._render_deterministic_fallback(context, ctx)
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to deterministic markdown if layout parsing fails
            ctx = self._build_revenue_context(context, "")
            return self._render_deterministic_fallback(context, ctx)
    
    def _get_period_label(self, context: dict) -> str:
        """
        Generate explicit period label from context.
        
        Examples:
        - "Last 3 months (2025-07-01 → 2025-09-30)"
        - "Q3 2024 (2024-07-01 → 2024-09-30)"
        - "Last completed quarter (Q2-2025: 2025-04-01 → 2025-06-30)"
        - "Quarter-over-Quarter: Q2-2025 vs Q1-2025" (when quarter_comparison exists)
        - "Monthly Breakdown for 2025" (when monthly_breakdown exists)
        - "2024-04-09 → 2025-09-28" (all time with explicit dates)
        """
        # If monthly breakdown exists, use that format (year-scoped or period-scoped)
        if 'monthly_breakdown' in context:
            mb = context['monthly_breakdown']
            period_label = mb.get('period_label', '')
            if period_label:
                # "last N months" format - check if bounds are in period_label
                if '→' in period_label or '-' in period_label:
                    return f"Last {mb.get('n_months', 'N')} months ({period_label})"
                else:
                    return f"Last {mb.get('n_months', 'N')} months ({period_label})"
            else:
                # Year-specific format (e.g., "Monthly Breakdown for 2025")
                year = mb.get('year', '')
                return f"Monthly Breakdown for {year}"
        
        # Check if time_filter_applied_data has period_label with bounds
        time_filter_data = context.get('time_filter_applied_data', {})
        if time_filter_data and time_filter_data.get('period_label'):
            return time_filter_data['period_label']
        
        # If quarter comparison exists, use that format
        if 'quarter_comparison' in context:
            qc = context['quarter_comparison']
            last_q = qc.get('last_quarter', {})
            prev_q = qc.get('previous_quarter', {})
            
            last_label = last_q.get('label', '')
            last_range = f"({last_q.get('start', '')} → {last_q.get('end', '')})" if last_q.get('start') else ''
            prev_label = prev_q.get('label', '')
            prev_range = f"({prev_q.get('start', '')} → {prev_q.get('end', '')})" if prev_q.get('start') else ''
            
            if last_label and prev_label:
                return f"Quarter-over-Quarter: {last_label} {last_range} vs {prev_label} {prev_range}"
            elif last_label or prev_label:
                return f"Quarter-over-Quarter: {last_label or prev_label}"
            else:
                return "Quarter-over-Quarter Comparison"
        
        time_filter = context.get('time_filter_applied', {})
        date_range = context.get('date_range', {})
        
        start = date_range.get('start', '')
        end = date_range.get('end', '')
        
        if time_filter:
            mode = time_filter.get('mode', '')
            if 'last_completed_quarter' in mode:
                # Determine quarter from dates
                if start and end:
                    try:
                        from datetime import datetime
                        start_dt = datetime.strptime(start, '%Y-%m-%d')
                        q = (start_dt.month - 1) // 3 + 1
                        return f"Last completed quarter (Q{q}-{start_dt.year}: {start} → {end})"
                    except:
                        return f"Last Quarter ({start} → {end})"
                return f"Last Quarter ({start} → {end})"
            elif 'this_quarter' in mode:
                if start and end:
                    try:
                        from datetime import datetime
                        start_dt = datetime.strptime(start, '%Y-%m-%d')
                        q = (start_dt.month - 1) // 3 + 1
                        return f"Q{q}-{start_dt.year} ({start} → {end})"
                    except:
                        return f"This Quarter ({start} → {end})"
                return f"This Quarter ({start} → {end})"
            elif 'last_month' in mode:
                return f"Last Month ({start} → {end})"
            elif 'this_month' in mode:
                return f"This Month ({start} → {end})"
            elif 'last_n_months' in mode:
                n = time_filter.get('n_months', 3)
                return f"Last {n} months ({start} → {end})"
            elif 'specific_quarter' in mode:
                return f"Q{time_filter.get('quarter', '?')}-{time_filter.get('year', '?')} ({start} → {end})"
            else:
                return f"{mode} ({start} → {end})"
        
        # Fallback: show explicit date range for all time
        if start and end:
            return f"{start} → {end}"
        
        return "(All time)"
    
    def _build_revenue_context(self, context: dict, question: str) -> dict[str, str]:
        """
        Build context dictionary from computed metrics (never from LLM).
        
        All numbers are computed locally and formatted here.
        
        Args:
            context: Context dictionary with computed aggregates
            question: User's question (for context)
            
        Returns:
            Dictionary of placeholder keys to formatted values
        """
        # Get period label
        period_label = self._get_period_label(context)
        
        # Get currency
        currency_symbol = context.get('currency', {}).get('symbol', '$')
        currency_code = context.get('currency', {}).get('code', 'USD')
        
        # Get totals (computed locally)
        total_revenue = context.get('total_revenue', 0)
        avg_order_value = context.get('avg_order_value', 0)
        
        # Format numbers locally
        total_revenue_value = f"{total_revenue:,.2f}"
        avg_order_value_value = f"{avg_order_value:,.2f}"
        
        # Get top/bottom performers (computed locally)
        top_salesmen = context.get('top_salesmen', {})
        bottom_salesmen = context.get('bottom_salesmen', {})
        
        top_salesman_name = "N/A"
        top_salesman_revenue = "0.00"
        bottom_salesman_name = "N/A"
        bottom_salesman_revenue = "0.00"
        
        if top_salesmen:
            top_item = next(iter(top_salesmen.items()))
            top_salesman_name = top_item[0]
            top_salesman_revenue = f"{top_item[1].get('total', 0):,.2f}"
        
        if bottom_salesmen:
            bot_item = next(iter(bottom_salesmen.items()))
            bottom_salesman_name = bot_item[0]
            bottom_salesman_revenue = f"{bot_item[1].get('total', 0):,.2f}"
        
        # Get rep count
        total_salesmen_count = str(context.get('all_salesmen_count', 0))
        
        # Get date range
        date_range = context.get('date_range', {})
        time_period = period_label
        if date_range.get('start') and date_range.get('end'):
            time_period = f"{date_range.get('start')} → {date_range.get('end')}"
        
        return {
            "total_revenue_value": total_revenue_value,
            "currency_symbol": currency_symbol,
            "currency_code": currency_code,
            "time_period": time_period,
            "top_salesman_name": top_salesman_name,
            "top_salesman_revenue": top_salesman_revenue,
            "bottom_salesman_name": bottom_salesman_name,
            "bottom_salesman_revenue": bottom_salesman_revenue,
            "total_salesmen_count": total_salesmen_count,
            "avg_order_value_value": avg_order_value_value,
            "period_label": period_label,
        }
    
    def _render_deterministic_fallback(self, context: dict, ctx: dict[str, str] = None) -> str:
        """
        Render deterministic markdown fallback when layout parsing fails or placeholders leak.
        
        All numbers are computed locally - never from LLM.
        
        Args:
            context: Context dictionary with computed aggregates
            ctx: Optional pre-built revenue context (if already computed)
        """
        if ctx is None:
            ctx = self._build_revenue_context(context, "")
        
        period_label = ctx.get('period_label', '(All time)')
        
        # If month-over-month comparison exists, render that format
        if 'month_over_month' in context:
            mom = context['month_over_month']
            year = mom.get('year', '')
            months = mom.get('months', {})
            comparisons = mom.get('comparisons', [])
            
            # Get currency formatting service
            from .currency_formatting_service import CurrencyFormattingService
            currency_settings = CurrencyFormattingService.get_currency_settings()
            currency_code = currency_settings.get('code', 'USD')
            currency_info = CurrencyFormattingService.get_currency_info(currency_code)
            currency_symbol = currency_info.get('symbol', '$')
            
            # Build period label (e.g., "Jul–Sep 2025")
            sorted_months = sorted(months.keys())
            if sorted_months:
                first_month_name = months[sorted_months[0]]['month_name']
                last_month_name = months[sorted_months[-1]]['month_name']
                # Use abbreviated month names for period label
                month_abbrevs = {
                    'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
                    'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
                    'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
                }
                first_abbrev = month_abbrevs.get(first_month_name, first_month_name[:3])
                last_abbrev = month_abbrevs.get(last_month_name, last_month_name[:3])
                period_label = f"{first_abbrev}–{last_abbrev} {year}"
            else:
                period_label = f"{year}"
            
            output = [f"## Month-over-Month Revenue Change\n"]
            output.append(f"**MoM:** {period_label}\n")
            
            # Show totals for each month with currency formatting
            output.append("\n**Totals:**\n")
            for month_num in sorted(months.keys()):
                month_info = months[month_num]
                month_name = month_info['month_name']
                total = month_info['total']
                count = month_info['count']
                # Format with currency
                formatted_total = CurrencyFormattingService.format_currency(total)
                output.append(f"- {month_name} {year}: {formatted_total} ({count} txns)")
            
            # Show month-over-month changes with currency formatting
            has_zero_prev = False  # Track if we need footnote
            if comparisons:
                output.append("\n**MoM Changes:**\n")
                for comp in comparisons:
                    from_month = comp['from_month']  # Previous month (e.g., Jul)
                    to_month = comp['to_month']  # Current month (e.g., Aug)
                    from_total = comp['from_total']
                    to_total = comp['to_total']
                    abs_change = comp['abs_change']
                    pct_change = comp['pct_change']  # None for N/A, otherwise formatted string
                    
                    # Format absolute change with currency
                    formatted_abs_change = CurrencyFormattingService.format_currency(abs(abs_change))
                    change_sign = "+" if abs_change >= 0 else "−"
                    change_str = f"{change_sign}{formatted_abs_change}"
                    
                    # Format percentage change
                    if pct_change is None:
                        pct_display = "N/A¹"
                        has_zero_prev = True
                    else:
                        pct_display = pct_change
                    
                    # Format: "Aug → Jul: Δ = −166,980.00 EGP; %Δ = −100.00%"
                    # Only show comparisons where to_month != from_month (no self-comparisons)
                    if to_month != from_month:
                        output.append(f"- {to_month} → {from_month}: Δ = {change_str}; %Δ = {pct_display}")
            
            # Add footnote if needed
            if has_zero_prev:
                output.append("\n¹ Previous month has zero revenue")
            
            # Return immediately - don't fall through to all-time KPIs
            return '\n'.join(output)
        
        # If monthly breakdown exists, render that format (year-scoped or period-scoped, no all-time KPIs)
        if 'monthly_breakdown' in context:
            mb = context['monthly_breakdown']
            period_label = mb.get('period_label', '')
            year = mb.get('year', '')
            n_months = mb.get('n_months', '')
            monthly_data = mb.get('monthly_data', {})
            highest_month = mb.get('highest_month', 'N/A')
            highest_total = mb.get('highest_total', 0)
            months_with_zero = mb.get('months_with_zero', 0)
            
            # Calculate period total from monthly data (not all-time)
            period_total = sum(data.get('total', 0) for data in monthly_data.values())
            period_transactions = sum(data.get('count', 0) for data in monthly_data.values())
            
            currency_symbol = ctx.get('currency_symbol', '$')
            
            # Determine header based on query type
            if period_label:
                # "last N months" format
                output = [f"## Monthly Sales Breakdown: {period_label}\n"]
                period_label_for_total = f"Last {n_months} months"
            else:
                # Year-specific format
                output = [f"## Monthly Sales Breakdown for {year}\n"]
                period_label_for_total = str(year)
            
            # Sort months chronologically
            sorted_months = sorted(monthly_data.items(), key=lambda x: x[1].get('month_period', ''))
            
            output.append("**Monthly Totals:**\n")
            for month_label, data in sorted_months:
                total = data.get('total', 0)
                count = data.get('count', 0)
                zero_indicator = " (zero sales)" if total == 0 else ""
                output.append(f"- {month_label}: {total:,.2f} {currency_symbol} ({count} transactions){zero_indicator}")
            
            output.append(f"\n**Highest Month:** {highest_month} — {highest_total:,.2f} {currency_symbol}")
            
            output.append(f"\n**{period_label_for_total} Total:** {period_total:,.2f} {currency_symbol} ({period_transactions} transactions)")
            
            if months_with_zero > 0:
                output.append(f"\n**Months with Zero Sales:** {months_with_zero} month(s)")
            
            # Return immediately - don't fall through to all-time KPIs
            return '\n'.join(output)
        
        # If quarter comparison exists, render that format
        if 'quarter_comparison' in context:
            qc = context['quarter_comparison']
            most_improved = qc.get('most_improved')
            last_q = qc.get('last_quarter', {})
            prev_q = qc.get('previous_quarter', {})
            comparison = qc.get('comparison', {})
            
            output = [f"## Quarter-over-Quarter: {last_q.get('label', '')} vs {prev_q.get('label', '')}\n"]
            
            if most_improved:
                mi_name = most_improved.get('salesman', 'N/A')
                mi_prev = most_improved.get('previous_quarter', 0)
                mi_curr = most_improved.get('last_quarter', 0)
                mi_change = most_improved.get('change', 0)
                mi_pct = most_improved.get('pct_change_str', 'N/A')
                
                output.append(f"**Most Improved:** {mi_name} — +{mi_change:,.2f} ({mi_prev:,.2f} → {mi_curr:,.2f}); % change: {mi_pct}\n")
            
            if comparison:
                output.append("\n**All Salesmen:**\n")
                for salesman, data in list(comparison.items())[:10]:  # Top 10
                    prev = data.get('previous', 0)
                    curr = data.get('current', 0)
                    change = data.get('change', 0)
                    pct = data.get('pct_change', 'N/A')
                    
                    change_str = f"+{change:,.2f}" if change >= 0 else f"{change:,.2f}"
                    output.append(f"- {salesman}: {prev:,.2f} → {curr:,.2f} (Δ: {change_str}, %: {pct})")
            
            return '\n'.join(output)
        
        # Regular format for non-comparative queries
        return f"""## Sales Performance Analysis {period_label}

**Total Revenue Overview:** {ctx['total_revenue_value']} {ctx['currency_symbol']} ({ctx['time_period']})

**Top performer:** {ctx['top_salesman_name']} — {ctx['top_salesman_revenue']} {ctx['currency_symbol']}

**Bottom performer:** {ctx['bottom_salesman_name']} — {ctx['bottom_salesman_revenue']} {ctx['currency_symbol']}

**Total reps:** {ctx['total_salesmen_count']}

## Key Insights

- Overall revenue: {ctx['total_revenue_value']} {ctx['currency_symbol']} ({ctx['time_period']})
- Average order value: {ctx['avg_order_value_value']} {ctx['currency_symbol']}
- Top performer {ctx['top_salesman_name']} generated {ctx['top_salesman_revenue']} {ctx['currency_symbol']}

## Recommended Actions

- Review top performer strategies for replication
- Provide coaching to bottom performers using top performer insights

*Note: All numbers computed from uploaded data. Period: {ctx['time_period']}*
"""
