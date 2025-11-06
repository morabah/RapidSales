"""
SQL Generation Service
Converts natural language questions to SQL using LLM with DuckDB execution.
"""

import json
import re
import difflib
import duckdb
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import google.generativeai as genai
import streamlit as st
import time
import hashlib

from .config_service import ConfigService
from .error_handling_service import ErrorHandlingService, ErrorCategory
from .time_resolver import TimeResolver
from .dataset_cache_service import DatasetCacheService
from .entity_normalization_service import EntityNormalizationService
from .query_plan_service import TimeFilter, TimeMode


class SQLGenerationService:
    """Service for generating and executing SQL from natural language queries."""
    
    def __init__(self):
        self.use_llm = self._check_gemini_availability()
        if self.use_llm:
            api_key = ConfigService.get_gemini_api_key()
            if api_key:
                genai.configure(api_key=api_key)
        
        # SQL safety blacklist
        self.SQL_BLACKLIST = [
            "CREATE ", "INSERT ", "UPDATE ", "DELETE ", "DROP ",
            "COPY ", "ATTACH ", "LOAD ", "HTTP", "EXECUTE ", "EXEC ",
            "ALTER ", "TRUNCATE ", "GRANT ", "REVOKE ", "CALL "
        ]
        
        # Maximum result rows
        self.MAX_RESULT_ROWS = 10000
        self.MAX_EXECUTION_TIME = 30  # seconds
        
        # Kill-switch thresholds
        self.LLM_LATENCY_THRESHOLD = 5.0  # seconds
        self.LLM_FAILURE_COUNT_THRESHOLD = 3
        self.llm_failure_count = 0
        self.llm_auto_fallback_enabled = False
        
        # Quota/rate limit tracking
        self._quota_tracking = {
            'daily_requests': 0,
            'last_reset_date': None,
            'quota_errors': 0,
            'model_usage': {}  # Track usage per model
        }
        self._current_model_index = 0  # Track which model we're using
        
        # Memory caps
        self.MAX_RESULT_BYTES = 50 * 1024 * 1024  # 50 MB
        self.MAX_DISPLAY_ROWS = 1000  # Cap before rendering
        
        # Cache for dataset hash
        self._dataset_cache = {}  # {dataset_hash: {normalized_view: bool, monthly_cube: Any, sql_plans: dict}}
    
    # Minimal heuristic fallback (dead-man's switch)
    @staticmethod
    def _heuristic_detect_single_salesman_month(question: str) -> bool:
        """
        Minimal heuristic fallback for single-salesman-month queries.
        Case-insensitive, handles diacritics, supports Arabic with digit normalization.
        
        Args:
            question: User's question
            
        Returns:
            True if heuristic matches
        """
        # Normalize Arabic digits and punctuation first
        from .data_formatting_service import normalize_digits
        q_normalized = normalize_digits(question)
        
        q_lower = q_normalized.lower()
        # Normalize: remove diacritics, spaces, punctuation
        q_clean = re.sub(r'[^\w\s]', '', q_lower)
        
        # English tokens
        has_exactly_one_en = (any(word in q_clean for word in ['only', 'exactly']) and 
                             'one' in q_clean)
        has_salesman_en = any(word in q_clean for word in ['salesman', 'rep', 'salesperson', 'salesmen'])
        has_month_en = any(word in q_clean for word in ['month', 'monthly'])
        
        # Arabic tokens (minimal set) - check in original question (Arabic preserved)
        has_exactly_one_ar = any(phrase in question for phrase in ['واحد فقط', 'فقط مندوب واحد', 'مندوب واحد'])
        has_salesman_ar = any(word in question for word in ['مندوب', 'بائع'])
        has_month_ar = any(word in question for word in ['شهر', 'شهري'])
        
        # Match if (English pattern OR Arabic pattern) is complete
        en_match = has_exactly_one_en and has_salesman_en and has_month_en
        ar_match = has_exactly_one_ar and has_salesman_ar and has_month_ar
        
        return en_match or ar_match
    
    @staticmethod
    def _tokenize_text(text: str | None) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        if not text:
            return []
        return re.findall(r"[a-z0-9]+", text.lower())
    
    @staticmethod
    def _fuzzy_ratio(a: str, b: str) -> float:
        """Return fuzzy matching ratio between two tokens."""
        return difflib.SequenceMatcher(None, a, b).ratio()
    
    def _contains_fuzzy_word(self, tokens: list[str], word: str, threshold: float = 0.8) -> bool:
        """Check if tokens contain a word (fuzzy)."""
        target = word.lower()
        return any(self._fuzzy_ratio(token, target) >= threshold for token in tokens)
    
    def _contains_any_fuzzy_word(self, tokens: list[str], words: list[str], threshold: float = 0.8) -> bool:
        """Check if any of the target words appear fuzzily in tokens."""
        return any(self._contains_fuzzy_word(tokens, word, threshold) for word in words)
    
    def _contains_fuzzy_phrase(self, tokens: list[str], phrase_tokens: list[str], threshold: float = 0.8) -> bool:
        """Check if sequence of tokens approximately matches phrase."""
        if not tokens or not phrase_tokens:
            return False
        length = len(phrase_tokens)
        for i in range(len(tokens) - length + 1):
            if all(self._fuzzy_ratio(tokens[i + j], phrase_tokens[j]) >= threshold for j in range(length)):
                return True
        return False
    
    def _heuristic_detect_month_ranking(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Heuristic detection for month-based ranking queries like:
        "Top 2 brands by revenue in August 2024 and each brand's % share of that month".
        Returns a ranking plan with explicit month range when detected.
        """
        try:
            q = (question or "").lower()
            tokens = self._tokenize_text(question)
            if not tokens:
                return None
            
            # Quick gates: requires "top" (fuzzy) and a recognised entity keyword
            if not self._contains_fuzzy_word(tokens, "top", threshold=0.75):
                return None
            
            entity_candidates = [
                (["brand", "brands", "product", "products"], "brand_name"),
                (["customer", "customers", "client", "clients", "buyer", "buyers", "account", "accounts"], "customer_name"),
                (["salesman", "salesmen", "salesperson", "rep", "reps", "representative", "representatives"], "salesman_name"),
            ]
            entity = None
            for keywords, candidate in entity_candidates:
                if self._contains_any_fuzzy_word(tokens, keywords, threshold=0.75):
                    entity = candidate
                    break
            if not entity:
                return None
            
            # Extract top N
            n = None
            m = re.search(r"top\s+(\d+)", q)
            if m:
                try:
                    n = int(m.group(1))
                except Exception:
                    n = None
            if n is None:
                n = 10
            # Month names map
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month_num = None
            for name, num in month_names.items():
                if re.search(rf"\b{name}\b", q):
                    month_num = num
                    break
            if not month_num:
                return None
            # Extract year or default to dataset max year
            year_match = re.search(r"20\d{2}", q)
            if year_match:
                year = int(year_match.group(0))
            else:
                # Default: dataset max year
                date_col = columns.get('date', 'VISITDATE')
                year = None
                if date_col in df.columns:
                    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                    if not dates.empty:
                        year = int(dates.max().year)
                if year is None:
                    return None
            # Build explicit month range
            from datetime import datetime
            start = datetime(year, month_num, 1)
            if month_num == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month_num + 1, 1)
            # Build requirements and plan heuristically
            requirements = {
                "group_by": [entity],
                "metrics": [{"name": "revenue", "agg": "SUM"}],
                "time": {
                    "type": "point",
                    "axis": "month",
                    "span": "explicit_range",
                    "explicit": {"start": start.strftime('%Y-%m-%d'), "end": end.strftime('%Y-%m-%d')}
                },
                "outputs": {"table": True, "share": True, "top_n": n},
                "filters": [],
                "presentation": {"sort": ["revenue:desc", f"{entity}:asc"]},
                "confidence": 0.9,
                "reason": "Heuristic month ranking"
            }
            plan = {
                "intent": "ranking",
                "time_grain": "month",
                "time_window": {
                    "mode": "explicit_range",
                    "explicit": {"start": start.strftime('%Y-%m-%d'), "end": end.strftime('%Y-%m-%d')}
                },
                "compare": None,
                "entity": entity,
                "measure": {"name": "revenue", "expr": "SUM(value)"},
                "top_n": n,
                "constraints": [],
                "filters": [],
                "confidence": 0.9,
                "reason": "Heuristic month ranking",
            }
            # Attach requirements for routing preemption
            plan['_requirements'] = requirements
            plan['_repairs_applied'] = 0
            plan['_question'] = question
            return plan
        except Exception:
            return None
    
    def _heuristic_salesman_range_totals(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Heuristic for explicit date range revenue per salesman queries.
        Example: "Show total revenue per salesman for 2025-01-01 to 2025-09-30 (inclusive)".
        """
        tokens = self._tokenize_text(question)
        if not tokens:
            return None
        salesman_keywords = ["salesman", "salesmen", "salesperson", "rep", "representative", "agent"]
        if not self._contains_any_fuzzy_word(tokens, salesman_keywords, threshold=0.75):
            return None
        # Require pattern similar to "per <salesman>" or "by <salesman>" using fuzzy matching
        if not (self._contains_fuzzy_phrase(tokens, ["per", "salesman"], threshold=0.7) or
                self._contains_fuzzy_phrase(tokens, ["by", "salesman"], threshold=0.7)):
            # allow generic cases if question explicitly says "per" anywhere
            if not self._contains_fuzzy_word(tokens, "per", threshold=0.75):
                return None
        if not columns.get('salesman'):
            return None
        date_range = self._parse_explicit_date_range(question)
        if not date_range:
            return None
        start_date, end_inclusive = date_range
        start_date = start_date.normalize()
        end_inclusive = end_inclusive.normalize()
        if start_date == end_inclusive:
            # Single day range still valid; add one day for exclusivity
            end_exclusive = end_inclusive + pd.Timedelta(days=1)
        else:
            end_exclusive = end_inclusive + pd.Timedelta(days=1)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_exclusive.strftime('%Y-%m-%d')
        requirements = {
            "group_by": ["salesman_name"],
            "metrics": [{"name": "revenue", "agg": "SUM"}],
            "time": {
                "type": "range",
                "axis": "day",
                "span": "explicit_range",
                "explicit": {"start": start_str, "end": end_str}
            },
            "outputs": {"table": True, "delta": False, "pct_change": False, "share": False, "top_n": None},
            "filters": [],
            "presentation": {"sort": ["revenue:desc", "salesman_name:asc"]},
            "confidence": 0.92,
            "reason": "Explicit range revenue per salesman"
        }
        plan = {
            "intent": "generic_agg",
            "time_grain": None,
            "time_window": {"mode": "explicit_range", "explicit": {"start": start_str, "end": end_str}},
            "compare": None,
            "entity": "salesman_name",
            "measure": {"name": "revenue", "expr": "SUM(value)"},
            "top_n": None,
            "constraints": [],
            "filters": [],
            "confidence": 0.92,
            "reason": "Explicit range revenue per salesman",
            "_requirements": requirements,
            "_repairs_applied": 0,
            "_question": question,
            "_include_transactions": True
        }
        return plan
    
    def _heuristic_last_completed_quarter_comparison(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Heuristic for "last completed quarter vs previous" per-salesman comparison queries.
        """
        question_raw = question or ""
        question_lower = question_raw.lower()
        tokens = self._tokenize_text(question)
        if not tokens:
            return None
        if not self._contains_fuzzy_phrase(tokens, ["last", "completed", "quarter"], threshold=0.7):
            return None
        comparison_signal = (
            self._contains_any_fuzzy_word(tokens, ["previous", "prior", "versus", "vs"], threshold=0.7)
            or any(sym in question_raw for sym in ["Δ", "∆", "%Δ", "%∆"])
            or any(term in question_lower for term in ["delta", "pct change", "pct-change", "% change", "percent change"])
            or bool(re.search(r'include[^a-z0-9]{0,5}(delta|Δ|∆|percent|%|pct)', question_raw, re.IGNORECASE))
        )
        if not comparison_signal:
            return None
        salesman_keywords = ["salesman", "salesmen", "salesperson", "rep", "representative"]
        if not self._contains_any_fuzzy_word(tokens, salesman_keywords, threshold=0.75):
            return None
        if not columns.get('salesman'):
            return None
        requirements = {
            "group_by": ["salesman_name"],
            "metrics": [{"name": "revenue", "agg": "SUM"}],
            "time": {
                "type": "comparison",
                "axis": "quarter",
                "span": "last_completed_quarter",
                "explicit": None
            },
            "outputs": {"table": True, "delta": True, "pct_change": True, "share": False, "top_n": None},
            "filters": [],
            "presentation": {"sort": ["revenue:desc", "salesman_name:asc"]},
            "confidence": 0.95,
            "reason": "Quarter-over-quarter per salesman"
        }
        plan = {
            "intent": "period_comparison",
            "time_grain": "quarter",
            "time_window": {"mode": "relative_to_dataset_max", "range": "last_completed_quarter"},
            "compare": {"base": "last_completed_quarter", "previous_by": 1},
            "entity": "salesman_name",
            "measure": {"name": "revenue", "expr": "SUM(value)"},
            "top_n": None,
            "constraints": [],
            "filters": [],
            "confidence": 0.95,
            "reason": "Quarter-over-quarter per salesman",
            "_requirements": requirements,
            "_repairs_applied": 0,
            "_question": question
        }
        return plan
    
    def _heuristic_highest_revenue_day(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Heuristic for questions asking for the highest revenue day in a given year.
        """
        q_lower = (question or "").lower()
        if "highest revenue" not in q_lower or "day" not in q_lower:
            return None
        if not columns.get('date'):
            return None
        year_match = re.search(r'20\d{2}', question)
        if year_match:
            year = int(year_match.group(0))
        else:
            date_col = columns.get('date')
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if dates.empty:
                    return None
                year = int(dates.max().year)
            except Exception:
                return None
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1)
        requirements = {
            "group_by": ["visit_date"],
            "metrics": [{"name": "revenue", "agg": "SUM"}],
            "time": {
                "type": "range",
                "axis": "day",
                "span": "explicit_range",
                "explicit": {
                    "start": start_date.strftime('%Y-%m-%d'),
                    "end": end_exclusive.strftime('%Y-%m-%d')
                }
            },
            "outputs": {"table": True, "delta": False, "pct_change": False, "share": False, "top_n": 1},
            "filters": [],
            "presentation": {"sort": ["revenue:desc", "visit_date:asc"]},
            "confidence": 0.92,
            "reason": "Highest revenue day in year"
        }
        plan = {
            "intent": "generic_agg",
            "time_grain": "day",
            "time_window": {
                "mode": "explicit_range",
                "explicit": {
                    "start": start_date.strftime('%Y-%m-%d'),
                    "end": end_exclusive.strftime('%Y-%m-%d')
                }
            },
            "compare": None,
            "entity": "visit_date",
            "measure": {"name": "revenue", "expr": "SUM(value)"},
            "top_n": 1,
            "constraints": [],
            "filters": [],
            "confidence": 0.92,
            "reason": "Highest revenue day in year",
            "_requirements": requirements,
            "_repairs_applied": 0,
            "_question": question,
            "_include_transactions": True
        }
        return plan
    
    def _heuristic_negative_revenue_months(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Heuristic for questions about months with negative net revenue.
        """
        q_lower = (question or "").lower()
        if "negative" not in q_lower or "month" not in q_lower:
            return None
        date_col = columns.get('date')
        value_col = columns.get('amount')
        if not date_col or not value_col:
            return None
        if date_col not in df.columns or value_col not in df.columns:
            return None
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        except Exception:
            return None
        if dates.empty:
            return None
        from dateutil.relativedelta import relativedelta
        start_date = dates.min().replace(day=1)
        end_exclusive = dates.max().replace(day=1) + relativedelta(months=1)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_exclusive.strftime('%Y-%m-%d')
        
        requirements = {
            "group_by": [],
            "metrics": [{"name": "revenue", "agg": "SUM"}],
            "time": {
                "type": "range",
                "axis": "month",
                "span": "explicit_range",
                "explicit": {"start": start_str, "end": end_str}
            },
            "outputs": {"table": True, "delta": False, "pct_change": False, "share": False, "top_n": None},
            "filters": [],
            "presentation": {"sort": ["month:asc"]},
            "confidence": 0.9,
            "reason": "Detect months with negative net revenue"
        }
        
        plan = {
            "intent": "timeseries",
            "time_grain": "month",
            "time_window": {
                "mode": "explicit_range",
                "explicit": {"start": start_str, "end": end_str}
            },
            "compare": None,
            "entity": None,
            "measure": {"name": "revenue", "expr": "SUM(value)"},
            "top_n": None,
            "constraints": [],
            "filters": [],
            "confidence": 0.9,
            "reason": "Detect months with negative net revenue",
            "_requirements": requirements,
            "_repairs_applied": 0,
            "_question": question,
            "_negative_only": True
        }
        return plan
    
    def _heuristic_top_performers(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """Heuristic for queries like "Who are the top performers last quarter"."""
        tokens = self._tokenize_text(question)
        if not tokens:
            return None
        # Require fuzzy match for "top" and a performer keyword
        if not self._contains_fuzzy_word(tokens, "top", threshold=0.75):
            return None
        performer_keywords = [
            "performer", "performers", "seller", "sellers", "sales", "salesman",
            "salesmen", "team", "rep", "reps", "representative"
        ]
        if not self._contains_any_fuzzy_word(tokens, performer_keywords, threshold=0.7):
            return None
        # Require last quarter phrasing
        if not (
            self._contains_fuzzy_phrase(tokens, ["last", "quarter"], threshold=0.7) or
            self._contains_fuzzy_phrase(tokens, ["last", "completed", "quarter"], threshold=0.7)
        ):
            return None
        if not columns.get('salesman'):
            return None
        date_col = columns.get('date', 'VISITDATE')
        if date_col not in df.columns:
            return None
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return None
        dataset_max = dates.max().to_pydatetime()
        from .query_plan_service import TimeFilter, TimeMode
        from .time_resolver import TimeResolver
        time_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
        start_date, end_exclusive, _ = TimeResolver.resolve_time_window(time_filter, dataset_max, dataset_max)
        if start_date is None or end_exclusive is None:
            return None
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_exclusive.strftime('%Y-%m-%d')
        # Determine requested top N (default 10)
        top_n = None
        match = re.search(r"top\s+(\d+)", (question or "").lower())
        if match:
            try:
                top_n = int(match.group(1))
            except Exception:
                top_n = None
        if not top_n:
            top_n = 10
        requirements = {
            "group_by": ["salesman_name"],
            "metrics": [{"name": "revenue", "agg": "SUM"}],
            "time": {
                "type": "point",
                "axis": "quarter",
                "span": "last_completed_quarter",
                "explicit": {"start": start_str, "end": end_str}
            },
            "outputs": {"table": True, "delta": False, "pct_change": False, "share": False, "top_n": top_n},
            "filters": [],
            "presentation": {"sort": ["revenue:desc", "salesman_name:asc"]},
            "confidence": 0.9,
            "reason": "Top performers last completed quarter"
        }
        plan = {
            "intent": "ranking",
            "time_grain": "quarter",
            "time_window": {
                "mode": "explicit_range",
                "explicit": {"start": start_str, "end": end_str}
            },
            "compare": None,
            "entity": "salesman_name",
            "measure": {"name": "revenue", "expr": "SUM(value)"},
            "top_n": top_n,
            "constraints": [],
            "filters": [],
            "confidence": 0.9,
            "reason": "Top performers last completed quarter",
            "_requirements": requirements,
            "_repairs_applied": 0,
            "_question": question
        }
        return plan
    
    def _parse_explicit_date_range(self, question: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Parse explicit YYYY-MM-DD date range from question text.
        
        Returns start and end (inclusive) as pandas Timestamps if two dates found.
        """
        matches = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', question)
        if len(matches) < 2:
            return None
        try:
            start = pd.to_datetime(matches[0])
            end = pd.to_datetime(matches[1])
            if end < start:
                start, end = end, start
            return start, end
        except Exception:
            return None
    
    def _check_gemini_availability(self) -> bool:
        """Check if Gemini API is available."""
        api_key = ConfigService.get_gemini_api_key()
        return api_key is not None
    
    def _track_quota_usage(self, model_name: str) -> None:
        """Track API quota usage for monitoring."""
        from datetime import date
        today = date.today()
        
        # Reset daily counter if new day
        if self._quota_tracking['last_reset_date'] != today:
            self._quota_tracking['daily_requests'] = 0
            self._quota_tracking['last_reset_date'] = today
        
        # Increment counters
        self._quota_tracking['daily_requests'] += 1
        if model_name not in self._quota_tracking['model_usage']:
            self._quota_tracking['model_usage'][model_name] = 0
        self._quota_tracking['model_usage'][model_name] += 1
        
        # Log if approaching limits (warn at 80% of free tier)
        if self._quota_tracking['daily_requests'] >= 40:  # 80% of 50
            print(f"[Quota Warning] Daily requests: {self._quota_tracking['daily_requests']}/50 (approaching limit)")
    
    def _get_current_model(self) -> Optional[str]:
        """Get current model with fallback logic."""
        models = ConfigService.GEMINI_MODELS
        if not models:
            return None
        
        # Try current model index first
        if self._current_model_index < len(models):
            return models[self._current_model_index]
        
        # Fallback to first model
        return models[0]
    
    def _switch_to_next_model(self) -> bool:
        """Switch to next available model. Returns True if switched, False if no more models."""
        models = ConfigService.GEMINI_MODELS
        if self._current_model_index + 1 < len(models):
            self._current_model_index += 1
            print(f"[Model Switch] Switching to {models[self._current_model_index]} (index {self._current_model_index})")
            return True
        return False
    
    def _reset_model_selection(self) -> None:
        """Reset model selection to first model."""
        self._current_model_index = 0
    
    def build_schema_json(self, df: pd.DataFrame, columns: Dict[str, str]) -> Dict[str, Any]:
        """
        Build schema JSON for LLM SQL generation.
        
        Args:
            df: DataFrame with sales data
            columns: Column mapping dictionary (e.g., {'date': 'VISITDATE', 'amount': 'VALUE'})
            
        Returns:
            Schema dictionary with tables, columns, metrics, and time aliases
        """
        # Ensure columns is a dictionary (not a string or other type)
        if not isinstance(columns, dict):
            columns = {}
        
        # Get actual column names from mapping
        date_col = columns.get('date', 'VISITDATE')
        amount_col = columns.get('amount', 'VALUE')
        salesman_col = columns.get('salesman', 'Salesman Name')
        customer_col = columns.get('customer', 'CUSTOMERNAME')
        product_col = columns.get('product') or columns.get('brand', 'BRANDNAME')
        
        # Build column definitions
        table_columns = []
        
        # Date column
        if date_col in df.columns:
            table_columns.append({
                "name": date_col,
                "type": "DATE",
                "meaning": "transaction date"
            })
        
        # Amount column
        if amount_col in df.columns:
            table_columns.append({
                "name": amount_col,
                "type": "NUMBER",
                "meaning": "revenue/sales amount"
            })
        
        # Salesman column
        if salesman_col in df.columns:
            table_columns.append({
                "name": salesman_col,
                "type": "TEXT",
                "meaning": "sales representative"
            })
        
        # Customer column
        if customer_col in df.columns:
            table_columns.append({
                "name": customer_col,
                "type": "TEXT",
                "meaning": "customer name"
            })
        
        # Product/Brand column
        if product_col in df.columns:
            table_columns.append({
                "name": product_col,
                "type": "TEXT",
                "meaning": "product/brand name"
            })
        
        # Get date range for context
        date_range = {}
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if not dates.empty:
                date_range = {
                    "min": str(dates.min().date()),
                    "max": str(dates.max().date())
                }
        
        schema = {
            "tables": [
                {
                    "name": "sales",
                    "columns": table_columns
                }
            ],
            "metrics": {
                "sales_value": f"SUM({amount_col})",
                "total_revenue": f"SUM({amount_col})",
                "revenue": f"SUM({amount_col})",
                "avg_order_value": f"AVG({amount_col})",
                "transaction_count": "COUNT(*)",
                "avg_sale": f"AVG({amount_col})"
            },
            "synonyms": {
                "salesman": ["rep", "agent", "sales person", "salesperson", "sales rep", "representative"],
                "revenue": ["sales", "value", "amount", "total", "sales value"],
                "customer": ["client", "account", "buyer"],
                "product": ["item", "sku", "brand", "product code"]
            },
            "time_aliases": {
                "last_completed_quarter": "quarter_before(quarter_of(MAX(VISITDATE)))",
                "last_3_calendar_months": "months_ago(3) .. end_of_month(MAX(VISITDATE))",
                "this_year": f"YEAR({date_col}) = YEAR(MAX({date_col}))",
                "last_year": f"YEAR({date_col}) = YEAR(MAX({date_col})) - 1"
            },
            "date_range": date_range,
            "max_date": date_range.get("max", "") if date_range else ""
        }
        
        return schema
    
    def generate_sql(self, question: str, schema: Dict[str, Any], _retry_count: int = 0) -> Dict[str, Any]:
        """
        Generate SQL from natural language question using LLM.
        
        Args:
            question: User's natural language question
            schema: Schema JSON dictionary
            _retry_count: Internal counter to prevent infinite recursion (max 1 retry)
            
        Returns:
            Dictionary with 'sql' key or 'clarify' key
        """
        if not self.use_llm:
            return {"clarify": "SQL generation requires Gemini API key"}
        
        # Prevent infinite recursion (max 1 model switch retry)
        if _retry_count > 1:
            return {"clarify": "All Gemini models exhausted. Using fallback heuristics."}
        
        # Get current model with fallback support
        model_name = self._get_current_model()
        if not model_name:
            return {"clarify": "No Gemini models configured"}
        
        # Track quota usage
        self._track_quota_usage(model_name)
        
        try:
            model = genai.GenerativeModel(model_name)
            
            # Get date and amount column names for reference
            date_col_name = None
            amount_col_name = None
            try:
                if isinstance(schema, dict) and 'tables' in schema:
                    tables = schema.get('tables', [])
                    if tables and isinstance(tables[0], dict) and 'columns' in tables[0]:
                        columns_list = tables[0].get('columns', [])
                        if columns_list:
                            for col in columns_list:
                                if isinstance(col, dict):
                                    col_type = col.get('type', '').upper()
                                    col_name = col.get('name')
                                    if col_type == 'DATE' and not date_col_name:
                                        date_col_name = col_name
                                    elif col_type == 'NUMBER' and not amount_col_name:
                                        # Check if it's the amount/revenue column
                                        meaning = col.get('meaning', '').lower()
                                        if 'revenue' in meaning or 'amount' in meaning or 'value' in meaning:
                                            amount_col_name = col_name
                                        elif not amount_col_name:
                                            # Fallback: use first NUMBER column as amount
                                            amount_col_name = col_name
            except (KeyError, TypeError, IndexError) as e:
                ErrorHandlingService.log_error(
                    e,
                    category=ErrorCategory.DATA_PROCESSING
                )
            
            # LLM queries must target the normalized view (sales_norm) with snake_case columns
            date_col_ref = 'visit_date'
            amount_col_ref = 'value'
            
            # Build semantic layer info for prompt
            synonyms_info = ""
            if 'synonyms' in schema:
                synonyms_info = "\nSynonyms (use these for column matching):\n"
                for key, values in schema['synonyms'].items():
                    synonyms_info += f"  - {key}: {', '.join(values)}\n"
            
            prompt = f"""Return JSON only. Keys: {{"sql"}} or {{"clarify"}}. No prose.

Rules:
- Read-only SQL only (SELECT statements)
- No DDL/DML (CREATE, INSERT, UPDATE, DELETE, DROP, COPY, ATTACH, LOAD, HTTPFS, etc.)
- Use semantic layer: metrics, dimensions, and synonyms from schema below
{synonyms_info}
- Query the normalized view `sales_norm` (already registered) which exposes snake_case columns:
  visit_date (DATE), value (NUMBER), salesman_name, customer_name, brand_name.
- Do not reference the raw uploaded table or mixed-case column names.
- Use DATE_TRUNC('month', {date_col_ref}) AS month for monthly grouping
- For "month where exactly one salesman had revenue" queries:
  Use pattern: GROUP BY month, salesman → SUM(value) AS revenue, then
  HAVING clause using COUNT_IF(revenue > 0) per month = 1.
  Return (month, salesman, amount).
- For date range filters, ALWAYS use END-EXCLUSIVE bounds to avoid time edge cases:
  WHERE DATE({date_col_ref}) >= DATE 'YYYY-MM-DD' AND DATE({date_col_ref}) < DATE 'YYYY-MM-DD'
  Example: For range 2025-01-01 to 2025-09-30 (inclusive), use:
  WHERE DATE({date_col_ref}) >= DATE '2025-01-01' AND DATE({date_col_ref}) < DATE '2025-10-01'
  Always cast timestamp columns to DATE in date predicates
- For "last completed quarter", compute the full quarter strictly before quarter(MAX({date_col_ref}))
  Example: If MAX(date) is 2025-09-28 (in Q3-2025), last completed quarter is Q2-2025 (Apr-Jun 2025)
  Use: DATE({date_col_ref}) >= DATE '2025-04-01' AND DATE({date_col_ref}) < DATE '2025-07-01'
- For "last N months", use calendar months anchored to MAX({date_col_ref})
  Example: For "last 6 months" with MAX=2025-09-28, use months Apr-Sep 2025:
  DATE({date_col_ref}) >= DATE '2025-04-01' AND DATE({date_col_ref}) < DATE '2025-10-01'
- For monthly breakdowns, include ALL months (Jan-Dec) even if zero sales
- For month-over-month comparisons, use LAG() window function to compare each month to previous:
  WITH monthly AS (
    SELECT DATE_TRUNC('month', {date_col_ref}) AS m, SUM({amount_col_ref}) AS total
    FROM sales_norm WHERE ...
    GROUP BY 1
  ),
  deltas AS (
    SELECT m, total, LAG(total) OVER (ORDER BY m) AS prev
    FROM monthly
  )
  SELECT m, total, (total - prev) AS abs_change,
    CASE WHEN prev = 0 THEN NULL ELSE (total - prev) / prev * 100 END AS pct_change
  FROM deltas WHERE prev IS NOT NULL ORDER BY m
- Always reference table/view name `sales_norm`
- Limit results to {self.MAX_RESULT_ROWS} rows (add LIMIT clause)
- Format numbers with ROUND(..., 2) for currency
- Use proper DATE functions (DATE_TRUNC, EXTRACT, DATE(), etc.)
- Never use SELECT * without GROUP BY or LIMIT

Examples:
Q: "Top 2 brands by revenue in August 2024 and each brand’s % share"
{{"sql": "\"\"\"
WITH month_window AS (
  SELECT DATE '2024-08-01' AS start_date,
         DATE '2024-09-01' AS end_date
),
brand_totals AS (
  SELECT brand_name,
         SUM(value) AS revenue
  FROM sales_norm, month_window
  WHERE visit_date >= start_date AND visit_date < end_date
  GROUP BY brand_name
),
month_total AS (
  SELECT SUM(value) AS total_revenue
  FROM sales_norm, month_window
  WHERE visit_date >= start_date AND visit_date < end_date
)
SELECT
  bt.brand_name,
  ROUND(bt.revenue, 2) AS revenue,
  ROUND(bt.revenue / NULLIF(mt.total_revenue, 0) * 100, 2) AS pct_share
FROM brand_totals bt
CROSS JOIN month_total mt
ORDER BY bt.revenue DESC, bt.brand_name ASC
LIMIT 2
\"\"\"}}

Q: "Who are the top performers last quarter"
{{"sql": "\"\"\"
WITH dataset_bounds AS (
  SELECT DATE_TRUNC('quarter', MAX(visit_date)) AS current_q_start
  FROM sales_norm
),
last_q AS (
  SELECT current_q_start - INTERVAL 3 MONTH AS start_date,
         current_q_start AS end_date
  FROM dataset_bounds
)
SELECT
  salesman_name,
  ROUND(SUM(value), 2) AS total_revenue
FROM sales_norm, last_q
WHERE visit_date >= start_date AND visit_date < end_date
GROUP BY salesman_name
ORDER BY total_revenue DESC, salesman_name ASC
LIMIT 10
\"\"\"}}

SCHEMA:
{json.dumps(schema, indent=2, ensure_ascii=False)}

QUESTION:
{question}

Return JSON only with either "sql" or "clarify" key:
"""
            
            response = model.generate_content(prompt)
            
            # Parse JSON response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith('```'):
                text = re.sub(r'```json?\s*', '', text).strip()
                text = text.rstrip('```').strip()
            
            # Parse JSON - ensure it's a dict
            try:
                plan = json.loads(text)
                if not isinstance(plan, dict):
                    # If LLM returned a string, wrap it
                    return {"clarify": f"LLM returned non-dict response: {type(plan).__name__}"}
                # Add model info to response
                plan["model_used"] = model_name
                return plan
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return clarify
                return {"clarify": f"Could not parse LLM response as JSON: {str(e)}"}
            
        except json.JSONDecodeError as e:
            ErrorHandlingService.log_error(
                f"Failed to parse SQL generation response: {e}",
                ErrorCategory.DATA_PROCESSING
            )
            return {"clarify": "Could not parse SQL generation response"}
        except Exception as e:
            error_message = str(e)
            error_lower = error_message.lower()
            
            # Handle quota/rate limit errors with model fallback
            if '429' in error_message or 'quota' in error_lower or 'rate limit' in error_lower:
                self._quota_tracking['quota_errors'] += 1
                
                # Extract retry delay if available
                retry_delay = None
                if 'retry' in error_lower or 'delay' in error_lower:
                    import re
                    delay_match = re.search(r'(\d+\.?\d*)\s*(?:seconds?|s)', error_lower)
                    if delay_match:
                        retry_delay = float(delay_match.group(1))
                
                # Try next model if available
                if self._switch_to_next_model():
                    print(f"[Quota Error] Model {model_name} hit quota limit. Switching to {self._get_current_model()}")
                    # Retry with new model (recursive call, but only once)
                    try:
                        return self.generate_sql(question, schema, _retry_count=_retry_count + 1)
                    except Exception:
                        pass  # Fall through to kill-switch
                
                # Log quota info
                daily_usage = self._quota_tracking['daily_requests']
                ErrorHandlingService.log_error(
                    f"Gemini quota exceeded: model={model_name}, daily_requests={daily_usage}/50, "
                    f"quota_errors={self._quota_tracking['quota_errors']}, retry_delay={retry_delay}s",
                    category=ErrorCategory.API
                )
                
                # Enable kill-switch if too many quota errors
                if self._quota_tracking['quota_errors'] >= 3:
                    self.llm_auto_fallback_enabled = True
                    st.session_state['llm_degraded'] = True
                    print(f"[Kill-Switch] Activated due to {self._quota_tracking['quota_errors']} quota errors")
                
                retry_msg = f" Please retry in {retry_delay:.1f}s." if retry_delay else " Please retry in a few seconds."
                return {
                    "clarify": f"Gemini API quota/rate limit reached (daily: {daily_usage}/50). "
                              f"I've switched to fallback heuristics.{retry_msg}",
                    "quota_exceeded": True,
                    "retry_delay": retry_delay
                }
            
            # Handle 404 errors (model not found) - try next model
            if '404' in error_message and ('not found' in error_lower or 'not supported' in error_lower):
                print(f"[Model Error] Model {model_name} not found or not supported (404). Trying next model...")
                if self._switch_to_next_model():
                    next_model = self._get_current_model()
                    print(f"[Model Switch] Switching from {model_name} to {next_model}")
                    try:
                        return self.generate_sql(question, schema, _retry_count=_retry_count + 1)
                    except Exception as retry_error:
                        ErrorHandlingService.log_error(
                            f"Failed to switch to {next_model}: {retry_error}",
                            category=ErrorCategory.API
                        )
                        return {
                            "clarify": f"Model {model_name} is not available. All models exhausted. Using fallback heuristics."
                        }
                else:
                    return {
                        "clarify": f"Model {model_name} is not available and no fallback models configured. Using fallback heuristics."
                    }
            
            ErrorHandlingService.log_error(
                f"SQL generation error: {e}",
                ErrorCategory.DATA_PROCESSING
            )
            return {"clarify": f"SQL generation failed: {error_message}"}
    
    def validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL for safety and correctness with token-level guardrails.
        Rejects multi-statements, DDL/DML/IO, UDFs, caps LIMIT, enforces single SELECT.
        Uses tokenization to detect obfuscated forms.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"
        
        # Token-level validation: split SQL into tokens
        # Normalize whitespace (collapse multiple spaces, tabs, newlines)
        sql_normalized = re.sub(r'\s+', ' ', sql.strip())
        sql_upper = sql_normalized.upper()
        
        # Tokenize: split on whitespace and punctuation boundaries
        tokens = re.findall(r'\b\w+\b|[^\w\s]', sql_upper)
        
        # Reject multi-statements (semicolons) - check in tokens and original
        semicolon_count = sql.count(';')
        if semicolon_count > 1:
            return False, f"Rejected plan: multiple statements detected ({semicolon_count} semicolons)"
        
        # Reject comments (-- and /* */) - check in normalized SQL
        if '--' in sql_normalized:
            return False, "Rejected plan: SQL comments (--) detected"
        if '/*' in sql_normalized and '*/' in sql_normalized:
            return False, "Rejected plan: multi-line comments (/* */) detected"
        
        # Check for blacklisted keywords (DDL/DML/IO operations) - token-level
        denied_keywords = [
            'PRAGMA', 'ATTACH', 'DETACH', 'LOAD', 'INSTALL', 'COPY', 'EXPORT',
            'CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE',
            'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL', 'HTTPFS',
            'READ_CSV', 'READ_PARQUET'  # Only allow in ingest path, not user query path
        ]
        for keyword in denied_keywords:
            if keyword in tokens:
                return False, f"Rejected plan: disallowed keyword detected ({keyword})"
        
        # Must be SELECT only (allowlist approach)
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Explicit allowlist of SQL keywords (token-level check)
        allowed_keywords = {
            'SELECT', 'WITH', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING',
            'ORDER', 'LIMIT', 'AS', 'AND', 'OR', 'NOT', 'IN', 'IS',
            'DATE_TRUNC', 'CAST', 'SUM', 'AVG', 'COUNT', 'COUNT_IF',
            'ROUND', 'COALESCE', 'LAG', 'OVER', 'PARTITION', 'JOIN',
            'LEFT', 'INNER', 'USING', 'ON', 'DATE', 'INTERVAL',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'NULL', 'LIKE',
            'BETWEEN', 'EXTRACT', 'YEAR', 'MONTH', 'DAY', 'MAX', 'MIN'
        }
        
        # Check for suspicious token patterns (obfuscated keywords)
        suspicious_patterns = [
            (r'UNION\s+SELECT', 'UNION injection'),
            (r'EXEC\s*\(', 'EXEC calls'),
        ]
        for pattern, description in suspicious_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False, f"Rejected plan: {description} detected"
        
        # Reject UDFs (function calls that might be unsafe)
        unsafe_functions = ['EXEC', 'EXECUTE', 'CALL', 'EVAL', 'RUN']
        for func in unsafe_functions:
            if re.search(rf'\b{func}\s*\(', sql_upper):
                return False, f"Rejected plan: unsafe function call detected ({func})"
        
        # Check for SELECT * without GROUP BY or LIMIT (potentially unsafe)
        if "SELECT *" in sql_upper:
            if "GROUP BY" not in sql_upper and "LIMIT" not in sql_upper:
                return False, "SELECT * without GROUP BY or LIMIT is not allowed"
        
        # Cap LIMIT clause
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > self.MAX_RESULT_ROWS:
                return False, f"LIMIT ({limit_value}) exceeds maximum allowed ({self.MAX_RESULT_ROWS})"
        
        return True, None
    
    def execute_sql(self, con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
        """
        Execute SQL safely on DuckDB connection.
        
        Args:
            con: DuckDB connection
            sql: SQL query string
            
        Returns:
            DataFrame with results
        """
        # Validate SQL
        is_valid, error_msg = self.validate_sql(sql)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Execute with timeout and row limit
        try:
            # Detect aggregate queries (GROUP BY or aggregate functions without GROUP BY)
            # For aggregates, don't add LIMIT (not needed, and can break results)
            sql_upper = sql.upper()
            is_aggregate_query = (
                'GROUP BY' in sql_upper or
                any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN(', 'COUNT_IF('])
            )
            
            # Add LIMIT only for non-aggregate queries (row-level results)
            if 'LIMIT' not in sql_upper and not is_aggregate_query:
                sql = f"{sql.rstrip(';')} LIMIT {self.MAX_RESULT_ROWS}"
            
            result = con.execute(sql).fetchdf()
            
            # Ensure result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                ErrorHandlingService.log_error(
                    f"DuckDB returned non-DataFrame: {type(result).__name__}",
                    category=ErrorCategory.DATA_PROCESSING
                )
                # Return empty DataFrame as fallback
                return pd.DataFrame()
            
            # Memory cap: limit result rows before returning
            if len(result) > self.MAX_RESULT_ROWS:
                result = result.head(self.MAX_RESULT_ROWS)
                ErrorHandlingService.log_error(
                    f"Result capped to {self.MAX_RESULT_ROWS} rows",
                    category=ErrorCategory.DATA_PROCESSING
                )
            
            # Memory cap: check result size in bytes
            try:
                result_bytes = result.memory_usage(deep=True).sum()
                if result_bytes > self.MAX_RESULT_BYTES:
                    # Cap to approximate byte limit
                    rows_per_mb = len(result) / (result_bytes / (1024 * 1024))
                    max_rows = int((self.MAX_RESULT_BYTES / (1024 * 1024)) * rows_per_mb)
                    result = result.head(max_rows)
                    ErrorHandlingService.log_error(
                        f"Result capped to {max_rows} rows (size limit: {self.MAX_RESULT_BYTES / (1024*1024):.1f}MB)",
                        category=ErrorCategory.DATA_PROCESSING
                    )
            except Exception:
                # If memory calculation fails, just cap rows
                result = result.head(self.MAX_DISPLAY_ROWS)
            
            return result
            
        except Exception as e:
            ErrorHandlingService.log_error(
                f"SQL execution error: {e}",
                ErrorCategory.DATA_PROCESSING
            )
            raise
    
    def fill_zero_months(self, df: pd.DataFrame, date_col: str, year: Optional[int] = None, start_date: Optional[pd.Timestamp] = None, end_date_exclusive: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Fill zero months for monthly series queries.
        Handles cross-year boundaries (e.g., Nov→Apr).
        
        Args:
            df: DataFrame with monthly data
            date_col: Name of date column
            year: Target year (if None, uses date range from df)
            start_date: Explicit start date (optional)
            end_date_exclusive: Explicit end date exclusive (optional)
            
        Returns:
            DataFrame with all months filled (zeros included)
        """
        if df.empty or date_col not in df.columns:
            return df
        
        # Convert to period
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Determine date range (prefer explicit, then year, then from data)
        if start_date is not None and end_date_exclusive is not None:
            # Use explicit range (cross-year safe)
            start = pd.Timestamp(start_date).replace(day=1)
            end_excl = pd.Timestamp(end_date_exclusive).replace(day=1)
        elif year:
            start = pd.Timestamp(f"{year}-01-01")
            end_excl = pd.Timestamp(f"{year+1}-01-01")  # Exclusive
        else:
            dates = df_copy[date_col].dropna()
            if dates.empty:
                return df
            start = dates.min().replace(day=1)
            end_excl = dates.max().replace(day=1) + pd.offsets.MonthEnd(0) + pd.offsets.Day(1)
        
        # Create all months in range (cross-year safe)
        idx = pd.period_range(start=start, end=end_excl - pd.offsets.Day(1), freq='M')
        
        # If df has a month column, reindex
        if 'month' in df_copy.columns:
            df_copy['month_period'] = pd.to_datetime(df_copy['month'], errors='coerce').dt.to_period('M')
            df_copy = df_copy.set_index('month_period')
            df_copy = df_copy.reindex(idx, fill_value=0)
            df_copy = df_copy.reset_index()
        elif date_col in df_copy.columns:
            # Group by month and reindex
            df_copy['month_period'] = df_copy[date_col].dt.to_period('M')
            monthly = df_copy.groupby('month_period').sum(numeric_only=True)
            monthly = monthly.reindex(idx, fill_value=0)
            monthly = monthly.reset_index()
            monthly['month'] = monthly['month_period'].astype(str)
            df_copy = monthly
        
        return df_copy
    
    def process_query(self, question: str, df: pd.DataFrame, columns: Dict[str, str]) -> Dict[str, Any]:
        """
        Process natural language query: generate SQL, execute, and return results.
        
        Includes observability: logs question, resolved window, SQL, row counts.
        
        Args:
            question: User's natural language question
            df: DataFrame with sales data
            columns: Column mapping dictionary
            
        Returns:
            Dictionary with 'data' (DataFrame), 'sql' (query used), 'observability' (metadata), or 'error'
        """
        import hashlib
        start_time = time.time()
        
        # Schema drift detection: validate required columns
        is_valid, error_msg = DatasetCacheService.validate_required_columns(columns, df)
        if not is_valid:
            return {
                "error": f"Schema validation failed: {error_msg}. Please check your column mapping.",
                "observability": {"error": error_msg, "schema_valid": False}
            }
        # Compute dataset hash for cache invalidation
        dataset_hash = DatasetCacheService.compute_dataset_hash(df, columns)
        
        # Check if cache should be invalidated
        cached_hash = st.session_state.get('dataset_hash')
        if cached_hash and DatasetCacheService.should_invalidate_cache(cached_hash, dataset_hash):
            # Invalidate caches
            self._dataset_cache.clear()
            if 'normalized_view_created' in st.session_state:
                del st.session_state['normalized_view_created']
        
        # Store current hash
        st.session_state['dataset_hash'] = dataset_hash
        
        # Observability metadata
        obs = {
            "question": question,
            "dataset_rows": len(df) if df is not None else 0,
            "path": "sql_generation",
            "sql": None,
            "period_label": None,
            "scanned_rows": None,
            "returned_rows": None,
            "result_hash": None,
            "dataset_hash": dataset_hash[:8],  # First 8 chars for logging
            "duration_ms": None,
            "validator_pass": None,
            "executor": None,
            "confidence": None,
            "schema_valid": True,
            "model_used": None,  # Track which Gemini model was used
            "model_index": self._current_model_index  # Track model selection index
        }
        
        def _finalize():
            """Attach wall-clock duration to observability payload before returning."""
            obs["duration_ms"] = int((time.time() - start_time) * 1000)
        
        # Ensure columns is a dictionary (not a string or other type)
        if not isinstance(columns, dict):
            columns = {}
        
        # Build schema
        schema = self.build_schema_json(df, columns)
        
        # Get dataset max date for time resolution
        date_col = columns.get('date', 'VISITDATE')
        dataset_max_date = None
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if not dates.empty:
                dataset_max_date = dates.max().to_pydatetime()
        
        # Register DataFrame in DuckDB with safe mode
        con = duckdb.connect()
        try:
            # DuckDB safe mode: disable extensions, set limits
            con.execute("PRAGMA disable_object_cache")
            con.execute("PRAGMA threads=4")
            con.execute("PRAGMA memory_limit='512MB'")
            
            con.register("sales", df)
            
            # Create normalized view (gated by dataset_hash cache)
            dataset_hash = st.session_state.get('dataset_hash', '')
            cache_key_view = DatasetCacheService.get_cache_key('normalized_view', dataset_hash) if dataset_hash else None
            
            # Check if normalized view cache is valid for this dataset hash
            if not cache_key_view or cache_key_view not in self._dataset_cache or not st.session_state.get('normalized_view_created', False):
                # Cache miss or invalidated - create new view
                self._create_normalized_view(con, df, columns)
                if cache_key_view:
                    # Store cache entry keyed by dataset_hash
                    if cache_key_view not in self._dataset_cache:
                        self._dataset_cache[cache_key_view] = {}
                    self._dataset_cache[cache_key_view]['normalized_view'] = True
                st.session_state['normalized_view_created'] = True
            
            # Planner guardrails: check required columns before processing
            date_col = columns.get('date')
            value_col = columns.get('amount')
            salesman_col = columns.get('salesman')
            
            # Use LLM/heuristic to understand intent
            plan = self._detect_deterministic_pattern(question, columns, df)
            is_single_salesman_query = plan is not None and plan.get('intent') == 'constraint_timeseries'
            
            if is_single_salesman_query:
                # Assert required columns present
                if not date_col or not value_col or not salesman_col:
                    missing = [k for k, v in [('date', date_col), ('amount', value_col), ('salesman', salesman_col)] if not v]
                    obs["error"] = f"Missing required columns: {', '.join(missing)}"
                    _finalize()
                    return {
                        "error": f"Required columns are missing: {', '.join(missing)}. Please check your data mapping.",
                        "observability": obs
                    }
                
                # Check if columns exist in DataFrame
                if date_col not in df.columns or value_col not in df.columns or salesman_col not in df.columns:
                    missing = [c for c in [date_col, value_col, salesman_col] if c not in df.columns]
                    obs["error"] = f"Columns not found in data: {', '.join(missing)}"
                    _finalize()
                    return {
                        "error": f"Required columns not found in data: {', '.join(missing)}. Please check your data mapping.",
                        "observability": obs
                    }
            
            # Check for deterministic query patterns first (before LLM SQL generation)
            plan = self._detect_deterministic_pattern(question, columns, df)
            if plan:
                intent = plan.get('intent')
                confidence = plan.get('confidence', 0.0)
                requirements = plan.get('_requirements')
                repairs_applied = plan.get('_repairs_applied', 0)
                
                # Store observability metadata (clean plan without internal fields)
                plan_clean = {k: v for k, v in plan.items() if not k.startswith('_')}
                obs["intent"] = intent
                obs["confidence"] = confidence
                obs["plan"] = plan_clean
                obs["requirements"] = requirements
                obs["repairs_applied"] = repairs_applied
                obs["model_used"] = "heuristic"  # Deterministic plans use heuristics, not LLM
                
                print(f"[SQLGenerationService] Query plan detected: {intent}, confidence: {confidence:.2f}")
                
                # Validate plan
                is_valid, error_msg = self._validate_query_plan(plan, columns, df)
                if not is_valid:
                    obs["error"] = error_msg
                    obs["validator_pass"] = False
                    _finalize()
                    return {"error": error_msg, "observability": obs}
                
                obs["validator_pass"] = True
                
                # Ensure requirements are stored in plan for executor preemption
                if '_requirements' not in plan and requirements:
                    plan['_requirements'] = requirements
                
                # Derive executor from plan content (no keywords, pure structure)
                # Requirements are checked FIRST for preemption
                executor = self.derive_executor(plan)
                obs["executor"] = executor
                
                # Fail-safe: If requirements indicate comparison/deltas, force period_comparison
                time_req_fs = requirements.get('time', {}) if requirements else {}
                outputs_req_fs = requirements.get('outputs', {}) if requirements else {}
                wants_comparison_fs = (
                    time_req_fs.get('type') == 'comparison' or
                    bool(outputs_req_fs.get('delta')) or
                    bool(outputs_req_fs.get('pct_change'))
                )
                if wants_comparison_fs and executor != 'period_comparison':
                    print(f"[SQLGenerationService] Preempting executor to 'period_comparison' due to comparison requirements (was: {executor})")
                    executor = 'period_comparison'
                    obs["executor_preempted"] = True
                    obs["executor"] = executor
                
                # CRITICAL: Fail-fast assert - single-month grouped request must use ranking executor
                if self._wants_ranking(requirements) and executor != 'ranking':
                    error_msg = "Planner/Router mismatch: expected ranking executor for single-month top/share request."
                    obs["mismatch_error"] = error_msg
                    raise RuntimeError(error_msg)
                
                # Enhanced logging for routing decisions (per request diagnostics)
                time_req = requirements.get('time', {}) if requirements else {}
                outputs = requirements.get('outputs', {}) if requirements else {}
                period_label = obs.get("period_label", "N/A")
                log_msg = (
                    f"[SQLGenerationService] Routing decision: intent={intent}, executor_selected={executor}, "
                    f"group_by={requirements.get('group_by', []) if requirements else 'N/A'}, "
                    f"time.type={time_req.get('type', 'N/A')}, time.axis={time_req.get('axis', 'N/A')}, "
                    f"outputs.top_n={outputs.get('top_n', 'N/A')}, outputs.share={outputs.get('share', 'N/A')}, "
                    f"period_label={period_label}"
                )
                print(log_msg)
                
                # Route to appropriate executor
                if executor == "period_comparison":
                    result_df = self._execute_period_comparison_plan(plan, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (period_comparison)"
                    obs["execution_method"] = "local"
                    
                    # Generate period label using TimeResolver (explicit dataset coverage)
                    from .time_resolver import TimeResolver
                    from .query_plan_service import TimeFilter, TimeMode
                    date_col = columns.get('date', 'VISITDATE')
                    if date_col in df.columns:
                        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                        if not dates.empty:
                            max_date = dates.max().to_pydatetime()
                            min_date = dates.min().to_pydatetime()
                            
                            # Check if global date filter is applied
                            global_filter_enabled = st.session_state.get('global_date_filter_enabled', False)
                            global_range = st.session_state.get('global_date_filter_range')
                            
                            compare = plan.get('compare', {})
                            base = compare.get('base', 'last_completed_quarter')
                            if base == 'last_completed_quarter':
                                base_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
                                base_start, base_end_excl, base_label = TimeResolver.resolve_time_window(
                                    base_filter, max_date, max_date
                                )
                                # Get previous quarter label
                                from dateutil.relativedelta import relativedelta
                                prev_start = base_start - relativedelta(months=3)
                                prev_filter = TimeFilter(mode=TimeMode.SPECIFIC_QUARTER, quarter=(prev_start.month - 1) // 3 + 1, year=prev_start.year)
                                prev_start_actual, prev_end_excl, prev_label = TimeResolver.resolve_time_window(
                                    prev_filter, max_date, max_date
                                )
                                
                                # Build explicit period label with dataset coverage
                                if global_filter_enabled and global_range:
                                    obs["period_label"] = f"Dataset coverage: {min_date.date()} → {max_date.date()} (filtered: {global_range[0]} → {global_range[1]}) | QoQ: {base_label} vs {prev_label}"
                                else:
                                    obs["period_label"] = f"Dataset coverage: {min_date.date()} → {max_date.date()} | QoQ: {base_label} vs {prev_label}"
                            else:
                                if global_filter_enabled and global_range:
                                    obs["period_label"] = f"Dataset coverage: {min_date.date()} → {max_date.date()} (filtered: {global_range[0]} → {global_range[1]}) | {base_label}"
                                else:
                                    obs["period_label"] = f"Dataset coverage: {min_date.date()} → {max_date.date()} | {base_label}"
                    
                    # Log observability
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: {executor}, confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    # Format result hash
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    obs["scanned_rows"] = len(df)
                    obs["time_window_label"] = time_window_label
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                elif executor == "constraint_timeseries":
                    result_df = self._execute_constraint_timeseries_plan(plan, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (deterministic)"
                    obs["execution_method"] = "local"
                    
                    # Generate period label using TimeResolver (single time-window engine)
                    time_window = plan.get('time_window', {})
                    if time_window.get('mode') == 'relative_to_dataset_max':
                        # Use TimeResolver for consistent period labeling
                        from .time_resolver import TimeResolver
                        date_col = columns.get('date', 'VISITDATE')
                        if date_col in df.columns:
                            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                            if not dates.empty:
                                max_date = dates.max().to_pydatetime()
                                min_date = dates.min().to_pydatetime()
                                # For "all time" constraint queries, label the actual range
                                if not result_df.empty and 'month' in result_df.columns:
                                    months = result_df['month'].tolist()
                                    if months:
                                        # Get min/max months from results
                                        min_month = min(months)
                                        max_month = max(months)
                                        obs["period_label"] = f"Months with single salesman: {min_month} → {max_month}"
                                else:
                                    # No results - show full dataset range
                                    obs["period_label"] = f"Dataset range: {min_date.date()} → {max_date.date()}"
                    
                    # Log observability (complete metadata)
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: {executor}, confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    # Format result hash
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    # Complete observability metadata
                    obs["scanned_rows"] = len(df)  # Approximate
                    obs["time_window_label"] = time_window_label
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                elif executor == "timeseries":
                    result_df = self._execute_timeseries_plan(plan, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (timeseries)"
                    obs["execution_method"] = "local"
                    obs.pop("period_label", None)
                    obs.pop("time_window_label", None)
                    
                    period_label = result_df.attrs.get('period_label') if hasattr(result_df, 'attrs') else None
                    if period_label:
                        obs["period_label"] = period_label
                        obs["time_window_label"] = period_label
                    if hasattr(result_df, 'attrs'):
                        if result_df.attrs.get('start_date') and result_df.attrs.get('end_date_exclusive'):
                            obs["period_bounds"] = (
                                result_df.attrs['start_date'],
                                result_df.attrs['end_date_exclusive']
                            )
                        if result_df.attrs.get('peak_period'):
                            obs["peak_period"] = result_df.attrs['peak_period']
                            obs["peak_value"] = result_df.attrs.get('peak_value')
                    
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: {executor}, confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    obs["scanned_rows"] = len(df)
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                elif executor == "ranking":
                    result_df = self._execute_ranking_plan(plan, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (ranking)"
                    obs["execution_method"] = "local"
                    obs.pop("period_label", None)
                    obs.pop("time_window_label", None)
                    
                    self._apply_ranking_period_label(plan, columns, df, obs)
                    start_bound = result_df.attrs.get('start_date')
                    end_bound = result_df.attrs.get('end_date_exclusive')
                    if start_bound and end_bound:
                        obs["period_bounds"] = (start_bound, end_bound)
                                
                    # Log observability
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: {executor}, confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    # Format result hash
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    obs["scanned_rows"] = len(df)
                    obs["time_window_label"] = time_window_label
                    
                    # Extract month total from result metadata if available (for ranking queries)
                    if hasattr(result_df, 'attrs') and 'month_total' in result_df.attrs:
                        obs["month_total"] = result_df.attrs['month_total']
                        obs["executor"] = "ranking"
                        obs["rows_returned"] = len(result_df)
                        # Enhanced logging for ranking queries
                        print(f"[SQLGenerationService] Ranking query executed: executor=ranking, period_label={time_window_label}, rows_returned={len(result_df)}, month_total={result_df.attrs['month_total']:.2f}")
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                elif executor == "share":
                    plan_share = {**plan, "intent": "ranking"}
                    result_df = self._execute_ranking_plan(plan_share, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (share)"
                    obs["execution_method"] = "local"
                    obs["executor"] = "share"
                    obs["share_mode"] = True
                    
                    obs.pop("period_label", None)
                    obs.pop("time_window_label", None)
                    self._apply_ranking_period_label(plan_share, columns, df, obs)
                    if hasattr(result_df, 'attrs') and 'month_total' in result_df.attrs:
                        obs["month_total"] = result_df.attrs['month_total']
                    start_bound = result_df.attrs.get('start_date')
                    end_bound = result_df.attrs.get('end_date_exclusive')
                    if start_bound and end_bound:
                        obs["period_bounds"] = (start_bound, end_bound)
                    
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: share (ranking path), confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    obs["scanned_rows"] = len(df)
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                elif executor == "generic_agg":
                    result_df = self._execute_generic_agg_plan(plan, columns, df)
                    obs["returned_rows"] = len(result_df)
                    obs["sql"] = "Local computation (generic_agg)"
                    obs["execution_method"] = "local"
                    
                    period_label = result_df.attrs.get('period_label') if hasattr(result_df, 'attrs') else None
                if period_label:
                    obs["period_label"] = period_label
                    obs["time_window_label"] = period_label
                start_bound = result_df.attrs.get('start_date')
                end_bound = result_df.attrs.get('end_date_exclusive')
                if start_bound and end_bound:
                    obs["period_bounds"] = (start_bound, end_bound)
                    
                    time_window_label = obs.get("period_label", "N/A")
                    print(f"[SQLGenerationService] Plan executed: {intent}, executor: {executor}, confidence: {confidence:.2f}, rows: {len(result_df)}, period: {time_window_label}")
                    
                    if not result_df.empty:
                        result_str = str(result_df.values.tolist()[:10])
                        obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    
                    obs["scanned_rows"] = len(df)
                    
                    _finalize()
                    return {
                        "data": result_df,
                        "sql": obs["sql"],
                        "observability": obs
                    }
                
                else:
                    print(f"[SQLGenerationService] Executor {executor} not yet implemented, falling through to SQL generation")
            
            # Generate SQL
            plan = self.generate_sql(question, schema)
            
            # Capture model used from generate_sql result
            if isinstance(plan, dict) and "model_used" in plan:
                obs["model_used"] = plan["model_used"]
            
            # Ensure plan is a dictionary
            if not isinstance(plan, dict):
                obs["error"] = "SQL generation returned invalid response"
                _finalize()
                return {"error": "SQL generation returned invalid response", "observability": obs}
            
            if "clarify" in plan:
                clarify_msg = plan["clarify"]
                obs["needs_clarification"] = True
                if isinstance(clarify_msg, str):
                    _finalize()
                    return {"needs_clarification": clarify_msg, "observability": obs}
                else:
                    _finalize()
                    return {"needs_clarification": str(clarify_msg), "observability": obs}
            
            if "sql" not in plan:
                obs["error"] = "No SQL generated in response"
                _finalize()
                return {"error": "No SQL generated in response", "observability": obs}
            
            sql = plan["sql"]
            if not isinstance(sql, str):
                obs["error"] = f"Invalid SQL type: {type(sql)}"
                _finalize()
                return {"error": f"Invalid SQL type: {type(sql)}", "observability": obs}
            
            obs["sql"] = sql
            obs["execution_method"] = "llm_sql"
            if not obs.get("executor"):
                obs["executor"] = "llm_sql"
            
            # Execute SQL
            result_df = self.execute_sql(con, sql)
            
            # Ensure result_df is a DataFrame
            if not isinstance(result_df, pd.DataFrame):
                ErrorHandlingService.log_error(
                    f"SQL execution returned non-DataFrame: {type(result_df).__name__}",
                    category=ErrorCategory.DATA_PROCESSING
                )
                obs["error"] = f"SQL execution returned unexpected type: {type(result_df).__name__}"
                _finalize()
                return {"error": obs["error"], "observability": obs}
            
            # Observability: record row counts
            obs["scanned_rows"] = len(df)  # Approximate - actual scan depends on SQL
            obs["returned_rows"] = len(result_df)
            
            # Smart fallback: if SQL returns 0 rows, try local computation
            if result_df.empty:
                fallback_result = self._try_local_fallback(question, df, columns)
                if fallback_result is not None and not fallback_result.empty:
                    result_df = fallback_result
                    obs["fallback_used"] = True
                    obs["sql"] = f"{sql} (fallback: local computation)"
            
            # Generate result hash for caching
            if not result_df.empty:
                result_str = str(result_df.values.tolist()[:10])  # First 10 rows for hash
                obs["result_hash"] = hashlib.md5(result_str.encode()).hexdigest()[:8]
            
            # Guardrails: check if result is empty and provide context-specific message
            # Use LLM/heuristic to understand intent
            plan = self._detect_deterministic_pattern(question, columns, df)
            is_single_salesman_query = plan is not None and plan.get('intent') == 'constraint_timeseries'
            
            if result_df.empty:
                # Re-check intent using LLM/heuristic (in case it wasn't detected earlier)
                plan = self._detect_deterministic_pattern(question, columns, df)
                is_single_salesman_query = plan is not None and plan.get('intent') == 'constraint_timeseries'
            
            if result_df.empty and is_single_salesman_query:
                # Check if required columns exist (already validated above, but double-check)
                date_col = columns.get('date')
                value_col = columns.get('amount')
                salesman_col = columns.get('salesman')
                
                if date_col and value_col and salesman_col:
                    # Check global date filter
                    global_filter_enabled = st.session_state.get('global_date_filter_enabled', False)
                    if global_filter_enabled:
                        global_range = st.session_state.get('global_date_filter_range')
                        if global_range:
                            obs["error"] = f"Global date filter excludes all months. Filter: {global_range}"
                            # Include clear filter action in error message
                            _finalize()
                            return {
                                "error": f"No data after applying the global date filter {global_range[0]} → {global_range[1]}. **Clear filter** and retry.",
                                "observability": obs,
                                "clear_filter_action": True  # Flag for UI to show clear button
                            }
                    
                    # No matches found (valid outcome)
                    obs["error"] = "No month found where exactly one salesman recorded revenue"
                    _finalize()
                    return {
                        "error": "No month found where exactly one salesman recorded revenue.",
                        "observability": obs
                    }
                else:
                    missing = [k for k, v in [('date', date_col), ('amount', value_col), ('salesman', salesman_col)] if not v]
                    obs["error"] = f"Missing required columns: {', '.join(missing)}"
                    _finalize()
                    return {
                        "error": f"Cannot run: required column mapping missing: {', '.join(missing)}",
                        "observability": obs
                    }
            
            # Post-process: fill zero months if needed
            question_lower = question.lower()
            if any(word in question_lower for word in ['monthly', 'month', 'list months']):
                try:
                    # Extract year if mentioned
                    import re
                    year_match = re.search(r'20\d{2}', question)
                    year = int(year_match.group(0)) if year_match else None
                    
                    # Ensure columns is a dict
                    if not isinstance(columns, dict):
                        columns = {}
                    
                    # Find date column
                    date_col = columns.get('date', 'VISITDATE') if isinstance(columns, dict) else 'VISITDATE'
                    if date_col in result_df.columns:
                        result_df = self.fill_zero_months(result_df, date_col, year)
                    else:
                        # If SQL already grouped by month, ensure all months are present
                        month_col = next((c for c in result_df.columns if 'month' in str(c).lower()), None)
                        if month_col:
                            # Reindex to include all months
                            if year:
                                all_months = pd.period_range(start=f"{year}-01", end=f"{year}-12", freq='M')
                                # Ensure month_col is a valid column name (string)
                                if isinstance(month_col, str) and month_col in result_df.columns:
                                    result_df['month_period'] = pd.to_datetime(result_df[month_col], errors='coerce').dt.to_period('M')
                                    result_df = result_df.set_index('month_period')
                                    result_df = result_df.reindex(all_months, fill_value=0)
                                    result_df = result_df.reset_index()
                                    result_df[month_col] = result_df['month_period'].astype(str)
                                else:
                                    ErrorHandlingService.log_error(
                                        f"Invalid month_col: {month_col} (type: {type(month_col)})",
                                        category=ErrorCategory.DATA_PROCESSING
                                    )
                except Exception as e:
                    # If post-processing fails, continue with raw results
                    import traceback
                    error_trace = traceback.format_exc()
                    ErrorHandlingService.log_error(
                        e,
                        category=ErrorCategory.DATA_PROCESSING
                    )
                    # Also print to console for debugging
                    print(f"[Post-processing Error] {e}")
                    print(f"[Post-processing Traceback]\n{error_trace}")
            
            # Log observability info
            print(f"[SQL Query] Question: {question[:100]}...")
            print(f"[SQL Query] SQL: {sql[:200]}...")
            print(f"[SQL Query] Scanned: ~{obs['scanned_rows']} rows, Returned: {obs['returned_rows']} rows")
            if obs.get('result_hash'):
                print(f"[SQL Query] Result hash: {obs['result_hash']}")
            
            _finalize()
            return {
                "data": result_df,
                "sql": sql,
                "schema": schema,
                "observability": obs
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            ErrorHandlingService.log_error(
                e,
                category=ErrorCategory.DATA_PROCESSING
            )
            # Also print to console for debugging
            print(f"[SQL Generation Error] {e}")
            print(f"[SQL Generation Traceback]\n{error_trace}")
            obs["error"] = str(e)
            _finalize()
            return {"error": str(e), "observability": obs}
        finally:
            con.close()
    
    def _create_normalized_view(self, con: duckdb.DuckDBPyConnection, df: pd.DataFrame, columns: Dict[str, str]) -> None:
        """
        Create normalized view with snake_case column names to prevent SQL fragility.
        Normalizes once, used by both LLM and executor.
        
        Args:
            con: DuckDB connection
            df: DataFrame with sales data
            columns: Column mapping dictionary
        """
        try:
            # Get column names from mapping or DataFrame
            date_col = columns.get('date', 'VISITDATE')
            value_col = columns.get('amount', 'VALUE')
            salesman_col = columns.get('salesman', 'Salesman Name')
            brand_col = columns.get('brand', 'BRANDNAME')
            customer_col = columns.get('customer', 'CUSTOMERNAME')
            
            # Build view SQL with all available columns (normalized once)
            view_sql = f"""
            CREATE OR REPLACE VIEW sales_norm AS
            SELECT
                CAST({self._quote_column(date_col)} AS DATE) AS visit_date,
                CAST({self._quote_column(value_col)} AS DOUBLE) AS value,
                {self._quote_column(salesman_col)} AS salesman_name"""
            
            # Add optional columns if they exist
            if brand_col and brand_col in df.columns:
                view_sql += f",\n                {self._quote_column(brand_col)} AS brand_name"
            if customer_col and customer_col in df.columns:
                view_sql += f",\n                {self._quote_column(customer_col)} AS customer_name"
            
            view_sql += "\n            FROM sales"
            
            con.execute(view_sql)
        except Exception as e:
            # If view creation fails, log but don't fail - we can still use raw table
            ErrorHandlingService.log_error(
                f"Failed to create normalized view: {e}",
                category=ErrorCategory.DATA_PROCESSING
            )
    
    def _quote_column(self, col_name: str) -> str:
        """Quote column name if it contains spaces or special characters."""
        if ' ' in col_name or '-' in col_name or not col_name.replace('_', '').isalnum():
            return f'"{col_name}"'
        return col_name
    
    def _parse_intent_via_llm(self, question: str) -> Optional[dict]:
        """
        Parse user question into a structured query plan using LLM.
        Uses a general schema that scales to many query types.
        
        Includes kill-switch: auto-fallback to heuristic if LLM latency > threshold
        or multiple parse failures in a row.
        
        Args:
            question: User's question
            
        Returns:
            Query plan dict with intent, time_grain, entity, measure, constraints, etc.
            Returns None if LLM unavailable or parsing fails.
        """
        # Kill-switch: auto-fallback if enabled
        if self.llm_auto_fallback_enabled:
            return None
        
        if not self.use_llm:
            return None
        
        try:
            start_time = time.time()
            model_name = self._get_current_model()
            if not model_name:
                return None
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""You are an intent parser. Output JSON only.

Rules:
- Never use "today"; anchor time to dataset max date.
- Use normalized view "sales_norm" with snake_case columns: visit_date, value, salesman_name, brand_name, customer_name
- "Exactly one salesman had revenue" means: in a given month, COUNT_DISTINCT(salesman_name) where SUM(value)>0 equals 1.
- Revenue positivity: net revenue > 0 after cleaning (exclude refunds/negatives from "active" count).
- Prefer month time grain for "month"/"monthly" questions.
- Output valid JSON only. No prose.

Schema to return:
{{
  "intent": "constraint_timeseries",
  "time_grain": "month",
  "entity": "salesman_name",
  "measure": {{"name": "revenue", "expr": "SUM(value)"}},
  "constraints": [
    {{
      "type": "distinct_count",
      "of": "salesman_name",
      "where": "SUM(value)>0",
      "eq": 1
    }}
  ],
  "filters": [],
  "time_window": {{"mode": "relative_to_dataset_max", "range": "all"}},
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}

Examples (positive):
- "month where only one rep sold" → intent: "constraint_timeseries", time_grain: "month"
- "months a single salesman contributed" → same intent
- "exactly one salesman instaed revenue" (typo) → same intent

Examples (negative - should NOT trigger):
- "top salesman per month" → different intent (ranking, not constraint)
- "months with no sales" → different intent (filter, not constraint)

User question: {question}

Return only the JSON:"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group(0))
                    # Validate plan structure
                    if plan.get('intent') == 'constraint_timeseries' and plan.get('confidence', 0) >= 0.6:
                        # Success: reset failure count
                        self.llm_failure_count = 0
                        return plan
                except json.JSONDecodeError:
                    # Try parsing the whole response
                    try:
                        plan = json.loads(response_text)
                        if plan.get('intent') == 'constraint_timeseries' and plan.get('confidence', 0) >= 0.6:
                            # Success: reset failure count
                            self.llm_failure_count = 0
                            return plan
                    except json.JSONDecodeError:
                        pass
            
            # Check latency threshold
            latency = time.time() - start_time
            if latency > self.LLM_LATENCY_THRESHOLD:
                self.llm_failure_count += 1
                ErrorHandlingService.log_error(
                    f"LLM latency exceeded threshold: {latency:.2f}s > {self.LLM_LATENCY_THRESHOLD}s",
                    category=ErrorCategory.DATA_PROCESSING
                )
                if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                    self.llm_auto_fallback_enabled = True
                    st.session_state['llm_degraded'] = True
                    print(f"[SQLGenerationService] Kill-switch activated: LLM auto-fallback enabled")
                return None
            
            # Parsing failed (no valid plan returned)
            return None
        except Exception as e:
            self.llm_failure_count += 1
            ErrorHandlingService.log_error(
                f"LLM intent parsing failed: {e}",
                category=ErrorCategory.DATA_PROCESSING
            )
            
            # Kill-switch: enable auto-fallback after threshold failures
            if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                self.llm_auto_fallback_enabled = True
                st.session_state['llm_degraded'] = True
                print(f"[SQLGenerationService] Kill-switch activated after {self.llm_failure_count} failures")
            
            return None
    
    def _parse_requirements_via_llm(self, question: str) -> Optional[dict]:
        """
        Parse user question into Requirements JSON (WHAT the user wants).
        Uses Prompt A: extracts group_by, metrics, time.type, outputs.delta/pct_change, etc.
        
        Args:
            question: User's natural language question
            
        Returns:
            Requirements dict with group_by, metrics, time, outputs, filters, presentation, etc.
            Returns None if LLM unavailable or parsing fails.
        """
        if self.llm_auto_fallback_enabled or not self.use_llm:
            return None
        
        try:
            start_time = time.time()
            model_name = self._get_current_model()
            if not model_name:
                return None
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""You extract analytical requirements from a user question about a single table `sales_norm`.

Columns (all snake_case):
- visit_date: DATE
- value: NUMBER (revenue)
- salesman_name: TEXT
- brand_name: TEXT
- customer_name: TEXT

Rules:
- Output VALID JSON only (no prose).
- Do not invent columns. 
- Time is anchored to the dataset's MAX(visit_date), never "today".
- If the question implies a comparison (e.g., vs, compare, Δ, %Δ), set time.type="comparison".

JSON schema:
{{
  "group_by": ["string"],                // e.g., ["salesman_name"]
  "metrics": [{{"name":"revenue","agg":"SUM"}}], 
  "time": {{ 
    "type": "point|range|comparison",
    "axis": "day|month|quarter|year",
    "span": "last_6_calendar_months|last_completed_quarter|explicit_range|all",
    "explicit": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} | null
  }},
  "outputs": {{ "table": true, "delta": bool, "pct_change": bool, "share": bool, "top_n": null|number }},
  "filters": [{{"field": "...", "op": "=", "value": "..."}}],
  "presentation": {{"sort": ["metric:desc","group_by_field:asc"]}},
  "confidence": 0.0-1.0,
  "reason": "short rationale"
}}

Return ONLY the JSON.

Few-shot examples:

Q: "Last completed quarter vs previous, revenue per salesman, include Δ and %Δ."
{{
  "group_by": ["salesman_name"],
  "metrics": [{{"name": "revenue", "agg": "SUM"}}],
  "time": {{
    "type": "comparison",
    "axis": "quarter",
    "span": "last_completed_quarter",
    "explicit": null
  }},
  "outputs": {{"table": true, "delta": true, "pct_change": true, "top_n": null}},
  "filters": [],
  "presentation": {{"sort": ["revenue:desc", "salesman_name:asc"]}},
  "confidence": 0.9,
  "reason": "QoQ per salesman"
}}

Q: "Top 2 salesmen by revenue in Q3 2024 with share%."
{{
  "group_by": ["salesman_name"],
  "metrics": [{{"name": "revenue", "agg": "SUM"}}],
  "time": {{
    "type": "point",
    "axis": "quarter",
    "span": "explicit_range",
    "explicit": {{"start": "2024-07-01", "end": "2024-10-01"}}
  }},
  "outputs": {{"table": true, "delta": false, "pct_change": false, "top_n": 2}},
  "filters": [],
  "presentation": {{"sort": ["revenue:desc", "salesman_name:asc"]}},
  "confidence": 0.86,
  "reason": "Q3 slice + ranking"
}}

Q: "Top 2 brands by revenue in August 2024 and each brand’s % share of that month"
{{
  "group_by": ["brand_name"],
  "metrics": [{{"name": "revenue", "agg": "SUM"}}],
  "time": {{
    "type": "point",
    "axis": "month",
    "span": "explicit_range",
    "explicit": {{"start": "2024-08-01", "end": "2024-09-01"}}
  }},
  "outputs": {{"table": true, "delta": false, "pct_change": false, "share": true, "top_n": 2}},
  "filters": [],
  "presentation": {{"sort": ["revenue:desc", "brand_name:asc"]}},
  "confidence": 0.92,
  "reason": "Top-2 ranking with share for specific month"
}}

Q: "Who are the top performers last quarter"
{{
  "group_by": ["salesman_name"],
  "metrics": [{{"name": "revenue", "agg": "SUM"}}],
  "time": {{
    "type": "point",
    "axis": "quarter",
    "span": "last_completed_quarter",
    "explicit": null
  }},
  "outputs": {{"table": true, "delta": false, "pct_change": false, "share": false, "top_n": 10}},
  "filters": [],
  "presentation": {{"sort": ["revenue:desc", "salesman_name:asc"]}},
  "confidence": 0.85,
  "reason": "Top performers implies ranking with default top_n=10"
}}

Q: "Show me sales by brand"
{{
  "group_by": ["brand_name"],
  "metrics": [{{"name": "revenue", "agg": "SUM"}}],
  "time": {{
    "type": "range",
    "axis": "all",
    "span": "all",
    "explicit": null
  }},
  "outputs": {{"table": true, "delta": false, "pct_change": false, "share": false, "top_n": null}},
  "filters": [],
  "presentation": {{"sort": ["revenue:desc", "brand_name:asc"]}},
  "confidence": 0.8,
  "reason": "Generic aggregation without explicit time filter"
}}

User question: {question}

Return only the JSON:"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    requirements = json.loads(json_match.group(0))
                    # Validate basic structure
                    if isinstance(requirements, dict) and 'group_by' in requirements:
                        # Success: reset failure count
                        self.llm_failure_count = 0
                        return requirements
                except json.JSONDecodeError:
                    pass
            
            # Try parsing the whole response
            try:
                requirements = json.loads(response_text)
                if isinstance(requirements, dict) and 'group_by' in requirements:
                    self.llm_failure_count = 0
                    return requirements
            except json.JSONDecodeError:
                pass
            
            # Check latency
            latency = time.time() - start_time
            if latency > self.LLM_LATENCY_THRESHOLD:
                self.llm_failure_count += 1
                if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                    self.llm_auto_fallback_enabled = True
                    st.session_state['llm_degraded'] = True
            
            return None
        except Exception as e:
            self.llm_failure_count += 1
            if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                self.llm_auto_fallback_enabled = True
                st.session_state['llm_degraded'] = True
            return None
    
    def _parse_plan_via_llm(self, requirements: dict, question: str) -> Optional[dict]:
        """
        Parse Requirements JSON into execution PLAN JSON (HOW to compute it).
        Uses Prompt B: converts Requirements → execution plan.
        
        Args:
            requirements: Requirements dict from _parse_requirements_via_llm
            question: Original user question (for context)
            
        Returns:
            Plan dict with intent, time_grain, compare, entity, measure, etc.
            Returns None if LLM unavailable or parsing fails.
        """
        if self.llm_auto_fallback_enabled or not self.use_llm:
            return None
        
        try:
            start_time = time.time()
            model_name = self._get_current_model()
            if not model_name:
                return None
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""Given Requirements JSON and the schema above, produce an execution PLAN JSON.

Rules:
- Output VALID JSON only (no prose).
- Never use DDL/DML/IO; read-only analytics only.
- Prefer time windows anchored to MAX(visit_date); last completed quarter = full quarter strictly before quarter(MAX).

PLAN schema:
{{
  "intent": "period_comparison|timeseries|ranking|share|constraint_timeseries|generic_agg",
  "time_grain": "day|month|quarter|year|null",
  "time_window": {{"mode":"explicit_range","explicit":{{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}}}}|{{"mode":"relative_to_dataset_max","range":"all"}}|null,
  "compare": {{"base":"last_completed_quarter","previous_by":1}}|null,
  "entity": "salesman_name|brand_name|customer_name|null",
  "measure": {{"name":"revenue","expr":"SUM(value)"}},
  "top_n": null|number,
  "constraints": [],
  "filters": [],
  "confidence": 0.0-1.0,
  "reason": "..."
}}

Return ONLY the JSON.

Mapping rules (structure-based, no keyword matching):
- If requirements.time.type == "comparison" → intent = "period_comparison", set time_grain to requirements.time.axis, and fill compare block.
- If requirements.time.type == "point" AND requirements.outputs.top_n is set → intent = "ranking", set time_grain to requirements.time.axis, set compare = null, set top_n from requirements.outputs.top_n.
- If group_by is set (e.g., ["brand_name"]) AND time.type="point" AND outputs.top_n is set → intent MUST be "ranking" (NOT "timeseries").
- If constraint "exactly one salesman active in month" → intent = "constraint_timeseries" with time_grain="month".
- Otherwise → intent = "generic_agg" or "timeseries".

CRITICAL: If group_by is set AND time.type="point" AND outputs.top_n is set → intent MUST be "ranking" (NOT "timeseries"). Do NOT output monthly totals for the whole year—only the top-N entities for that specific period.

Few-shot examples:

Requirements (positive):
{{
  "group_by": ["brand_name"],
  "metrics": [{{"name":"revenue","agg":"SUM"}}],
  "time": {{"type":"point","axis":"month","span":"explicit_range","explicit":{{"start":"2024-08-01","end":"2024-09-01"}}}},
  "outputs": {{"table":true,"delta":false,"pct_change":false,"top_n":2}},
  "presentation": {{"sort":["revenue:desc","brand_name:asc"]}}
}}

Plan (correct):
{{
  "intent": "ranking",
  "time_grain": "month",
  "time_window": {{"mode":"explicit_range","explicit":{{"start":"2024-08-01","end":"2024-09-01"}}}},
  "compare": null,
  "entity": "brand_name",
  "measure": {{"name":"revenue","expr":"SUM(value)"}},
  "top_n": 2,
  "constraints": [],
  "filters": [],
  "confidence": 0.95,
  "reason": "Top-2 ranking with share for specific month"
}}

Requirements (negative - DO NOT map to timeseries):
{{
  "group_by": ["brand_name"],
  "metrics": [{{"name":"revenue","agg":"SUM"}}],
  "time": {{"type":"point","axis":"month","span":"explicit_range","explicit":{{"start":"2024-08-01","end":"2024-09-01"}}}},
  "outputs": {{"table":true,"delta":false,"pct_change":false,"top_n":2}}
}}

Plan (WRONG - DO NOT USE):
{{
  "intent": "timeseries",  ← WRONG! Should be "ranking"
  ...
}}

Requirements JSON:
{json.dumps(requirements, indent=2, ensure_ascii=False)}

User question: {question}

Return only the JSON:"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group(0))
                    if isinstance(plan, dict) and 'intent' in plan:
                        self.llm_failure_count = 0
                        return plan
                except json.JSONDecodeError:
                    pass
            
            # Try parsing the whole response
            try:
                plan = json.loads(response_text)
                if isinstance(plan, dict) and 'intent' in plan:
                    self.llm_failure_count = 0
                    return plan
            except json.JSONDecodeError:
                pass
            
            # Check latency
            latency = time.time() - start_time
            if latency > self.LLM_LATENCY_THRESHOLD:
                self.llm_failure_count += 1
                if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                    self.llm_auto_fallback_enabled = True
                    st.session_state['llm_degraded'] = True
            
            return None
        except Exception as e:
            self.llm_failure_count += 1
            if self.llm_failure_count >= self.LLM_FAILURE_COUNT_THRESHOLD:
                self.llm_auto_fallback_enabled = True
                st.session_state['llm_degraded'] = True
            return None
    
    def _wants_ranking(self, requirements: dict) -> bool:
        """
        Check if requirements structure indicates a ranking query (pure structure-based, no keywords).
        
        Args:
            requirements: Requirements dict from _parse_requirements_via_llm
            
        Returns:
            True if requirements indicate ranking query
        """
        if not requirements:
            return False
        
        t = requirements.get("time", {})
        out = requirements.get("outputs", {})
        group_by = requirements.get("group_by") or []
        top_n = out.get("top_n")
        wants_top_n = top_n is not None and top_n > 0
        wants_share = bool(out.get("share"))
        axis = (t or {}).get("axis")
        time_type = (t or {}).get("type")
        
        return bool(
            group_by
            and time_type == "point"
            and axis in {"month", "quarter"}
            and (wants_top_n or wants_share)
        )
    
    def _enforce_plan(self, requirements: dict, plan: dict) -> dict:
        """
        Enforce plan consistency with requirements.
        - If requirements indicate ranking → force intent='ranking' and drop compare
        - If requirements indicate comparison (or delta/pct_change requested) → force intent='period_comparison'
        
        Call this right after validation/repair, before picking an executor.
        """
        if not requirements:
            return plan
        
        # 1) Ranking enforcement (structure-based)
        if self._wants_ranking(requirements):
            original_intent = plan.get('intent')
            # Use the requested axis if provided, default to month
            axis = (requirements.get('time') or {}).get('axis') or 'month'
            plan = {
                **plan,
                "intent": "ranking",
                "time_grain": axis,
                "compare": None
            }
            # Ensure top_n is set from requirements if available
            req_top_n = requirements.get('outputs', {}).get('top_n')
            if req_top_n is not None:
                plan['top_n'] = req_top_n
            # Ensure entity aligns with group_by if missing
            if not plan.get('entity'):
                gb = requirements.get('group_by') or []
                if gb:
                    plan['entity'] = gb[0]
            print(f"[SQLGenerationService] Enforced plan to 'ranking' (from {original_intent}): group_by={requirements.get('group_by')}, axis={axis}, top_n={plan.get('top_n')}")
        
        # 2) Period comparison enforcement (no keywords; driven by structure)
        time_req = requirements.get('time', {}) or {}
        outputs_req = requirements.get('outputs', {}) or {}
        wants_comparison = (
            time_req.get('type') == 'comparison' or
            bool(outputs_req.get('delta')) or
            bool(outputs_req.get('pct_change'))
        )
        if wants_comparison:
            original_intent = plan.get('intent')
            # Ensure compare block exists
            compare = plan.get('compare') or {"base": "last_completed_quarter", "previous_by": 1}
            # Choose time_grain from requirements axis if available
            tg = plan.get('time_grain') or time_req.get('axis') or 'quarter'
            # Ensure entity aligns with group_by when missing
            entity = plan.get('entity')
            if not entity:
                gb = requirements.get('group_by') or []
                entity = gb[0] if gb else entity
            # Ensure measure default exists
            measure = plan.get('measure') or {"name": "revenue", "expr": "SUM(value)"}
            plan = {
                **plan,
                "intent": "period_comparison",
                "time_grain": tg,
                "compare": compare,
                "measure": measure
            }
            if entity:
                plan['entity'] = entity
            print(f"[SQLGenerationService] Enforced plan to 'period_comparison' (from {original_intent}): axis={tg}, compare={plan.get('compare')}, entity={plan.get('entity')}")
        
        return plan
    
    def _check_consistency(self, requirements: dict, plan: dict) -> list[str]:
        """
        Check consistency between Requirements and Plan.
        
        Args:
            requirements: Requirements dict
            plan: Plan dict
            
        Returns:
            List of issue strings (empty if consistent)
        """
        issues = []
        
        # Check: comparison requested but plan.intent != period_comparison
        if requirements.get('time', {}).get('type') == 'comparison':
            if plan.get('intent') != 'period_comparison':
                issues.append("plan.intent must be period_comparison for a comparison request")
        
        # Check: ranking query (top-N with point time) but plan.intent != ranking
        time_type = requirements.get('time', {}).get('type')
        time_axis = requirements.get('time', {}).get('axis')
        outputs = requirements.get('outputs', {})
        group_by = requirements.get('group_by', [])
        
        # CRITICAL: Ranking gate - pure structure-based check
        # If group_by + point time (month/quarter) + top_n → MUST be ranking
        has_top_n = outputs.get('top_n') is not None and outputs.get('top_n') > 0
        
        if (time_type == 'point' and 
            time_axis in ['month', 'quarter', 'day', 'year'] and 
            group_by and 
            has_top_n):
            if plan.get('intent') != 'ranking':
                issues.append(f"plan.intent must be 'ranking' for group_by={group_by} with point time (axis={time_axis}) and top_n={has_top_n}, but got '{plan.get('intent')}'")
            if plan.get('compare'):
                issues.append("plan.compare must be null for ranking queries (no comparison)")
        
        # Check: delta/pct_change requested but plan.compare missing
        if (outputs.get('delta') or outputs.get('pct_change')):
            if not plan.get('compare'):
                issues.append("compare block missing while delta/pct_change requested")
        
        # Check: group_by entities not aligned with plan.entity
        if group_by:
            plan_entity = plan.get('entity')
            if plan_entity and plan_entity not in group_by:
                issues.append(f"plan.entity ({plan_entity}) not aligned with group_by ({group_by})")
        
        return issues
    
    def _repair_plan_via_llm(self, requirements: dict, plan: dict, issues: list[str]) -> Optional[dict]:
        """
        Repair PLAN JSON when inconsistencies are found.
        
        Args:
            requirements: Requirements dict
            plan: Original plan dict with issues
            issues: List of consistency issue strings
            
        Returns:
            Repaired plan dict or None if repair fails
        """
        if self.llm_auto_fallback_enabled or not self.use_llm:
            return None
        
        try:
            model_name = self._get_current_model()
            if not model_name:
                return None
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""Fix the PLAN JSON so it satisfies these constraints:

{chr(10).join(f'- {issue}' for issue in issues)}

Requirements JSON:
{json.dumps(requirements, indent=2, ensure_ascii=False)}

Current PLAN JSON (with issues):
{json.dumps(plan, indent=2, ensure_ascii=False)}

Return ONLY the fixed PLAN JSON (same schema as PLAN):"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    repaired_plan = json.loads(json_match.group(0))
                    if isinstance(repaired_plan, dict) and 'intent' in repaired_plan:
                        return repaired_plan
                except json.JSONDecodeError:
                    pass
            
            # Try parsing whole response
            try:
                repaired_plan = json.loads(response_text)
                if isinstance(repaired_plan, dict) and 'intent' in repaired_plan:
                    return repaired_plan
            except json.JSONDecodeError:
                pass
            
            return None
        except Exception:
            return None
    
    def derive_executor(self, plan: dict) -> str:
        """
        Derive executor name from plan content (no keywords, structure-based).
        Checks requirements first to preempt based on structure.
        
        Args:
            plan: Plan dict (must have _requirements stored)
            
        Returns:
            Executor name: "period_comparison", "constraint_timeseries", "timeseries", "ranking", "share", or "generic_agg"
        """
        requirements = plan.get('_requirements', {})
        intent = plan.get('intent', '')
        compare = plan.get('compare')
        constraints = plan.get('constraints', [])
        
        if intent == 'share':
            return 'share'
        
        # CRITICAL: Preempt based on requirements structure FIRST (before checking plan intent)
        # This ensures ranking queries are always routed correctly, even if plan has wrong intent
        if self._wants_ranking(requirements):
            original_intent = plan.get('_original_intent', intent)
            if original_intent != 'ranking':
                print(f"[SQLGenerationService] Preempted executor: {original_intent} → ranking (requirements indicate ranking)")
            else:
                print(f"[SQLGenerationService] Requirements indicate ranking, routing to 'ranking' executor")
            return 'ranking'
        
        # Preemption: if both comparison and constraint present → period_comparison wins
        has_constraint = any(c.get('type') == 'distinct_count' and c.get('eq') == 1 for c in constraints)
        if (intent == 'period_comparison' or compare) and has_constraint:
            return 'period_comparison'
        
        # Primary routing by intent
        if intent == 'period_comparison' or compare:
            return 'period_comparison'
        if intent == 'constraint_timeseries':
            return 'constraint_timeseries'
        if intent == 'ranking':
            return 'ranking'
        if intent == 'timeseries' and not compare:
            return 'timeseries'
        
        return 'generic_agg'
    
    def _resolve_time_window_from_plan(
        self,
        plan: dict,
        dataset_max_date: pd.Timestamp,
        default_months: int = 6
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
        """
        Resolve plan/requirements time window to start/end-exclusive bounds and label.
        
        Args:
            plan: Plan dictionary (may contain time_window + _requirements)
            dataset_max_date: Max date in dataset (used as anchor)
            default_months: Fallback month span when not provided
        
        Returns:
            Tuple of (start_date, end_date_exclusive, period_label)
        """
        requirements = plan.get('_requirements') or {}
        time_window = plan.get('time_window') or {}
        
        start_date = None
        end_date_exclusive = None
        label = None
        
        # 1) Explicit range handling (prefer plan-level, then requirements)
        explicit = None
        if isinstance(time_window, dict) and time_window.get('mode') == 'explicit_range':
            explicit = time_window.get('explicit')
        if explicit is None and isinstance(requirements, dict):
            time_req = requirements.get('time') or {}
            if isinstance(time_req.get('explicit'), dict):
                explicit = time_req.get('explicit')
            elif time_req.get('type') == 'point' and isinstance(time_req.get('explicit'), dict):
                explicit = time_req.get('explicit')
        
        if isinstance(explicit, dict):
            start_str = explicit.get('start')
            end_str = explicit.get('end')
            if start_str and end_str:
                start_date = pd.to_datetime(start_str)
                end_date_exclusive = pd.to_datetime(end_str)
        
        # 2) Derived ranges from structured spans
        if start_date is None or end_date_exclusive is None:
            time_req = requirements.get('time') or {}
            span = time_req.get('span')
            time_type = time_req.get('type')
            axis = time_req.get('axis')
            time_filter: Optional[TimeFilter] = None
            
            if span == 'last_completed_quarter':
                time_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
            elif span == 'last_6_calendar_months':
                time_filter = TimeFilter(mode=TimeMode.LAST_N_MONTHS, n_months=6)
            elif time_type == 'comparison':
                # Comparison requests rely on period comparison executor; fallback to last quarter span here
                time_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
            elif time_type == 'point' and axis == 'month':
                explicit = time_req.get('explicit')
                if isinstance(explicit, dict) and explicit.get('start') and explicit.get('end'):
                    start_date = pd.to_datetime(explicit['start'])
                    end_date_exclusive = pd.to_datetime(explicit['end'])
                else:
                    from dateutil.relativedelta import relativedelta
                    anchor_month = dataset_max_date.replace(day=1)
                    start_date = anchor_month
                    end_date_exclusive = anchor_month + relativedelta(months=1)
            elif span == 'explicit_range' and isinstance(time_req.get('explicit'), dict):
                exp = time_req.get('explicit')
                start_date = pd.to_datetime(exp.get('start')) if exp.get('start') else None
                end_date_exclusive = pd.to_datetime(exp.get('end')) if exp.get('end') else None
            else:
                time_filter = TimeFilter(mode=TimeMode.LAST_N_MONTHS, n_months=default_months)
            
            if (start_date is None or end_date_exclusive is None) and time_filter:
                start_date, end_date_exclusive, _ = TimeResolver.resolve_time_window(
                    time_filter,
                    dataset_max_date,
                    dataset_max_date
                )
        
        if start_date is None or end_date_exclusive is None:
            return None, None, None
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date_exclusive.strftime('%Y-%m-%d')
        
        end_minus_one = end_date_exclusive - pd.Timedelta(days=1)
        if end_minus_one.month == start_date.month and end_minus_one.year == start_date.year:
            label_prefix = start_date.strftime('%b-%Y')
        else:
            label_prefix = f"{start_date.strftime('%b-%Y')} → {end_minus_one.strftime('%b-%Y')}"
        
        label = f"{label_prefix} ({start_str} → {end_str})"
        return start_date, end_date_exclusive, label
    
    def _apply_ranking_period_label(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame, obs: Dict[str, Any]) -> None:
        """
        Populate period label metadata for ranking/share executors using resolved time window.
        """
        date_col = columns.get('date', 'VISITDATE')
        if date_col not in df.columns:
            return
        
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return
        
        dataset_max_ts = pd.Timestamp(dates.max())
        start_date, end_date_exclusive, label = self._resolve_time_window_from_plan(
            plan,
            dataset_max_ts,
            default_months=1
        )
        if not label:
            min_date = dates.min().date()
            max_date = dates.max().date()
            label = f"Dataset coverage: {min_date} → {max_date}"
        
        global_filter_enabled = st.session_state.get('global_date_filter_enabled', False)
        global_range = st.session_state.get('global_date_filter_range')
        if global_filter_enabled and global_range:
            obs["period_label"] = f"{label} | filtered: {global_range[0]} → {global_range[1]}"
        else:
            obs["period_label"] = label
        obs["time_window_label"] = obs.get("period_label")
    
    def _execute_period_comparison_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute period_comparison plan (quarter-over-quarter, month-over-month, etc.).
        Uses DuckDB with normalized view (sales_norm) for tight, read-only queries.
        Includes transaction counts and returns raw numbers (formatting in render layer).
        
        Args:
            plan: Plan dict with compare block
            columns: Column mapping dictionary
            df: DataFrame with sales data
            
        Returns:
            DataFrame with columns: entity, base_period_revenue, previous_period_revenue, base_transactions, 
            prev_transactions, base_aov, prev_aov, delta, pct_change
        """
        from .time_resolver import TimeResolver
        from .query_plan_service import TimeFilter, TimeMode
        from .comparison_service import ComparisonService
        
        date_col = columns.get('date', 'VISITDATE')
        value_col = columns.get('amount', 'VALUE')
        entity_col_name = plan.get('entity', 'salesman_name')
        
        # Get dataset max date
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return pd.DataFrame()
        dataset_max_date = dates.max().to_pydatetime()
        
        # Extract compare block
        compare = plan.get('compare', {})
        base = compare.get('base', 'last_completed_quarter')
        previous_by = compare.get('previous_by', 1)
        time_grain = plan.get('time_grain', 'quarter')
        
        # Resolve base period bounds
        if base == 'last_completed_quarter':
            base_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
            base_start, base_end_excl, base_label = TimeResolver.resolve_time_window(
                base_filter, dataset_max_date, dataset_max_date
            )
        else:
            # Fallback: use last_completed_quarter
            base_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
            base_start, base_end_excl, base_label = TimeResolver.resolve_time_window(
                base_filter, dataset_max_date, dataset_max_date
            )
        
        # Resolve previous period bounds (previous_by quarters/months before base)
        from dateutil.relativedelta import relativedelta
        if time_grain == 'quarter':
            prev_start = base_start - relativedelta(months=3 * previous_by)
            prev_end_excl = base_start
        elif time_grain == 'month':
            prev_start = base_start - relativedelta(months=previous_by)
            prev_end_excl = base_start
        else:
            # Default to quarter
            prev_start = base_start - relativedelta(months=3 * previous_by)
            prev_end_excl = base_start
        
        # Use DuckDB with normalized view (sales_norm)
        con = duckdb.connect()
        try:
            # Register DataFrame and create normalized view if needed
            con.register("sales", df)
            
            # Check if normalized view exists, create if not
            view_exists = False
            try:
                con.execute("SELECT 1 FROM sales_norm LIMIT 1")
                view_exists = True
            except:
                pass
            
            if not view_exists:
                self._create_normalized_view(con, df, columns)
            
            # Map entity name to normalized column name
            entity_col_norm = entity_col_name  # Already in snake_case format (salesman_name, brand_name, customer_name)
            
            # Build date filters (end-exclusive for base, end-inclusive for SQL BETWEEN)
            base_start_str = base_start.strftime('%Y-%m-%d')
            base_end_str = (base_end_excl - relativedelta(days=1)).strftime('%Y-%m-%d')  # Make inclusive for BETWEEN
            prev_start_str = prev_start.strftime('%Y-%m-%d')
            prev_end_str = (prev_end_excl - relativedelta(days=1)).strftime('%Y-%m-%d')  # Make inclusive for BETWEEN
            
            # Single query with LEFT JOIN to include all entities (even zero-revenue)
            # Uses normalized view and keeps formatting in render layer
            sql = f"""
            WITH all_entities AS (
                SELECT DISTINCT {entity_col_norm} AS entity
                FROM sales_norm
            ),
            base_period AS (
                SELECT
                    {entity_col_norm} AS entity,
                    SUM(value) AS revenue_raw,
                    COUNT(*) AS transactions
                FROM sales_norm
                WHERE visit_date BETWEEN DATE '{base_start_str}' AND DATE '{base_end_str}'
                GROUP BY {entity_col_norm}
            ),
            prev_period AS (
                SELECT
                    {entity_col_norm} AS entity,
                    SUM(value) AS revenue_raw,
                    COUNT(*) AS transactions
                FROM sales_norm
                WHERE visit_date BETWEEN DATE '{prev_start_str}' AND DATE '{prev_end_str}'
                GROUP BY {entity_col_norm}
            )
            SELECT
                a.entity,
                COALESCE(b.revenue_raw, 0) AS base_period_revenue,
                COALESCE(b.transactions, 0) AS base_transactions,
                COALESCE(p.revenue_raw, 0) AS previous_period_revenue,
                COALESCE(p.transactions, 0) AS prev_transactions
            FROM all_entities a
            LEFT JOIN base_period b USING (entity)
            LEFT JOIN prev_period p USING (entity)
            ORDER BY base_period_revenue DESC, entity ASC
            """
            
            # Execute single query
            result = con.execute(sql).fetchdf()
            
            # Calculate delta and percentage change (raw numbers, formatting in render layer)
            result['delta'] = result['base_period_revenue'] - result['previous_period_revenue']
            result['pct_change'] = result.apply(
                lambda row: ComparisonService.calculate_percentage_change(
                    row['base_period_revenue'], row['previous_period_revenue'], format_result=True
                )[0],
                axis=1
            )
            
            # Apply sorting if specified in plan (override SQL ORDER BY if needed)
            presentation = plan.get('presentation', {})
            sort_fields = presentation.get('sort', [])
            if sort_fields:
                # Parse sort fields (e.g., "revenue:desc", "salesman_name:asc")
                sort_cols = []
                sort_asc = []
                for field in sort_fields:
                    if ':' in field:
                        col_name, direction = field.split(':')
                        if col_name == 'revenue' or col_name == 'metric':
                            sort_cols.append('base_period_revenue')
                            sort_asc.append(direction.lower() == 'asc')
                        elif col_name == 'entity' or col_name == 'group_by_field':
                            sort_cols.append('entity')
                            sort_asc.append(direction.lower() == 'asc')
                if sort_cols:
                    result = result.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
            # Otherwise, SQL ORDER BY already applied
            
            if 'entity' in result.columns:
                result['entity'] = result['entity'].apply(
                    EntityNormalizationService.normalize_entity_name
                )
            
            return result
        finally:
            con.close()
    
    def _execute_ranking_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute ranking plan (top-N entities with share calculation).
        Uses DuckDB with normalized view (sales_norm) for tight, read-only queries.
        
        Args:
            plan: Plan dict with ranking intent
            columns: Column mapping dictionary
            df: DataFrame with sales data
            
        Returns:
            DataFrame with columns: entity, revenue, pct_share (and transaction count if requested)
        """
        from .time_resolver import TimeResolver
        from .query_plan_service import TimeFilter, TimeMode
        
        date_col = columns.get('date', 'VISITDATE')
        value_col = columns.get('amount', 'VALUE')
        entity_col_name = plan.get('entity', 'salesman_name')
        
        # Get dataset max date
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return pd.DataFrame()
        dataset_max_date = dates.max().to_pydatetime()
        
        # Extract time window from plan or requirements
        time_grain = plan.get('time_grain', 'month')
        time_window = plan.get('time_window', {})
        
        # If time_window not in plan, try to extract from requirements (stored in plan)
        if not time_window or not time_window.get('mode'):
            requirements = plan.get('_requirements')
            if requirements and requirements.get('time'):
                time_req = requirements['time']
                if time_req.get('span') == 'explicit_range' and time_req.get('explicit'):
                    time_window = {
                        'mode': 'explicit_range',
                        'explicit': time_req['explicit']
                    }
                elif time_req.get('type') == 'point' and time_req.get('axis') == 'month':
                    # Extract month and year from question if not explicitly provided
                    # This handles "August 2024" type queries
                    import re
                    from datetime import datetime
                    month_names = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                    }
                    # Try to extract from question stored in requirements or plan
                    question_text = str(plan.get('_question', ''))
                    if not question_text:
                        question_text = str(requirements.get('_question', ''))
                    
                    year_match = re.search(r'20\d{2}', question_text)
                    target_year = int(year_match.group(0)) if year_match else dataset_max_date.year
                    
                    month_num = None
                    for month_name, num in month_names.items():
                        if month_name in question_text.lower():
                            month_num = num
                            break
                    
                    if month_num:
                        start_date = datetime(target_year, month_num, 1)
                        if month_num == 12:
                            end_date = datetime(target_year + 1, 1, 1)
                        else:
                            end_date = datetime(target_year, month_num + 1, 1)
                        
                        time_window = {
                            'mode': 'explicit_range',
                            'explicit': {
                                'start': start_date.strftime('%Y-%m-%d'),
                                'end': end_date.strftime('%Y-%m-%d')
                            }
                        }
        
        start_date_str = None
        end_date_str = None
        if time_window.get('mode') == 'explicit_range':
            explicit = time_window.get('explicit', {})
            start_str = explicit.get('start')
            end_str = explicit.get('end')
            if start_str and end_str:
                start_date = pd.to_datetime(start_str)
                end_date = pd.to_datetime(end_str)
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
        elif time_window.get('mode') == 'relative_to_dataset_max':
            range_mode = time_window.get('range')
            if range_mode == 'all':
                start_date = dates.min().normalize()
                end_date = dates.max().normalize() + pd.Timedelta(days=1)
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                time_window = {
                    'mode': 'explicit_range',
                    'explicit': {'start': start_date_str, 'end': end_date_str}
                }
            elif range_mode == 'last_completed_quarter':
                time_filter = TimeFilter(mode=TimeMode.LAST_COMPLETED_QUARTER)
                start_date, end_date_excl, _ = TimeResolver.resolve_time_window(
                    time_filter, dataset_max_date, dataset_max_date
                )
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date_excl.strftime('%Y-%m-%d')
                time_window = {
                    'mode': 'explicit_range',
                    'explicit': {'start': start_date_str, 'end': end_date_str}
                }
        
        if not start_date_str or not end_date_str:
            time_filter = TimeFilter(mode=TimeMode.LAST_MONTH)
            start_date, end_date_excl, _ = TimeResolver.resolve_time_window(
                time_filter, dataset_max_date, dataset_max_date
            )
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date_excl.strftime('%Y-%m-%d')
            # Update time_window to explicit range for downstream usage
            time_window = {
                'mode': 'explicit_range',
                'explicit': {'start': start_date_str, 'end': end_date_str}
            }
        
        # Get top_n from plan or requirements
        top_n = plan.get('top_n') or plan.get('limit') or 10  # Default to 10
        
        start_date_obj = pd.to_datetime(start_date_str)
        end_date_obj = pd.to_datetime(end_date_str)
        single_month_window = (
            time_grain == 'month'
            and start_date_obj.day == 1
            and end_date_obj == (start_date_obj + pd.offsets.MonthBegin(1))
        )
        
        # Determine if share is requested
        requirements = plan.get('_requirements', {})
        outputs = requirements.get('outputs', {}) if isinstance(requirements, dict) else {}
        include_share = bool(outputs.get('share'))
        if plan.get('intent') == 'share':
            include_share = True
        
        # Use DuckDB with normalized view
        con = duckdb.connect()
        try:
            con.register("sales", df)
            view_exists = False
            try:
                con.execute("SELECT 1 FROM sales_norm LIMIT 1")
                view_exists = True
            except Exception:
                pass
            if not view_exists:
                self._create_normalized_view(con, df, columns)
            entity_col_norm = entity_col_name
            
            if single_month_window:
                sql = self._sql_top_n_group_for_month(
                    entity_col_norm,
                    start_date_obj.year,
                    start_date_obj.month,
                    top_n,
                    include_share
                )
                result = con.execute(sql).fetchdf()
                month_total_query = f"""
                SELECT SUM(value) AS total
                FROM sales_norm
                WHERE visit_date >= DATE '{start_date_obj.year:04d}-{start_date_obj.month:02d}-01'
                  AND visit_date < (DATE '{start_date_obj.year:04d}-{start_date_obj.month:02d}-01' + INTERVAL 1 MONTH)
                """
                month_total_result = con.execute(month_total_query).fetchdf()
                if not month_total_result.empty and 'total' in month_total_result.columns:
                    result.attrs['month_total'] = float(month_total_result['total'].iloc[0])
            else:
                limit_clause = f"LIMIT {int(top_n)}" if top_n else ""
                if include_share:
                    sql = f"""
WITH params AS (
  SELECT DATE '{start_date_str}'::DATE AS start_date,
         DATE '{end_date_str}'::DATE AS end_date
),
entities AS (
  SELECT DISTINCT {entity_col_norm} AS entity
  FROM sales_norm
),
revenues AS (
  SELECT {entity_col_norm} AS entity,
         SUM(value) AS revenue
  FROM sales_norm, params
  WHERE visit_date >= params.start_date
    AND visit_date < params.end_date
  GROUP BY {entity_col_norm}
),
joined AS (
  SELECT e.entity, COALESCE(r.revenue, 0) AS revenue
  FROM entities e
  LEFT JOIN revenues r USING (entity)
),
totals AS (
  SELECT SUM(revenue) AS total_revenue FROM joined
)
SELECT
  entity AS {entity_col_norm},
  ROUND(revenue, 2) AS revenue,
  ROUND(revenue / NULLIF(t.total_revenue, 0) * 100, 2) AS pct_share
FROM joined
CROSS JOIN totals t
ORDER BY revenue DESC, {entity_col_norm} ASC
{limit_clause}
"""
                else:
                    sql = f"""
WITH params AS (
  SELECT DATE '{start_date_str}'::DATE AS start_date,
         DATE '{end_date_str}'::DATE AS end_date
),
entities AS (
  SELECT DISTINCT {entity_col_norm} AS entity
  FROM sales_norm
)
SELECT
  e.entity AS {entity_col_norm},
  ROUND(COALESCE(r.revenue, 0), 2) AS revenue
FROM entities e
LEFT JOIN (
  SELECT {entity_col_norm} AS entity,
         SUM(value) AS revenue
  FROM sales_norm, params
  WHERE visit_date >= params.start_date
    AND visit_date < params.end_date
  GROUP BY {entity_col_norm}
) r USING (entity)
ORDER BY revenue DESC, {entity_col_norm} ASC
{limit_clause}
"""
                result = con.execute(sql).fetchdf()
            
            if include_share and 'pct_share' in result.columns:
                result['pct_share'] = result['pct_share'].fillna(0.0)
            if entity_col_norm in result.columns:
                result[entity_col_norm] = result[entity_col_norm].apply(
                    EntityNormalizationService.normalize_entity_name
                )
            return result
        finally:
            con.close()
    
    def _execute_timeseries_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute timeseries plan using DuckDB (read-only) and include zero-fill reindexing.
        Highlights peak period via DataFrame attrs for downstream formatting.
        """
        date_col = columns.get('date', 'VISITDATE')
        value_col = columns.get('amount', 'VALUE')
        
        if date_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return pd.DataFrame()
        
        dataset_max_ts = pd.Timestamp(dates.max())
        start_date, end_date_exclusive, label = self._resolve_time_window_from_plan(
            plan,
            dataset_max_ts,
            default_months=6
        )
        if start_date is None or end_date_exclusive is None:
            time_filter = TimeFilter(mode=TimeMode.LAST_N_MONTHS, n_months=6)
            start_date, end_date_exclusive, label = TimeResolver.resolve_time_window(
                time_filter,
                dataset_max_ts,
                dataset_max_ts
            )
            start_date = pd.Timestamp(start_date)
            end_date_exclusive = pd.Timestamp(end_date_exclusive)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date_exclusive.strftime('%Y-%m-%d')
        
        time_grain = (plan.get('time_grain') or 'month').lower()
        grain_expr_map = {
            'day': "DATE(visit_date)",
            'month': "DATE_TRUNC('month', visit_date)",
            'quarter': "DATE_TRUNC('quarter', visit_date)",
            'year': "DATE_TRUNC('year', visit_date)"
        }
        grain_expr = grain_expr_map.get(time_grain, "DATE_TRUNC('month', visit_date)")
        period_col_map = {
            'day': 'day',
            'month': 'month',
            'quarter': 'quarter',
            'year': 'year'
        }
        period_col = period_col_map.get(time_grain, 'month')
        measure_name = (plan.get('measure') or {}).get('name') or 'revenue'
        measure_alias = re.sub(r'[^a-zA-Z0-9_]', '_', measure_name.lower())
        if not measure_alias:
            measure_alias = 'metric'
        
        con = duckdb.connect()
        try:
            con.register("sales", df)
            view_exists = False
            try:
                con.execute("SELECT 1 FROM sales_norm LIMIT 1")
                view_exists = True
            except Exception:
                pass
            if not view_exists:
                self._create_normalized_view(con, df, columns)
            
            sql = f"""
            SELECT
              {grain_expr} AS period_start,
              SUM(value) AS {measure_alias}
            FROM sales_norm
            WHERE visit_date >= DATE '{start_str}'
              AND visit_date < DATE '{end_str}'
            GROUP BY 1
            ORDER BY 1
            """
            result = con.execute(sql).fetchdf()
        finally:
            con.close()
        
        if result.empty:
            # Build empty frame with expected columns for downstream formatting
            return pd.DataFrame(columns=[period_col, measure_name, 'is_peak'])
        
        result['period_start'] = pd.to_datetime(result['period_start'], errors='coerce')
        if measure_alias != measure_name and measure_alias in result.columns:
            result = result.rename(columns={measure_alias: measure_name})
        elif measure_alias not in result.columns:
            result[measure_name] = 0.0
        freq_map = {
            'day': 'D',
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }
        freq = freq_map.get(time_grain, 'M')
        end_inclusive = end_date_exclusive - pd.Timedelta(days=1)
        
        period_index = pd.period_range(
            start=start_date.to_period(freq),
            end=end_inclusive.to_period(freq),
            freq=freq
        )
        if len(period_index) == 0:
            empty_df = pd.DataFrame(columns=[period_col, measure_name, 'is_peak'])
            empty_df.attrs['period_label'] = label
            empty_df.attrs['start_date'] = start_str
            empty_df.attrs['end_date_exclusive'] = end_str
            empty_df.attrs['time_grain'] = time_grain
            return empty_df
        result['__period__'] = result['period_start'].dt.to_period(freq)
        result = result.set_index('__period__').reindex(period_index, fill_value=0.0).reset_index()
        result = result.rename(columns={'__period__': period_col})
        result[period_col] = result[period_col].astype(str)
        result = result.drop(columns=['period_start'], errors='ignore')
        
        # Ensure numeric column exists even after fill_value
        if measure_name not in result.columns:
            result[measure_name] = 0.0
        result[measure_name] = pd.to_numeric(result[measure_name], errors='coerce').fillna(0.0)
        
        # Highlight peak row
        peak_idx = result[measure_name].idxmax() if not result.empty else None
        result['is_peak'] = False
        if peak_idx is not None and pd.notna(peak_idx):
            result.loc[peak_idx, 'is_peak'] = True
            result.attrs['peak_period'] = result.loc[peak_idx, period_col]
            result.attrs['peak_value'] = float(result.loc[peak_idx, measure_name])
        
        result.attrs['period_label'] = label
        result.attrs['start_date'] = start_str
        result.attrs['end_date_exclusive'] = end_str
        result.attrs['time_grain'] = time_grain
        
        if plan.get('_negative_only'):
            result = result[result[measure_name] < 0].reset_index(drop=True)
        
        return result
    
    def _execute_generic_agg_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute generic aggregate plan (sum/avg over optional groupings) deterministically.
        """
        date_col = columns.get('date', 'VISITDATE')
        value_col = columns.get('amount', 'VALUE')
        
        if date_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return pd.DataFrame()
        
        dataset_max_ts = pd.Timestamp(dates.max())
        start_date, end_date_exclusive, label = self._resolve_time_window_from_plan(
            plan,
            dataset_max_ts,
            default_months=6
        )
        if start_date is None or end_date_exclusive is None:
            time_filter = TimeFilter(mode=TimeMode.LAST_N_MONTHS, n_months=6)
            start_date, end_date_exclusive, label = TimeResolver.resolve_time_window(
                time_filter,
                dataset_max_ts,
                dataset_max_ts
            )
            start_date = pd.Timestamp(start_date)
            end_date_exclusive = pd.Timestamp(end_date_exclusive)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date_exclusive.strftime('%Y-%m-%d')
        
        requirements = plan.get('_requirements') or {}
        group_by_fields = []
        if plan.get('entity'):
            group_by_fields.append(plan['entity'])
        elif isinstance(requirements.get('group_by'), list):
            group_by_fields.extend([g for g in requirements['group_by'] if g])
        
        # Deduplicate while preserving order
        seen = set()
        group_by_fields = [g for g in group_by_fields if not (g in seen or seen.add(g))]
        
        measure_name = (plan.get('measure') or {}).get('name') or 'revenue'
        measure_alias = re.sub(r'[^a-zA-Z0-9_]', '_', measure_name.lower())
        if not measure_alias:
            measure_alias = 'metric'
        
        include_transactions = bool(plan.get('_include_transactions')) or bool(group_by_fields)
        top_n = plan.get('top_n') or plan.get('limit')
        
        con = duckdb.connect()
        try:
            con.register("sales", df)
            view_exists = False
            try:
                con.execute("SELECT 1 FROM sales_norm LIMIT 1")
                view_exists = True
            except Exception:
                pass
            if not view_exists:
                self._create_normalized_view(con, df, columns)
            
            select_parts = []
            if group_by_fields:
                select_parts.extend(group_by_fields)
            select_parts.append(f"SUM(value) AS {measure_alias}")
            if include_transactions:
                select_parts.append("COUNT(*) AS transactions")
            select_clause = ", ".join(select_parts)
            
            group_clause = ""
            order_clause = ""
            if group_by_fields:
                group_cols = ", ".join(group_by_fields)
                group_clause = f"GROUP BY {group_cols}"
                primary_sort = f"{measure_alias} DESC"
                secondary_sort = ", ".join(f"{col} ASC" for col in group_by_fields)
                order_terms = [primary_sort]
                if secondary_sort:
                    order_terms.append(secondary_sort)
                order_clause = f"ORDER BY {', '.join(order_terms)}"
            else:
                if top_n:
                    order_clause = f"ORDER BY {measure_alias} DESC"
                else:
                    order_clause = ""
            
            limit_clause = ""
            if top_n and isinstance(top_n, int) and top_n > 0:
                limit_clause = f"LIMIT {int(top_n)}"
            
            sql = f"""
            SELECT
              {select_clause}
            FROM sales_norm
            WHERE visit_date >= DATE '{start_str}'
              AND visit_date < DATE '{end_str}'
            {group_clause}
            {order_clause}
            {limit_clause}
            """
            result = con.execute(sql).fetchdf()
        finally:
            con.close()
        
        if measure_alias != measure_name and measure_alias in result.columns:
            result = result.rename(columns={measure_alias: measure_name})
        elif measure_alias not in result.columns:
            result[measure_name] = 0.0
        
        for field in group_by_fields:
            if field in result.columns:
                if pd.api.types.is_object_dtype(result[field]):
                    result[field] = result[field].apply(
                        lambda x: EntityNormalizationService.normalize_entity_name(x) if isinstance(x, str) else x
                    )
        
        if 'visit_date' in result.columns:
            try:
                result['visit_date'] = pd.to_datetime(result['visit_date']).dt.date.astype(str)
            except Exception:
                pass
        
        result.attrs['period_label'] = label
        result.attrs['start_date'] = start_str
        result.attrs['end_date_exclusive'] = end_str
        return result
    
    def _sql_top_n_group_for_month(self, group_col: str, year: int, month: int, n: int = 2, include_share: bool = True) -> str:
        """
        Generate deterministic SQL for top-N ranking within a specific month.
        
        Args:
            group_col: Column name to group by (e.g., "brand_name")
            year: Year (e.g., 2024)
            month: Month (1-12)
            n: Number of top results (default: 2)
            include_share: Whether to include percentage share calculation
            
        Returns:
            SQL query string
        """
        # Start inclusive, end exclusive (using INTERVAL for month boundaries)
        share_clause = ', ROUND(revenue / NULLIF(mt.total, 0) * 100, 2) AS pct_share' if include_share else ''
        
        sql = f"""
WITH month_total AS (
  SELECT SUM(value) AS total
  FROM sales_norm
  WHERE visit_date >= DATE '{year:04d}-{month:02d}-01'
    AND visit_date < (DATE '{year:04d}-{month:02d}-01' + INTERVAL 1 MONTH)
),
group_rev AS (
  SELECT {group_col} AS grp, SUM(value) AS revenue
  FROM sales_norm
  WHERE visit_date >= DATE '{year:04d}-{month:02d}-01'
    AND visit_date < (DATE '{year:04d}-{month:02d}-01' + INTERVAL 1 MONTH)
  GROUP BY 1
)
SELECT
  grp AS {group_col},
  ROUND(revenue, 2) AS revenue{share_clause}
FROM group_rev gr
CROSS JOIN month_total mt
ORDER BY revenue DESC, {group_col} ASC
LIMIT {int(n)}
"""
        return sql.strip()
    
    def _validate_query_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """
        Validate query plan before execution.
        
        Args:
            plan: Query plan dict
            columns: Column mapping dictionary
            df: DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        intent = plan.get('intent', '')
        
        # Check required columns exist (date and amount are always required)
        date_col = columns.get('date')
        value_col = columns.get('amount')
        
        if not date_col or not value_col:
            missing = [k for k, v in [('date', date_col), ('amount', value_col)] if not v]
            return False, f"Required columns missing: {', '.join(missing)}"
        
        # Check columns exist in DataFrame
        if date_col not in df.columns or value_col not in df.columns:
            missing = [c for c in [date_col, value_col] if c not in df.columns]
            return False, f"Columns not found in data: {', '.join(missing)}"
        
        # Validate ranking intent - check for brand_name column
        if intent == 'ranking':
            entity = plan.get('entity')
            if entity == 'brand_name':
                brand_col = columns.get('brand')
                # Check if brand column is mapped
                if not brand_col:
                    # Try to infer from common column names
                    possible_brand_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['brand', 'product', 'item'])]
                    if not possible_brand_cols:
                        return False, 'Cannot run: required column "brand_name" is missing. Please map the brand column in column mapping.'
                    # Use the first possible brand column as fallback
                    brand_col = possible_brand_cols[0]
                    print(f"[SQLGenerationService] Inferred brand column: {brand_col}")
                
                # Check if column exists in DataFrame
                if brand_col not in df.columns:
                    return False, f'Cannot run: required column "brand_name" is missing. Column "{brand_col}" not found in data.'
            
            # Must have entity
            if not entity:
                return False, "ranking intent requires entity field"
        
        # Validate period_comparison intent
        if intent == 'period_comparison':
            # Must have compare block
            compare = plan.get('compare')
            if not compare:
                return False, "period_comparison intent requires compare block"
            
            # Must have valid time_grain
            time_grain = plan.get('time_grain', '').lower()
            valid_time_grains = ['month', 'quarter', 'day', 'year']
            if time_grain and time_grain not in valid_time_grains:
                return False, f"Invalid time grain: {time_grain}. Must be one of {valid_time_grains}"
            
            # Must have entity
            if not plan.get('entity'):
                return False, "period_comparison intent requires entity field"
            
            # Must have measure
            if not plan.get('measure'):
                return False, "period_comparison intent requires measure field"
            
            return True, None
        
        # Validate constraint_timeseries intent
        if intent == 'constraint_timeseries':
            salesman_col = columns.get('salesman')
            if not salesman_col:
                return False, "constraint_timeseries intent requires salesman column"
            
            if salesman_col not in df.columns:
                return False, f"Salesman column not found in data: {salesman_col}"
            
            # Validate time grain
            time_grain = plan.get('time_grain', '').lower()
            valid_time_grains = ['month', 'quarter', 'day', 'year']
            if time_grain not in valid_time_grains:
                return False, f"Invalid time grain: {time_grain}. Must be one of {valid_time_grains}"
            
            # Validate constraints
            constraints = plan.get('constraints', [])
            if not constraints:
                return False, "constraint_timeseries intent requires constraints"
            
            # Check if constraint makes sense (distinct_count eq 1)
            for constraint in constraints:
                if constraint.get('type') == 'distinct_count' and constraint.get('eq') == 1:
                    # Valid constraint
                    pass
                else:
                    return False, f"Unsupported constraint type: {constraint.get('type')}"
            
            return True, None
        
        # For other intents, basic validation
        if plan.get('entity') and plan.get('measure'):
            return True, None
        
        return False, f"Plan validation failed: missing required fields for intent {intent}"
    
    def _detect_deterministic_pattern(self, question: str, columns: Dict[str, str], df: pd.DataFrame) -> Optional[dict]:
        """
        Detect and validate query pattern using two-step LLM parsing (Requirements → Plan) + heuristic fallback.
        Returns validated query plan or None.
        
        Args:
            question: User's question
            columns: Column mapping dictionary
            df: DataFrame for validation
            
        Returns:
            Query plan dict if detected and valid, None otherwise
        """
        requirements = None
        plan = None
        confidence = 0.0
        repairs_applied = 0
        
        # Step 1: Parse requirements (WHAT the user wants) - LLM first, no hardcoding
        if self.use_llm and not self.llm_auto_fallback_enabled:
            requirements = self._parse_requirements_via_llm(question)
            if requirements:
                print(f"[SQLGenerationService] Requirements parsed: group_by={requirements.get('group_by')}, time.type={requirements.get('time', {}).get('type')}, top_n={requirements.get('outputs', {}).get('top_n')}")
        
        # Step 2: Parse plan (HOW to compute it) - LLM first, no hardcoding
        if requirements and self.use_llm and not self.llm_auto_fallback_enabled:
            plan = self._parse_plan_via_llm(requirements, question)
            if plan:
                confidence = plan.get('confidence', 0.0)
                original_intent = plan.get('intent')  # Track original intent from LLM
                print(f"[SQLGenerationService] Plan parsed: intent={original_intent}, entity={plan.get('entity')}, top_n={plan.get('top_n')}, confidence={confidence:.2f}")
                
                # Extract top_n from requirements if not in plan
                if 'top_n' not in plan and requirements.get('outputs', {}).get('top_n'):
                    plan['top_n'] = requirements['outputs']['top_n']
                
                # Smart defaults for ranking-style questions when LLM omits top_n
                if requirements:
                    outputs = requirements.setdefault('outputs', {})
                    if outputs.get('top_n') is None:
                        question_lower = (question or "").lower()
                        has_explicit_top = re.search(r'\btop\s+\d+', question_lower)
                        mentions_top = re.search(r'\btop\b', question_lower)
                        if (
                            any(phrase in question_lower for phrase in ['top performers', 'top ', 'best ', 'leading ', 'highest '])
                            or (mentions_top and not has_explicit_top)
                        ):
                            outputs['top_n'] = 10
                            plan.setdefault('top_n', 10)
                
                # Step 3: Check consistency between requirements and plan
                issues = self._check_consistency(requirements, plan)
                if issues:
                    # Try to repair plan via LLM
                    repaired_plan = self._repair_plan_via_llm(requirements, plan, issues)
                    if repaired_plan:
                        plan = repaired_plan
                        repairs_applied = 1
                        confidence = plan.get('confidence', confidence)
                        # Ensure top_n is still in repaired plan
                        if 'top_n' not in plan and requirements.get('outputs', {}).get('top_n'):
                            plan['top_n'] = requirements['outputs']['top_n']
                
                # Step 4: CRITICAL - Enforce plan consistency AFTER validation/repair, BEFORE executor selection
                # This ensures ranking queries are forced even if LLM returned wrong intent
                plan_before_enforcement = plan.copy()
                plan = self._enforce_plan(requirements, plan)
                if plan.get('intent') != plan_before_enforcement.get('intent'):
                    print(f"[SQLGenerationService] Plan enforced: intent changed from '{plan_before_enforcement.get('intent')}' to '{plan.get('intent')}' (requirements indicate ranking)")
                plan['_original_intent'] = original_intent  # Track original intent for logging
                
                # Diagnostic logging for Plan (after enforcement)
                print(f"[SQLGenerationService] Plan (after enforce_plan): {plan}")
        
        # If we have a plan with sufficient confidence, validate and return it
        if plan and confidence >= 0.6:
            print(f"[SQLGenerationService] Validating plan with confidence {confidence:.2f}")
            is_valid, error_msg = self._validate_query_plan(plan, columns, df)
            if is_valid:
                print(f"[SQLGenerationService] Plan validation passed")
            else:
                print(f"[SQLGenerationService] Plan validation failed: {error_msg}")
            if is_valid:
                # Store requirements and repairs in plan for observability
                plan['_requirements'] = requirements
                plan['_repairs_applied'] = repairs_applied
                plan['_question'] = question  # Store original question for date extraction
                plan['_original_intent'] = plan.get('intent')  # Track original intent before enforcement
                
                # CRITICAL: Final enforcement before returning (after validation)
                plan = self._enforce_plan(requirements, plan)
                if plan.get('intent') != plan.get('_original_intent'):
                    print(f"[SQLGenerationService] Final plan enforcement: intent changed from {plan.get('_original_intent')} to {plan.get('intent')}")
                
                return plan
        elif plan:
            print(f"[SQLGenerationService] Plan confidence below acceptance threshold ({confidence:.2f}); using fallback path.")
        
        # Heuristic fallbacks (ONLY when LLM is unavailable/degraded or confidence too low)
        # These are true fallbacks, not preemptive hardcoding
        if (not plan) or (confidence < 0.6) or self.llm_auto_fallback_enabled or not self.use_llm:
            heuristic_candidates = [
                self._heuristic_detect_month_ranking(question, columns, df),
                self._heuristic_salesman_range_totals(question, columns, df),
                self._heuristic_last_completed_quarter_comparison(question, columns, df),
                self._heuristic_highest_revenue_day(question, columns, df),
                self._heuristic_top_performers(question, columns, df),
            ]
            for h_plan in heuristic_candidates:
                if not h_plan:
                    continue
                is_valid, error_msg = self._validate_query_plan(h_plan, columns, df)
                if is_valid:
                    enforced = self._enforce_plan(h_plan.get('_requirements'), h_plan)
                    return enforced
        
        # Fallback to heuristic for constraint_timeseries queries only
        heuristic_match = self._heuristic_detect_single_salesman_month(question)
        if heuristic_match:
            # Build minimal plan from heuristic
            plan = {
                "intent": "constraint_timeseries",
                "time_grain": "month",
                "entity": "salesman_name",
                "measure": {"name": "revenue", "expr": "SUM(value)"},
                "constraints": [{
                    "type": "distinct_count",
                    "of": "salesman_name",
                    "where": "SUM(value)>0",
                    "eq": 1
                }],
                "filters": [],
                "time_window": {"mode": "relative_to_dataset_max", "range": "all"},
                "confidence": 0.5,  # Lower confidence for heuristic
                "reason": "Heuristic fallback detected"
            }
            is_valid, error_msg = self._validate_query_plan(plan, columns, df)
            if is_valid:
                plan['_requirements'] = None
                plan['_repairs_applied'] = 0
                return plan
        
        return None
    
    def _execute_constraint_timeseries_plan(self, plan: dict, columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute constraint_timeseries query plan deterministically.
        Robust to messy numbers, defines revenue positivity clearly.
        
        Args:
            plan: Query plan dict
            columns: Column mapping dictionary
            df: DataFrame
            
        Returns:
            DataFrame with results (month, salesman, revenue)
        """
        date_col = columns.get('date', 'VISITDATE')
        value_col = columns.get('amount', 'VALUE')
        salesman_col = columns.get('salesman', 'Salesman Name')
        
        # Use local computation (robust to messy numbers)
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        
        # Clean numeric values: remove currency marks, commas, non-numeric
        ser = d[value_col].astype(str)
        ser = ser.str.replace(r"[^0-9\-\.\,]+", "", regex=True).str.replace(",", "", regex=False)
        d["_amt_"] = pd.to_numeric(ser, errors="coerce").fillna(0.0)
        
        # Create month period
        d["_ym_"] = d[date_col].dt.to_period("M").astype(str)
        
        # Group by month and salesman (dropna=False to handle nulls explicitly)
        per_rep = d.groupby(["_ym_", salesman_col], dropna=False)["_amt_"].sum().reset_index(name="revenue")
        
        # "Active" means net revenue > 0 (exclude refunds/negatives)
        active = (per_rep[per_rep["revenue"] > 0]
                 .groupby("_ym_")[salesman_col]
                 .nunique()
                 .reset_index(name="active_reps"))
        
        # Find months where exactly one salesman is active
        only_one = active[active["active_reps"] == 1]["_ym_"]
        
        # Filter to months with exactly one salesman (revenue > 0)
        # Tie rule: sort by revenue DESC, then salesman name ASC (stable output)
        out = (per_rep[(per_rep["_ym_"].isin(only_one)) & (per_rep["revenue"] > 0)]
               .sort_values(["_ym_", "revenue", salesman_col], ascending=[True, False, True])
               .rename(columns={"_ym_": "month", salesman_col: "salesman"}))
        
        if not out.empty and "salesman" in out.columns:
            out["salesman"] = out["salesman"].apply(
                EntityNormalizationService.normalize_entity_name
            )
        
        return out[["month", "salesman", "revenue"]] if not out.empty else pd.DataFrame()
    
    def _try_local_fallback(self, question: str, df: pd.DataFrame, columns: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Try local computation fallback when SQL returns 0 rows.
        
        Args:
            question: User's question
            df: DataFrame with sales data
            columns: Column mapping dictionary
            
        Returns:
            DataFrame with results if computation succeeds, None otherwise
        """
        # Use LLM/heuristic to understand intent
        plan = self._detect_deterministic_pattern(question, columns, df)
        if plan and plan.get('intent') == 'constraint_timeseries':
            return self._execute_constraint_timeseries_plan(plan, columns, df)
        
        return None
    
    def _local_fallback_months_single_salesman(
        self, 
        df: pd.DataFrame, 
        columns: Dict[str, str],
        question: str
    ) -> Optional[pd.DataFrame]:
        """
        Local computation: find months where exactly one salesman has revenue.
        Exact implementation matching user's specification.
        
        Args:
            df: DataFrame with sales data
            columns: Column mapping dictionary
            question: User's question (for context)
            
        Returns:
            DataFrame with columns: month, salesman, revenue
        """
        try:
            date_col = columns.get('date', 'VISITDATE')
            value_col = columns.get('amount', 'VALUE')
            rep_col = columns.get('salesman', 'Salesman Name')
            
            # Guardrail: check required columns exist
            if date_col not in df.columns or value_col not in df.columns or rep_col not in df.columns:
                return None
            
            d = df.copy()
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            
            # Clean numeric values (exact match to user's spec)
            ser = d[value_col]
            if ser.dtype == "object":
                ser = (ser.astype(str)
                         .str.replace(r"[^0-9\-\.\,]+", "", regex=True)
                         .str.replace(",", "", regex=False))
            d["_amt_"] = pd.to_numeric(ser, errors="coerce").fillna(0.0)
            d["_ym_"]  = d[date_col].dt.to_period("M").astype(str)
            
            # Group by month and salesman (exact match to user's spec)
            per_rep = d.groupby(["_ym_", rep_col])["_amt_"].sum().reset_index(name="revenue")
            
            # Find months with exactly one active salesman (exact match to user's spec)
            active  = (per_rep[per_rep["revenue"] > 0]
                       .groupby("_ym_")[rep_col].nunique()
                       .reset_index(name="active_reps"))
            
            only_one = active[active["active_reps"] == 1]["_ym_"]
            
            # Filter to months with exactly one salesman (exact match to user's spec)
            out = (per_rep[(per_rep["_ym_"].isin(only_one)) & (per_rep["revenue"] > 0)]
                   .sort_values(["_ym_", rep_col])
                   .rename(columns={"_ym_":"month", rep_col:"salesman"}))
            
            # Return exact columns as specified
            return out[["month","salesman","revenue"]] if not out.empty else None
            
        except Exception as e:
            ErrorHandlingService.log_error(
                f"Local fallback computation failed: {e}",
                category=ErrorCategory.DATA_PROCESSING
            )
            return None
