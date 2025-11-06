"""
Intent Parser Service
Uses LLM to parse user questions into structured QueryPlans for safe execution.
"""

import json
import google.generativeai as genai
from typing import Optional

from .query_plan_service import QueryPlan, TimeMode, TimeFilter
from .config_service import ConfigService


class IntentParserService:
    """Service for parsing user questions into structured query plans using LLM."""
    
    def __init__(self):
        self.use_llm = self._check_gemini_availability()
        if self.use_llm:
            api_key = ConfigService.get_gemini_api_key()
            genai.configure(api_key=api_key)
    
    def _check_gemini_availability(self) -> bool:
        """Check if Gemini API is available."""
        api_key = ConfigService.get_gemini_api_key()
        return api_key is not None
    
    def parse_question(self, question: str) -> QueryPlan:
        """
        Parse user question into structured QueryPlan.
        
        Uses LLM for intent parsing if available, falls back to pattern matching.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryPlan with structured query specification
        """
        if self.use_llm:
            try:
                return self._parse_with_llm(question)
            except Exception:
                # Fallback to pattern matching
                pass
        
        # Fallback to pattern-based parsing
        from .query_plan_service import QueryPlanService
        return QueryPlanService.parse_question(question)
    
    def _parse_with_llm(self, question: str) -> QueryPlan:
        """
        Parse question using LLM to return structured QueryPlan JSON.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryPlan parsed from LLM response
        """
        prompt = f"""You are a query intent parser. Parse the user's business question into a structured query plan.

USER QUESTION: {question}

Return ONLY a JSON object with this exact structure:
{{
  "metric": "sales_value",
  "aggregation": "sum",
  "dimensions": ["salesman"],
  "time": {{
    "mode": "all_time",
    "n_months": null,
    "quarter": null,
    "year": null,
    "month": null
  }},
  "sort": [{{"by": "sales_value", "dir": "desc"}}],
  "limit": 10,
  "include_bottom": false,
  "include_top": true
}}

RULES:
- metric: "sales_value" (money), "units" (quantity), or "avg_price"
- aggregation: "sum", "avg", "count"
- dimensions: ["salesman"], ["customer"], ["product"], or combinations
- time.mode: "all_time", "last_completed_quarter", "this_quarter", "last_month", "this_month", "last_n_months", "specific_quarter", "specific_month"
- time.n_months: number if mode is "last_n_months" (e.g., 3 for "last 3 months")
- time.quarter: 1-4 if mode is "specific_quarter"
- time.year: year number if mode is "specific_quarter" or "specific_month"
- time.month: 1-12 if mode is "specific_month"
- sort: array of {{"by": "sales_value", "dir": "desc"}} or {{"by": "sales_value", "dir": "asc"}}
- limit: number from question (e.g., "top 5" → 5) or default 10
- include_bottom: true if question asks about "worst", "bottom", "lowest", "poor", "underperform"
- include_top: true if question asks about "top", "best", "highest", "leading", "great"

EXAMPLES:
- "Who is the worst salesman?" → {{"dimensions": ["salesman"], "include_bottom": true, "include_top": false, "limit": 10}}
- "Show top 5 salesmen in last quarter" → {{"dimensions": ["salesman"], "time": {{"mode": "last_completed_quarter"}}, "include_top": true, "limit": 5}}
- "Which customers bought the most in July 2024?" → {{"dimensions": ["customer"], "time": {{"mode": "specific_month", "month": 7, "year": 2024}}, "include_top": true}}

Return ONLY the JSON object, no other text."""

        model = genai.GenerativeModel(ConfigService.GEMINI_MODELS[0])
        response = model.generate_content(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response.text.strip()
            if text.startswith('```'):
                # Remove markdown code blocks
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()
            
            plan_data = json.loads(text)
            
            # Convert to QueryPlan
            time_filter = None
            if plan_data.get('time') and plan_data['time'].get('mode') != 'all_time':
                time_mode = TimeMode(plan_data['time']['mode'])
                time_filter = TimeFilter(
                    mode=time_mode,
                    n_months=plan_data['time'].get('n_months'),
                    quarter=plan_data['time'].get('quarter'),
                    year=plan_data['time'].get('year'),
                    month=plan_data['time'].get('month')
                )
            
            return QueryPlan(
                metric=plan_data.get('metric', 'sales_value'),
                aggregation=plan_data.get('aggregation', 'sum'),
                dimensions=plan_data.get('dimensions', ['salesman']),
                time_filter=time_filter,
                filters=plan_data.get('filters', []),
                sort=plan_data.get('sort', []),
                limit=plan_data.get('limit', 10),
                include_bottom=plan_data.get('include_bottom', False),
                include_top=plan_data.get('include_top', True)
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If LLM parsing fails, fall back to pattern matching
            from .query_plan_service import QueryPlanService
            return QueryPlanService.parse_question(question)

