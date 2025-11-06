"""
Configuration Service
Centralized configuration management for the application.
"""

import streamlit as st
from typing import Optional


class ConfigService:
    """Service for managing application configuration."""
    
    # AI/API Configuration
    GEMINI_TIMEOUT: int = 30
    GEMINI_RETRY_ATTEMPTS: int = 2
    GEMINI_RETRY_DELAY_BASE: float = 0.8
    GEMINI_MODELS: list[str] = ['gemini-2.0-flash-exp', 'gemini-1.5-flash']
    
    # UI/Theme Configuration
    CHART_COLORS: list[str] = [
        "#2563EB", "#10B981", "#F59E0B", "#EF4444", 
        "#A855F7", "#06B6D4"
    ]
    
    # Chart Layout Configuration
    CHART_PAPER_BG: str = 'rgba(0,0,0,0)'
    CHART_PLOT_BG: str = 'rgba(0,0,0,0)'
    CHART_FONT_COLOR: str = '#E5E7EB'
    CHART_GRID_COLOR: str = '#1F2937'
    CHART_TICK_COLOR: str = '#9CA3AF'
    CHART_LEGEND_COLOR: str = '#E5E7EB'
    
    # Data Processing Configuration
    CACHE_TTL: int = 600  # 10 minutes
    DEFAULT_TOP_N: int = 25  # Changed from 10 to 25-50 range
    DEFAULT_MAX_ROWS: int = 25  # Default for display cap
    MAX_FILE_SIZE_MB: int = 50
    MAX_DATA_ROWS_PREVIEW: int = 20
    MAX_COLUMNS_SUMMARY: int = 50
    MAX_TOP_ITEMS: int = 10
    MAX_SAMPLE_DATA_ROWS: int = 5
    
    # Date/Time Configuration
    DEFAULT_DATE_RANGE_DAYS: int = 30
    USE_DATASET_MAX_AS_NOW: bool = True  # Time anchoring: use dataset max date as "now"
    INCLUDE_ZERO_MONTHS: bool = True  # Include zero months in time series
    FISCAL_YEAR_START_MONTH: int = 1  # Calendar quarters (1=Jan). Use 7 for fiscal (July start)
    DEFAULT_TIMEZONE: str = "UTC"  # Timezone for date parsing
    
    # Currency/Number Format Configuration
    DEFAULT_CURRENCY_PREFIX: str = ""
    DEFAULT_CURRENCY_CODE: str = "USD"  # USD, EGP, SAR, etc.
    DEFAULT_DECIMAL_PLACES: int = 2
    DEFAULT_PERCENT_DECIMAL_PLACES: int = 1
    
    # Currency symbols mapping
    CURRENCY_SYMBOLS: dict[str, str] = {
        'USD': '$',
        'EGP': 'EGP',
        'SAR': 'SAR',
        'EUR': '€',
        'GBP': '£',
    }
    
    # Analysis Configuration
    MIN_DATE_COVERAGE_RATIO: float = 0.7  # For auto-detecting date columns
    
    @classmethod
    def get_gemini_api_key(cls) -> Optional[str]:
        """Get Gemini API key from Streamlit secrets."""
        try:
            return st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            return None
    
    @classmethod
    def use_gemini(cls) -> bool:
        """Check if Gemini is available and should be used."""
        return cls.get_gemini_api_key() is not None
    
    @classmethod
    def get_chart_layout(cls) -> dict:
        """
        Get Plotly chart layout configuration.
        
        Returns:
            Dictionary with layout settings
        """
        import plotly.graph_objects as go
        
        return go.Layout(
            paper_bgcolor=cls.CHART_PAPER_BG,
            plot_bgcolor=cls.CHART_PLOT_BG,
            font=dict(color=cls.CHART_FONT_COLOR),
            xaxis=dict(
                showgrid=True,
                gridcolor=cls.CHART_GRID_COLOR,
                tickfont=dict(color=cls.CHART_TICK_COLOR),
                title_font=dict(color=cls.CHART_FONT_COLOR)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=cls.CHART_GRID_COLOR,
                tickfont=dict(color=cls.CHART_TICK_COLOR),
                title_font=dict(color=cls.CHART_FONT_COLOR)
            ),
            legend=dict(font=dict(color=cls.CHART_LEGEND_COLOR)),
            colorway=cls.CHART_COLORS,
        )
    
    @classmethod
    def get_analysis_top_n(cls) -> int:
        """Get the top N value for analysis from session state or default."""
        return st.session_state.get('analysis_top_n', cls.DEFAULT_TOP_N)
    
    @classmethod
    def get_currency_prefix(cls) -> str:
        """Get currency prefix from session state or default."""
        return st.session_state.get('currency_prefix', cls.DEFAULT_CURRENCY_PREFIX)
    
    @classmethod
    def format_currency(cls, amount: float) -> str:
        """
        Format currency amount with proper symbol and formatting.
        
        Args:
            amount: Numeric amount to format
            
        Returns:
            Formatted currency string (e.g., "$1,234.56" or "EGP 1,234.56")
        """
        currency_code = st.session_state.get('currency_code', cls.DEFAULT_CURRENCY_CODE)
        symbol = cls.CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        # Format with thousands separators
        formatted = f"{amount:,.{cls.DEFAULT_DECIMAL_PLACES}f}"
        
        # Add symbol (prefix for most, suffix for some)
        if currency_code in ['EGP', 'SAR']:
            return f"{symbol} {formatted}"
        else:
            return f"{symbol}{formatted}"


# Singleton instance
_config_service = ConfigService()

def get_config() -> ConfigService:
    """Get the configuration service instance."""
    return _config_service

