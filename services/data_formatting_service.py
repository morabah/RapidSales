"""
Data Formatting Service
Handles numeric formatting and DataFrame display formatting.

Numeric Precision Policy:
- Computations: Use float64 for raw math (Pandas default)
- Formatting: Format at render time (display precision)
- Large totals: For very large totals or long accumulation, consider using
  decimal.Decimal and convert to string at the edge (not implemented by default)
- Rounding: Half-up to 2 decimals for currency (standard monetary rounding)
- Reuse: Use centralized rounding function (ROUND_HALF_UP) for consistency
"""

import re
import pandas as pd

# Placeholder rendering for LLM templates
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_\. ]+)\}")


def render_placeholders(template: str, ctx: dict[str, str]) -> str:
    """
    Render placeholders in template string with values from context.
    
    Args:
        template: Template string with placeholders like {total_revenue_value}
        ctx: Dictionary of placeholder keys to values
        
    Returns:
        String with placeholders replaced by values
    """
    def repl(m):
        key = m.group(1)
        return str(ctx.get(key, f"{{{key}}}"))  # Leave untouched if missing
    
    filled = PLACEHOLDER_RE.sub(repl, template)
    
    # Placeholder gate: assert no unresolved placeholders before returning
    try:
        assert_no_placeholders(filled)
    except ValueError:
        # If placeholders remain, raise to trigger fallback
        raise ValueError("Unresolved placeholders in narrative")
    
    return filled


def has_unfilled_placeholders(text: str) -> bool:
    """
    Check if text contains any unfilled placeholders.
    
    Args:
        text: Text to check
        
    Returns:
        True if any placeholders remain
    """
    return bool(PLACEHOLDER_RE.search(text))


def assert_no_placeholders(text: str) -> str:
    """
    Assert there are no unresolved placeholders in text.
    
    Raises ValueError if placeholders are found (gate before rendering).
    
    Args:
        text: Text to check
        
    Returns:
        Text if valid
        
    Raises:
        ValueError: If unresolved placeholders found
    """
    if PLACEHOLDER_RE.search(text):
        raise ValueError("Unresolved placeholders in narrative")
    return text


def normalize_digits(s: str) -> str:
    """
    Normalize Arabic-Indic digits and punctuation for heuristics/filters.
    
    Converts: ٠١٢٣٤٥٦٧٨٩٫٬ → 0123456789..
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789..")
    return s.translate(trans)


class DataFormattingService:
    """
    Service for formatting data for display.
    
    Numeric precision: Keep raw math in float64; format at render.
    For very large totals or many rows, consider decimal.Decimal for currency aggregation behind a flag.
    """
    
    @staticmethod
    def abbreviate_number(value: float | str) -> str:
        """
        Abbreviate large numbers to K/M/B format.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted string (e.g., "1.5M", "2.3K")
        """
        try:
            v = float(value)
        except Exception:
            return value
        
        av = abs(v)
        if av >= 1_000_000_000:
            return f"{v/1_000_000_000:.1f}B"
        if av >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if av >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,.0f}"
    
    def format_dataframe_km(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format numeric columns in DataFrame using K/M/B abbreviations.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(self.abbreviate_number)
        return out
    
    def format_chat_dataframe(
        self,
        df: pd.DataFrame,
        currency_settings: dict = None
    ) -> pd.DataFrame:
        """
        Format DataFrame for chat display with currency and percentage formatting.
        
        Args:
            df: DataFrame to format
            currency_settings: Dictionary with currency formatting settings (from CurrencyFormattingService)
            
        Returns:
            Formatted DataFrame
        """
        from .currency_formatting_service import CurrencyFormattingService
        
        out = df.copy()
        num_cols = [
            c for c in out.columns
            if pd.api.types.is_numeric_dtype(out[c])
        ]
        money_cols = [
            c for c in ['Revenue', 'Prev 3M', 'Last 3M', 'Change', 'Total', 'Amount', 'Value']
            if c in out.columns
        ]
        pct_cols = [c for c in ['Change %', 'Percentage'] if c in out.columns]
        
        # Use CurrencyFormattingService if settings provided
        if currency_settings:
            for c in money_cols:
                if c in num_cols:
                    out[c] = out[c].apply(
                        lambda x: CurrencyFormattingService.format_currency(
                            x,
                            currency_code=currency_settings.get('code'),
                            symbol_position=currency_settings.get('symbol_position'),
                            thousand_sep=currency_settings.get('thousand_separator'),
                            decimal_sep=currency_settings.get('decimal_separator'),
                            negative_style=currency_settings.get('negative_style')
                        )
                    )
        else:
            # Fallback to simple formatting
            for c in money_cols:
                if c in num_cols:
                    out[c] = out[c].map(lambda x: f"{x:,.2f}")
        
        for c in pct_cols:
            if c in num_cols:
                out[c] = out[c].map(lambda x: f"{x:.1f}%")
        
        return out


