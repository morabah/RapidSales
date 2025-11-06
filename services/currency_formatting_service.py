"""
Currency Formatting Service
Handles currency formatting with configurable options (symbol position, separators, etc.)
"""

import streamlit as st
from typing import Optional
from .config_service import ConfigService


class CurrencyFormattingService:
    """Service for formatting currency with configurable options."""
    
    # Supported currencies
    SUPPORTED_CURRENCIES = {
        'USD': {'symbol': '$', 'name': 'US Dollar'},
        'EGP': {'symbol': 'EGP', 'name': 'Egyptian Pound'},
        'SAR': {'symbol': 'SAR', 'name': 'Saudi Riyal'},
        'EUR': {'symbol': '€', 'name': 'Euro'},
        'GBP': {'symbol': '£', 'name': 'British Pound'},
        'AED': {'symbol': 'AED', 'name': 'UAE Dirham'},
        'KWD': {'symbol': 'KWD', 'name': 'Kuwaiti Dinar'},
    }
    
    # Symbol position options
    SYMBOL_POSITIONS = {
        'prefix': 'Before amount (e.g., $1,234.56)',
        'suffix': 'After amount (e.g., 1,234.56 EGP)'
    }
    
    # Thousand separator options
    THOUSAND_SEPARATORS = {
        'comma': ',',
        'period': '.',
        'space': ' ',
        'none': ''
    }
    
    # Decimal separator options
    DECIMAL_SEPARATORS = {
        'period': '.',
        'comma': ','
    }
    
    # Negative style options
    NEGATIVE_STYLES = {
        'minus': '-1,234.56',
        'parentheses': '(1,234.56)',
        'minus_space': '- 1,234.56'
    }
    
    @classmethod
    def get_currency_settings(cls) -> dict:
        """Get current currency settings from session state."""
        return {
            'code': st.session_state.get('currency_code', ConfigService.DEFAULT_CURRENCY_CODE),
            'symbol_position': st.session_state.get('currency_symbol_position', 'prefix'),
            'thousand_separator': st.session_state.get('currency_thousand_sep', 'comma'),
            'decimal_separator': st.session_state.get('currency_decimal_sep', 'period'),
            'negative_style': st.session_state.get('currency_negative_style', 'minus'),
        }
    
    @classmethod
    def format_currency(
        cls,
        amount: float,
        currency_code: Optional[str] = None,
        symbol_position: Optional[str] = None,
        thousand_sep: Optional[str] = None,
        decimal_sep: Optional[str] = None,
        negative_style: Optional[str] = None
    ) -> str:
        """
        Format currency amount with configurable options.
        
        Args:
            amount: Numeric amount to format
            currency_code: Currency code (defaults to session state)
            symbol_position: 'prefix' or 'suffix' (defaults to session state)
            thousand_sep: Thousand separator key (defaults to session state)
            decimal_sep: Decimal separator key (defaults to session state)
            negative_style: Negative style key (defaults to session state)
            
        Returns:
            Formatted currency string
        """
        # Get settings from parameters or session state
        settings = cls.get_currency_settings()
        currency_code = currency_code or settings['code']
        symbol_position = symbol_position or settings['symbol_position']
        thousand_sep_key = thousand_sep or settings['thousand_separator']
        decimal_sep_key = decimal_sep or settings['decimal_separator']
        negative_style_key = negative_style or settings['negative_style']
        
        # Get currency symbol
        currency_info = cls.SUPPORTED_CURRENCIES.get(currency_code, {'symbol': currency_code})
        symbol = currency_info['symbol']
        
        # Get separators
        thousand_sep_char = cls.THOUSAND_SEPARATORS.get(thousand_sep_key, ',')
        decimal_sep_char = cls.DECIMAL_SEPARATORS.get(decimal_sep_key, '.')
        
        # Handle negative
        is_negative = amount < 0
        abs_amount = abs(amount)
        
        # Format number with separators
        # Use Python's format with custom separators
        formatted_parts = f"{abs_amount:,.{ConfigService.DEFAULT_DECIMAL_PLACES}f}".split('.')
        integer_part = formatted_parts[0]
        decimal_part = formatted_parts[1] if len(formatted_parts) > 1 else ''
        
        # Replace default separators with custom ones
        if thousand_sep_char != ',':
            integer_part = integer_part.replace(',', thousand_sep_char)
        if decimal_sep_char != '.':
            decimal_part = decimal_part.replace('.', decimal_sep_char)
        
        formatted_number = f"{integer_part}{decimal_sep_char}{decimal_part}" if decimal_part else integer_part
        
        # Apply negative style
        if is_negative:
            if negative_style_key == 'parentheses':
                formatted_number = f"({formatted_number})"
            elif negative_style_key == 'minus_space':
                formatted_number = f"- {formatted_number}"
            else:  # 'minus'
                formatted_number = f"-{formatted_number}"
        
        # Add symbol
        if symbol_position == 'suffix':
            return f"{formatted_number} {symbol}"
        else:  # prefix
            return f"{symbol}{formatted_number}"
    
    @classmethod
    def get_preview(cls, amount: float = 123456.78) -> str:
        """
        Get formatted preview of currency settings.
        
        Args:
            amount: Amount to use for preview
            
        Returns:
            Formatted preview string
        """
        return cls.format_currency(amount)
    
    @classmethod
    def get_currency_info(cls, currency_code: str) -> dict:
        """Get currency information."""
        return cls.SUPPORTED_CURRENCIES.get(currency_code, {'symbol': currency_code, 'name': currency_code})

