"""
Error Handling Service
Centralized error processing and user-friendly error messages.
"""

from datetime import datetime
from typing import Any, Optional
import traceback


class ErrorCategory:
    """Error categories for classification."""
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    DATA_PROCESSING = "DATA_PROCESSING"
    API = "API"
    VALIDATION = "VALIDATION"
    AUTH = "AUTH"
    BUSINESS = "BUSINESS"


class ErrorHandlingService:
    """Service for centralized error handling and processing."""
    
    @staticmethod
    def process_error(
        error: Exception,
        context: str = "",
        category: str = ErrorCategory.SYSTEM,
        user_message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Process and structure an error for logging and user display.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred (e.g., "calculate_insights")
            category: Error category (ErrorCategory enum)
            user_message: Optional user-friendly message override
            details: Additional error details
            
        Returns:
            Dictionary with error information
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Generate user-friendly message if not provided
        if not user_message:
            user_message = ErrorHandlingService._generate_user_message(
                error, error_type, context
            )
        
        # Get stack trace for debugging
        stack_trace = traceback.format_exc()
        
        error_info = {
            "message": error_message,
            "type": error_type,
            "context": context,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "stack_trace": stack_trace,
        }
        
        return error_info
    
    @staticmethod
    def _generate_user_message(
        error: Exception,
        error_type: str,
        context: str
    ) -> str:
        """
        Generate user-friendly error message based on error type.
        
        Args:
            error: The exception
            error_type: Type name of the exception
            context: Context where error occurred
            
        Returns:
            User-friendly error message
        """
        error_msg = str(error).lower()
        
        # File/Data related errors
        if any(keyword in error_msg for keyword in [
            'file', 'excel', 'xlsx', 'xls', 'read', 'parse', 'decode'
        ]):
            if 'not found' in error_msg or 'no such file' in error_msg:
                return "The file you're trying to upload was not found. Please check the file path."
            if 'permission' in error_msg or 'access' in error_msg:
                return "You don't have permission to access this file. Please check file permissions."
            if 'corrupt' in error_msg or 'invalid' in error_msg:
                return "The Excel file appears to be corrupted or invalid. Please try a different file."
            return "There was an error reading your Excel file. Please ensure it's a valid .xlsx or .xls file."
        
        # Column/Data validation errors
        if any(keyword in error_msg for keyword in [
            'column', 'key', 'index', 'not found', 'missing'
        ]):
            return "Required data columns are missing. Please check your Excel file structure or use column mapping."
        
        # Type conversion errors
        if any(keyword in error_msg for keyword in [
            'cannot convert', 'invalid type', 'dtype', 'numeric'
        ]):
            return "Data type conversion error. Please check that numeric columns contain valid numbers."
        
        # API/Network errors
        if any(keyword in error_msg for keyword in [
            'api', 'request', 'timeout', 'connection', 'network'
        ]):
            return "AI service connection error. Please check your internet connection and try again."
        
        # Memory/Performance errors
        if any(keyword in error_msg for keyword in [
            'memory', 'out of memory', 'too large'
        ]):
            return "File is too large to process. Please try a smaller file or reduce the number of rows."
        
        # Default messages based on error type
        type_messages = {
            'KeyError': "A required data field is missing.",
            'ValueError': "Invalid data value detected. Please check your input.",
            'TypeError': "Data type mismatch. Please verify your data format.",
            'AttributeError': "An internal error occurred. Please try again or contact support.",
            'IndexError': "Data indexing error. Please check your data structure.",
        }
        
        return type_messages.get(
            error_type,
            f"An error occurred while {context}. Please try again or check your data."
        )
    
    @staticmethod
    def log_error(
        error_info: dict[str, Any] | str | Exception,
        category: str = ErrorCategory.SYSTEM,
        log_level: str = "ERROR"
    ) -> None:
        """
        Log error information (placeholder for structured logging).
        
        Args:
            error_info: Error dictionary from process_error, or error message string, or Exception
            category: Error category (if error_info is string/Exception)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Handle different input types
        if isinstance(error_info, Exception):
            error_info = ErrorHandlingService.process_error(
                error_info,
                context="unknown",
                category=category
            )
        elif isinstance(error_info, str):
            # Create a simple error dict from string
            error_info = {
                "message": error_info,
                "type": "Error",
                "context": "unknown",
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "details": {},
                "stack_trace": ""
            }
        
        # In production, this would integrate with structured logging
        # For now, we can use print or Streamlit's logging
        print(f"[{log_level}] {error_info['timestamp']} - {error_info['context']}: {error_info['message']}")
        if error_info.get('details'):
            print(f"Details: {error_info['details']}")
        if error_info.get('stack_trace'):
            print(f"Stack trace:\n{error_info['stack_trace']}")
    
    @staticmethod
    def display_error(
        error_info: dict[str, Any],
        show_details: bool = False
    ) -> str:
        """
        Generate error message for user display.
        
        Args:
            error_info: Error dictionary from process_error
            show_details: Whether to include technical details (for debugging)
            
        Returns:
            Formatted error message for display
        """
        message = error_info['message']
        
        if show_details and error_info.get('details'):
            details_str = ", ".join([
                f"{k}: {v}" for k, v in error_info['details'].items()
            ])
            return f"{message} ({details_str})"
        
        return message


# Singleton instance (optional, for consistent state if needed)
_error_handling_service = ErrorHandlingService()

def get_error_handler() -> ErrorHandlingService:
    """Get the error handling service instance."""
    return _error_handling_service


