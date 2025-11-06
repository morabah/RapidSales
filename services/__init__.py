"""
Services package for Rapid Sales application.
Provides business logic separation from UI components.
"""

from .column_detection_service import ColumnDetectionService
from .insight_service import InsightService
from .ai_service import AIService
from .data_formatting_service import (
    DataFormattingService,
    render_placeholders,
    has_unfilled_placeholders,
)
from .error_handling_service import (
    ErrorHandlingService,
    ErrorCategory,
    get_error_handler,
)
from .config_service import ConfigService, get_config
from .query_plan_service import QueryPlanService, QueryPlan, TimeMode, TimeFilter
from .time_filter_service import TimeFilterService
from .intent_parser_service import IntentParserService
from .plan_validator import validate_plan
from .sql_generation_service import SQLGenerationService
from .time_resolver import TimeResolver
from .currency_formatting_service import CurrencyFormattingService
from .dataset_cache_service import DatasetCacheService
from .comparison_service import ComparisonService
from .entity_normalization_service import EntityNormalizationService

__all__ = [
    'ColumnDetectionService',
    'InsightService',
    'AIService',
    'DataFormattingService',
    'render_placeholders',
    'has_unfilled_placeholders',
    'ErrorHandlingService',
    'ErrorCategory',
    'get_error_handler',
    'ConfigService',
    'get_config',
    'QueryPlanService',
    'QueryPlan',
    'TimeMode',
    'TimeFilter',
    'TimeFilterService',
    'IntentParserService',
    'validate_plan',
    'SQLGenerationService',
    'TimeResolver',
    'CurrencyFormattingService',
    'DatasetCacheService',
    'ComparisonService',
    'EntityNormalizationService',
]

