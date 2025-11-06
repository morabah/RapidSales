"""
AI Prompt Templates
Centralized prompt templates for AI services.
"""


def get_bi_analyst_prompt(question: str, json_data: str) -> str:
    """
    Get the BI analyst prompt template for text generation with placeholders.
    
    Returns a markdown template with placeholders that will be filled with authoritative numbers.
    
    Args:
        question: Business question to answer
        json_data: JSON string containing pre-computed aggregates (for context)
        
    Returns:
        Formatted prompt string
    """
    return f"""
You are a business intelligence analyst writing a concise report.

CONTEXT DATA (for understanding only - numbers will be injected automatically):
{json_data}

BUSINESS QUESTION:
{question}

Write a markdown report using ONLY placeholders for numbers. Do NOT include actual numbers.

Available placeholders:
- {{total_revenue_value}} - Total revenue
- {{currency_symbol}} - Currency symbol ($, EGP, etc.)
- {{time_period}} - Time period (e.g., "2024-11 → 2025-01")
- {{top_salesman_name}} - Top performer name
- {{top_salesman_revenue}} - Top performer revenue
- {{bottom_salesman_name}} - Bottom performer name
- {{bottom_salesman_revenue}} - Bottom performer revenue
- {{total_salesmen_count}} - Total number of salesmen
- {{avg_order_value_value}} - Average order value

If the context contains "quarter_comparison" data, use these placeholders for comparative questions:
- {{last_quarter_label}} - Last quarter label (e.g., "Q2-2025")
- {{last_quarter_range}} - Last quarter date range (e.g., "2025-04-01 → 2025-06-30")
- {{previous_quarter_label}} - Previous quarter label (e.g., "Q1-2025")
- {{previous_quarter_range}} - Previous quarter date range (e.g., "2025-01-01 → 2025-03-31")
- {{most_improved_salesman}} - Most improved salesman name
- {{most_improved_prev}} - Previous quarter sales (formatted)
- {{most_improved_current}} - Last quarter sales (formatted)
- {{most_improved_change}} - Absolute change (delta, formatted)
- {{most_improved_pct}} - Percentage change string (N/A if previous was 0.00)

IMPORTANT: When previous quarter is 0.00, report the absolute delta ({{most_improved_change}}) and percentage as "N/A (prev: 0.00)" or "N/A". Do NOT calculate percentage change when baseline is zero.

RULES:
1. Use placeholders like {{total_revenue_value}}, {{top_salesman_name}}, etc.
2. Do NOT include actual numbers - use placeholders only
3. Write in markdown format
4. Keep it concise and professional
5. For "worst/bottom" questions, reference {{bottom_salesman_name}} and {{bottom_salesman_revenue}}
6. For "top/best" questions, reference {{top_salesman_name}} and {{top_salesman_revenue}}
7. Always include {{time_period}} for context

EXAMPLE OUTPUT FORMAT:
## Sales Performance Analysis {{time_period}}

**Total Revenue:** {{total_revenue_value}} {{currency_symbol}} ({{time_period}})

**Top Performer:** {{top_salesman_name}} — {{top_salesman_revenue}} {{currency_symbol}}

**Bottom Performer:** {{bottom_salesman_name}} — {{bottom_salesman_revenue}} {{currency_symbol}}

**Total Salesmen:** {{total_salesmen_count}}

## Key Insights

- Overall revenue reached {{total_revenue_value}} {{currency_symbol}} for {{time_period}}
- Average order value: {{avg_order_value_value}} {{currency_symbol}}
- Top performer {{top_salesman_name}} generated {{top_salesman_revenue}} {{currency_symbol}}

## Recommended Actions

- Analyze strategies from top performer {{top_salesman_name}}
- Provide targeted coaching to bottom performers

Return ONLY the markdown text with placeholders, no explanations.
"""


def get_layout_prompt(question: str, json_data: str) -> str:
    """
    Alternative prompt for generating layout JSON (used when LLM is available).
    """
    return get_bi_analyst_prompt(question, json_data)


