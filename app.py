import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import numpy as np
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta
px.defaults.color_discrete_sequence = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#A855F7", "#06B6D4"]
_rapid_layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E5E7EB'),
    xaxis=dict(showgrid=True, gridcolor='#1F2937', tickfont=dict(color='#9CA3AF'), title_font=dict(color='#E5E7EB')),
    yaxis=dict(showgrid=True, gridcolor='#1F2937', tickfont=dict(color='#9CA3AF'), title_font=dict(color='#E5E7EB')),
    legend=dict(font=dict(color='#E5E7EB')),
    colorway=["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#A855F7", "#06B6D4"],
)
pio.templates["rapid_dark"] = go.layout.Template(layout=_rapid_layout)
px.defaults.template = "rapid_dark"

# Optional Gemini availability check (graceful fallback if key missing)
USE_GEMINI = False
try:
    _k = st.secrets.get("GEMINI_API_KEY", None)
    if _k:
        genai.configure(api_key=_k)
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# Page configuration
st.set_page_config(
    page_title="Rapid Sales - TopSeven",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_css():
    st.markdown("""
    <style>
      :root{
        --rs-bg:#0B1020;
        --rs-surface:#111827;
        --rs-border:#1F2937;
        --rs-text:#E5E7EB;
        --rs-text-dim:#9CA3AF;
        --rs-primary:#2563EB;
        --rs-primary-hover:#1D4ED8;
        --rs-primary-active:#1E40AF;
        --rs-ring:#93C5FD;
        --rs-accent:#10B981;
        --rs-warn:#F59E0B;
        --rs-danger:#EF4444;
      }

      /* Page + text */
      .stApp { background: var(--rs-bg); color: var(--rs-text) !important; font-family: system-ui, -apple-system, 'Segoe UI', Inter, Roboto, Arial, sans-serif !important; }
      section.main > div { padding-top: 8px; }
      .stMarkdown, .stText, .stDownloadButton, .stDataFrame { color: var(--rs-text) !important; }
      .stApp .stMarkdown, .stApp .stMarkdown * { color: var(--rs-text) !important; }
      .stMarkdown *, .stDataFrame *, .stMetric *, .stExpander *, .stAlert *, .stSelectbox *, .stTextInput *, .stFileUploader * { color: var(--rs-text) !important; }
      h1, h2, h3, h4, h5, h6, p, span, label, li, td, th, a { color: var(--rs-text) !important; }

      /* Cards (expanders as cards) */
      div[data-testid="stExpander"] > details {
        background: var(--rs-surface) !important;
        border: 1px solid var(--rs-border) !important;
        border-radius: 16px !important;
      }

      /* Buttons */
      .stButton>button {
        background: var(--rs-primary) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        padding: 8px 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(37,99,235,.2) !important;
      }
      .stButton>button:hover { background: var(--rs-primary-hover) !important; }
      .stButton>button:active { background: var(--rs-primary-active) !important; }
      .stButton>button:focus { outline: 3px solid var(--rs-ring) !important; }

      /* Secondary (outline) */
      .rs-outline button {
        background: transparent !important;
        color: var(--rs-primary) !important;
        border: 1px solid var(--rs-primary) !important;
        box-shadow: none !important;
      }
      .rs-outline button:hover { background: rgba(37,99,235,.1) !important; }

      /* Tabs (pill style) */
      div[role="tablist"] > div[role="tab"] {
        border-radius: 12px !important;
        border: 1px solid transparent !important;
        color: var(--rs-text-dim) !important;
        padding: 6px 12px !important;
        margin-right: 8px !important;
      }
      div[role="tab"][aria-selected="true"]{
        background: var(--rs-surface) !important;
        color: var(--rs-text) !important;
        border: 1px solid var(--rs-border) !important;
        box-shadow: inset 0 -2px 0 var(--rs-primary) !important;
      }

      /* Dataframe */
      div[data-testid="stDataFrame"] thead tr th {
        background: #0F172A !important; color: var(--rs-text) !important;
      }
      div[data-testid="stDataFrame"] tbody tr:hover td {
        background: #0E152A !important;
      }
      div[data-testid="stDataFrame"] tbody tr:nth-child(even) td { background: #0D1326 !important; }
      div[data-testid="stDataFrame"] tbody td { border-color: var(--rs-border) !important; }
      div[data-testid="stDataFrame"] td { text-align: right !important; }
      div[data-testid="stDataFrame"] tbody td:first-child { text-align: left !important; }

      /* Chips */
      .rs-chip{display:inline-block;border-radius:999px;padding:4px 10px;font-size:12px;margin-right:6px}
      .rs-chip.neutral{background:#1F2937;color:#E5E7EB}
      .rs-chip.ok{background:#064E3B;color:#D1FAE5}
      .rs-chip.warn{background:#3F2C00;color:#FDE68A}
      .rs-chip.bad{background:#450A0A;color:#FECACA}

      /* Inputs */
      .stApp input, .stApp textarea, .stApp select {
        background: var(--rs-surface) !important;
        color: var(--rs-text) !important;
        border: 1px solid var(--rs-border) !important;
      }
      .stApp input::placeholder, .stApp textarea::placeholder { color: var(--rs-text-dim) !important; }
      .stApp input:focus, .stApp textarea:focus, .stApp select:focus { outline: 3px solid var(--rs-ring) !important; }

      /* Cards, metrics, containers */
      .metric-card, .insight-card { background: var(--rs-surface) !important; border: 1px solid var(--rs-border) !important; color: var(--rs-text) !important; }
      .stPlotlyChart { background: var(--rs-surface) !important; border-radius: 12px !important; }
      .main .block-container { background: transparent !important; }

      /* Header tweaks for dark */
      .professional-header { background: var(--rs-surface) !important; border-bottom: 1px solid var(--rs-border) !important; }
      .professional-header * { color: var(--rs-text) !important; }
      .company-badge { color: #fff !important; }

      /* Buttons disabled */
      .stButton>button:disabled { opacity: 0.6 !important; }

      /* Header layout (preserve structure) */
      .header-container { max-width: 1400px; margin: 0 auto; padding: 0 1rem; }
      .header-content { display: grid; grid-template-columns: auto 1fr; align-items: center; gap: 2rem; }
      .logo-section { display: flex; align-items: center; gap: 1rem; }
      .logo-icon { font-size: 2.25rem; }

      /* Mobile responsiveness */
      @media (max-width: 768px) {
        .header-content { grid-template-columns: 1fr !important; gap: 12px !important; }
        .stPlotlyChart { padding: 8px !important; margin: 8px 0 !important; }
        div[role="tab"] { font-size: 0.9rem !important; padding: 6px 10px !important; }
        .metric-card { padding: 1rem !important; }
        div[data-testid="stDataFrame"] { font-size: 0.9rem !important; }
        /* Force Streamlit columns to stack */
        div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        div[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
      }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'columns' not in st.session_state:
    st.session_state.columns = None
if 'selected_insight' not in st.session_state:
    st.session_state.selected_insight = None

# Period parsing helpers
def parse_period(text: str):
    """Return ('this_month'| 'last_month' | 'last_n_months', n | None) or None."""
    q = (text or "").lower()
    if "this month" in q:
        return ("this_month", None)
    if "last month" in q:
        return ("last_month", None)
    m = re.search(r'last\s+(\d+)\s+months?', q)
    if m:
        try:
            return ("last_n_months", int(m.group(1)))
        except Exception:
            return None

# Numeric formatting helpers (K/M/B) for tables
def _abbrev_num(v):
    try:
        v = float(v)
    except Exception:
        return v
    av = abs(v)
    if av >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}B"
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,.0f}"

def format_df_km(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(_abbrev_num)
    return out

def filter_by_period(df, date_col, period):
    if not period or not date_col or date_col not in df.columns:
        return df
    now = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    p, n = period
    if p == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start + relativedelta(months=1)
    elif p == "last_month":
        end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = end - relativedelta(months=1)
    elif p == "last_n_months" and n:
        end = now
        start = end - relativedelta(months=n)
    else:
        return df
    d = pd.to_datetime(df[date_col], errors='coerce')
    return df[(d >= start) & (d < end)].copy()

# Helper functions
def find_column(df, keywords):
    """
    Prefer exact/word-bound matches; fall back to safe substring if needed.
    Returns the first best match.
    """
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    # exact match
    for kw in keywords:
        for c in cols:
            if lower[c] == kw.lower():
                return c

    # word-boundary match
    for kw in keywords:
        pat = re.compile(rf'\b{re.escape(kw.lower())}\b')
        for c in cols:
            if pat.search(lower[c]):
                return c

    # safe substring match (avoid colliding words like salesman)
    blocklist = {"salesman", "sales_person", "sales man", "rep", "agent", "sales rep"}
    for kw in keywords:
        k = kw.lower()
        for c in cols:
            lc = lower[c]
            if k in lc and lc not in blocklist:
                return c
    return None

@st.cache_data(ttl=600)
def calculate_insights(df):
    """Calculate insights from data"""
    if df is None or len(df) == 0:
        return None
    
    # Auto-detect columns
    columns = {
        'customer': find_column(df, ['customer', 'client', 'account']),
        'amount': find_column(df, ['amount', 'sales', 'revenue', 'total', 'value']),
        'salesman': find_column(df, ['salesman', 'sales_person', 'sales man', 'rep', 'agent', 'sales rep']),
        'product': find_column(df, ['product', 'item', 'sku']),
        'date': find_column(df, ['date', 'transaction_date', 'order_date', 'visitdate', 'invoicedate']),
        'quantity': find_column(df, ['quantity', 'qty', 'units']),
        'price': find_column(df, ['price', 'unit_price', 'rate'])
    }

    # Parse dates early
    if columns['date'] and columns['date'] in df.columns:
        df[columns['date']] = pd.to_datetime(df[columns['date']], errors='coerce')

    # Derive amount if missing but qty & price exist
    if not columns['amount'] and columns['quantity'] and columns['price']:
        q, p = columns['quantity'], columns['price']
        if q in df.columns and p in df.columns:
            df['_AMOUNT_'] = pd.to_numeric(df[q], errors='coerce').fillna(0) * pd.to_numeric(df[p], errors='coerce').fillna(0)
            columns['amount'] = '_AMOUNT_'
    
    st.session_state.columns = columns
    
    insights = {
        'total_records': len(df),
        'columns': columns
    }
    
    # Calculate revenue metrics
    amount_col = columns['amount']
    if amount_col and amount_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[amount_col]):
            insights['total_revenue'] = float(df[amount_col].sum())
            insights['avg_order_value'] = float(df[amount_col].mean())
            insights['min_order'] = float(df[amount_col].min())
            insights['max_order'] = float(df[amount_col].max())
    
    # Top salesmen
    salesman_col = columns['salesman']
    if salesman_col and salesman_col in df.columns and amount_col:
        if pd.api.types.is_numeric_dtype(df[amount_col]):
            top_salesmen = df.groupby(salesman_col)[amount_col].sum().sort_values(ascending=False)
            insights['top_salesmen'] = top_salesmen.head(5).to_dict()
    
    # Top customers
    customer_col = columns['customer']
    if customer_col and customer_col in df.columns and amount_col:
        if pd.api.types.is_numeric_dtype(df[amount_col]):
            top_customers = df.groupby(customer_col)[amount_col].sum().sort_values(ascending=False)
            insights['top_customers'] = top_customers.head(5).to_dict()
            # Calculate percentages
            total_rev = insights.get('total_revenue', 1)
            insights['top_customers_list'] = [
                {
                    'name': name,
                    'revenue': float(revenue),
                    'percentage': round((revenue / total_rev) * 100, 1)
                }
                for name, revenue in top_customers.head(5).items()
            ]
    
    # Top products
    product_col = columns['product']
    if product_col and product_col in df.columns and amount_col:
        if pd.api.types.is_numeric_dtype(df[amount_col]):
            top_products = df.groupby(product_col)[amount_col].sum().sort_values(ascending=False)
            insights['top_products'] = top_products.head(5).to_dict()
            total_rev = insights.get('total_revenue', 1)
            insights['top_products_list'] = [
                {
                    'name': name,
                    'revenue': float(revenue),
                    'percentage': round((revenue / total_rev) * 100, 1)
                }
                for name, revenue in top_products.head(5).items()
            ]
    
    # Churn risk analysis
    if customer_col and customer_col in df.columns:
        customer_frequency = df[customer_col].value_counts().to_dict()
        if customer_frequency:
            avg_frequency = sum(customer_frequency.values()) / len(customer_frequency)
            churn_risk = [
                {
                    'name': name,
                    'visits': int(freq),
                    'risk': 'High'
                }
                for name, freq in customer_frequency.items()
                if freq < avg_frequency * 0.5
            ]
            insights['churn_risk'] = sorted(churn_risk, key=lambda x: x['visits'])[:5]
    
    # Forecast accuracy (simulated)
    insights['forecast_accuracy'] = 85
    
    return insights

def prepare_data_summary(df, insights):
    """Prepare data summary for Gemini"""
    if df is None or insights is None:
        return {}
    
    summary = {
        'total_records': insights.get('total_records', 0),
        'columns': list(df.columns),
        'column_types': {col: str(df[col].dtype) for col in df.columns}
    }
    
    # Add sample data with proper serialization
    sample_df = df.head(10).copy()
    
    # Convert timestamps and other non-serializable objects to strings
    for col in sample_df.columns:
        if sample_df[col].dtype == 'datetime64[ns]':
            sample_df[col] = sample_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif sample_df[col].dtype == 'object':
            # Convert any remaining non-serializable objects to strings
            sample_df[col] = sample_df[col].astype(str)
    
    summary['sample_data'] = sample_df.to_dict('records')
    
    # Add insights
    if 'total_revenue' in insights:
        summary['total_revenue'] = float(insights['total_revenue']) if insights['total_revenue'] is not None else 0
        summary['avg_order_value'] = float(insights['avg_order_value']) if insights['avg_order_value'] is not None else 0
    
    if 'top_salesmen' in insights:
        summary['top_salesmen'] = {k: float(v) for k, v in insights['top_salesmen'].items()}
    
    if 'top_customers_list' in insights:
        summary['top_customers'] = insights['top_customers_list']
    
    if 'top_products_list' in insights:
        summary['top_products'] = insights['top_products_list']
    
    if 'churn_risk' in insights:
        summary['churn_risk_customers'] = insights['churn_risk']
    
    return summary

def create_data_table(df, question, insights):
    """Return a DataFrame relevant to the question with period awareness."""
    if df is None:
        return None

    ql = (question or "").lower()
    cols = st.session_state.get('columns', {}) or {}
    amt = cols.get('amount')
    sm  = cols.get('salesman')
    dt  = cols.get('date')

    # Apply period filter if present
    per = parse_period(ql)
    if per:
        df = filter_by_period(df, dt, per)

    # If amount is missing after all, bail
    if not amt or amt not in df.columns:
        return None

    # Normalize amount to numeric
    vals = pd.to_numeric(df[amt], errors='coerce').fillna(0)
    df = df.assign(__amt=vals)

    # 1) Salesman questions
    if any(w in ql for w in ['salesman', 'sales rep', 'rep', 'best salesman', 'top salesman']):
        if sm and sm in df.columns:
            out = df.groupby(sm, dropna=False)['__amt'].sum().reset_index()
            out = out.sort_values('__amt', ascending=False).rename(columns={sm: 'Salesman', '__amt': 'Revenue'})
            out['Revenue'] = out['Revenue'].round(2)
            return out.head(10)

    # 2) Product questions
    prod = cols.get('product')
    if any(w in ql for w in ['product', 'item', 'sku', 'top product', 'best product']) and prod and prod in df.columns:
        out = df.groupby(prod, dropna=False)['__amt'].sum().reset_index()
        out = out.sort_values('__amt', ascending=False).rename(columns={prod: 'Product', '__amt': 'Revenue'})
        out['Revenue'] = out['Revenue'].round(2)
        return out.head(10)

    # 3) Customer questions
    cust = cols.get('customer')
    if any(w in ql for w in ['customer', 'client', 'account', 'top customer', 'best customer']) and cust and cust in df.columns:
        out = df.groupby(cust, dropna=False)['__amt'].sum().reset_index()
        out = out.sort_values('__amt', ascending=False).rename(columns={cust: 'Customer', '__amt': 'Revenue'})
        out['Revenue'] = out['Revenue'].round(2)
        return out.head(10)

    # Default: monthly rollup if date exists
    if dt and dt in df.columns:
        month = pd.to_datetime(df[dt], errors='coerce').dt.to_period('M').astype(str)
        out = df.groupby(month, dropna=False)['__amt'].sum().reset_index()
        out = out.rename(columns={dt: 'Month', '__amt': 'Revenue'})
        out['Revenue'] = out['Revenue'].round(2)
        out = out.sort_values('Month')
        return out

    return None

def query_ai(question, data_summary, df=None, insights=None):
    """
    If Gemini is available, ask it for a polished write-up.
    Otherwise, build a local, data-grounded answer.
    """
    try:
        if USE_GEMINI:
            # Ensure data_summary is JSON serializable
            try:
                json_data = json.dumps(data_summary, indent=2, default=str)
            except Exception:
                safe_summary = {}
                for key, value in data_summary.items():
                    if isinstance(value, (dict, list)):
                        safe_summary[key] = str(value)
                    else:
                        safe_summary[key] = value
                json_data = json.dumps(safe_summary, indent=2, default=str)

            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""You are a professional business intelligence analyst for Rapid Sales, a TopSeven enterprise solution.

KNOWLEDGE BASE (RAG - Retrieval Augmented Generation):
The uploaded Excel file serves as your complete knowledge base. All analysis must be grounded in this data.

SALES DATA CONTEXT:
{json_data}

BUSINESS QUERY: {question}

PROFESSIONAL RESPONSE GUIDELINES:
1. **Tone**: Formal business communication - precise, clear, executive-ready
2. **Data Grounding**: Base all insights strictly on the provided Excel data (RAG approach)
3. **Format**: Use professional markdown tables for all data presentations
4. **Structure**: 
   - Executive Summary (2-3 sentences)
   - Data Analysis (markdown table format)
   - Key Insights (bullet points)
   - Recommended Actions (numbered list)

5. **Table Format**: 
   | Metric | Value | Performance |
   |--------|-------|-------------|
   Use proper alignment and clear headers

6. **Precision**: Include specific numbers, percentages, and trends
7. **Clarity**: Avoid jargon; use clear business terminology
8. **Actionability**: Every insight must have a recommended action

RESPONSE STRUCTURE:
## Executive Summary
[Brief 2-3 sentence overview]

## Analysis
[Markdown table with relevant data]

## Key Insights
- [Insight 1]
- [Insight 2]

## Recommended Actions
1. [Action 1]
2. [Action 2]

Generate response:"""
            response = model.generate_content(prompt)
            return response.text

        # ---- Local fallback (no external LLM) ----
        table = create_data_table(df, question, insights)
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
        return "I couldn’t find the needed columns yet. Please check that amount/value and date/salesman exist."

    except Exception as e:
        return f"Error while answering: {e}"

# Professional TopSeven Header
st.markdown("""
<div class="professional-header">
    <div class="header-container">
        <div class="header-content">
            <div class="logo-section">
                <div class="logo-icon">🚀</div>
                <div class="logo-text">
                    <h1>Rapid Sales</h1>
                    <span class="company-badge">POWERED BY TOPSEVEN</span>
                </div>
            </div>
            <div class="header-info">
                <p class="tagline">Enterprise Sales Intelligence Platform</p>
                <p class="subtitle">AI-Powered Business Analytics & Data Insights</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Tabs - All 4 tabs should be visible
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "💬 Chat with AI", "📊 Detailed Analysis", "🤖 AI Insights"])

# Debug: Show tab count
st.sidebar.write(f"Total tabs: 4")

# Tab 1: Upload Data
with tab1:
    # Professional Upload Section
    st.markdown("""
    <div style="background: var(--rs-surface); padding: 2rem; border-radius: 12px; border: 1px solid var(--rs-border); margin-bottom: 2rem;">
        <h2 style="color: var(--rs-text); font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">Upload Your Sales Data</h2>
        <p style="color: var(--rs-text-dim); font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;">
            Upload your Excel file to unlock AI-powered insights into your sales performance, 
            customer behavior, and business opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload container with TopSeven branding
    upload_container = st.container()
    with upload_container:
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 2rem; border-radius: 12px; border: 2px dashed var(--rs-primary); margin-bottom: 1.5rem; text-align: center; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);">
            <div style="color: var(--rs-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">📁 Ready to Upload</div>
            <div style="color: var(--rs-text-dim); font-size: 0.9rem;">Drag and drop your Excel file here or click browse</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an Excel file to upload",
            type=['xlsx', 'xls'],
            help="Supported formats: .xlsx, .xls | Maximum file size: 200MB",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing your data... Please wait."):
                df = pd.read_excel(uploaded_file)
                st.session_state.data = df
                
                # Calculate insights
                insights = calculate_insights(df)
                st.session_state.insights = insights
            
            st.markdown("""
            <div style="background: rgba(16,185,129,0.12); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--rs-accent); margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">✅</span>
                    <div>
                        <strong style="color: #A7F3D0; font-size: 1rem;">Successfully loaded {:,} records</strong>
                        <p style="color: #6EE7B7; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                            Your data is ready for analysis. Head to the Chat with AI tab to start exploring!
                        </p>
                    </div>
                </div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
            
            # Data preview in a styled container
            with st.expander("📊 Preview Data (First 20 Rows)", expanded=False):
                st.dataframe(format_df_km(df.head(20)), use_container_width=True, height=400)
                
        except Exception as e:
            st.markdown(f"""
            <div style="background: rgba(239,68,68,0.12); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--rs-danger); margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">❌</span>
                    <div>
                        <strong style="color: #FCA5A5; font-size: 1rem;">Error loading file</strong>
                        <p style="color: #F87171; margin: 0.25rem 0 0 0; font-size: 0.875rem;">{str(e)}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(f"""
        <div style="background: rgba(37,99,235,0.12); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--rs-primary); margin: 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <div>
                    <strong style="color: #BFDBFE; font-size: 1rem;">Currently loaded: {len(df):,} records</strong>
                    <p style="color: #93C5FD; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                        Data is ready for analysis. Upload a new file to replace it.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 View Full Dataset", expanded=False):
            st.dataframe(df, use_container_width=True, height=400)
    else:
        # Show helpful instructions when no data is loaded
        st.markdown("""
        <div style="background: rgba(245,158,11,0.12); padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--rs-warn); margin: 1.5rem 0;">
            <h4 style="color: #FDE68A; margin: 0 0 0.75rem 0; font-size: 1.1rem;">💡 Getting Started</h4>
            <div style="color: #FCD34D; font-size: 0.95rem; line-height: 1.8;">
                <p style="margin: 0 0 0.5rem 0;"><strong>Step 1:</strong> Upload your Excel file using the file uploader above</p>
                <p style="margin: 0 0 0.5rem 0;"><strong>Step 2:</strong> Wait for the data to process (usually takes a few seconds)</p>
                <p style="margin: 0;"><strong>Step 3:</strong> Navigate to the "Chat with AI" tab to start asking questions about your data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Chat with AI
with tab2:
    if st.session_state.data is None:
        st.markdown("""
        <div style="background: rgba(245,158,11,0.12); padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--rs-warn); margin: 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">👆</span>
                <div>
                    <strong style="color: #FDE68A; font-size: 1rem;">No data uploaded</strong>
                    <p style="color: #FCD34D; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                        Please upload your data in the 'Upload Data' tab first to start chatting with AI.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--rs-border); margin-bottom: 1.5rem;">
            <h2 style="color: var(--rs-text); font-size: 1.75rem; font-weight: 700; margin: 0 0 0.5rem 0;">💬 Ask Questions About Your Data</h2>
            <p style="color: var(--rs-text-dim); font-size: 1rem; margin: 0; line-height: 1.6;">
                Try asking questions like: <strong>"Who is the best salesman?"</strong> or <strong>"Show me top 5 customers"</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">🧑 <strong>You:</strong><br>{content}</div>', unsafe_allow_html=True)
            else:
                # Handle both old format (string) and new format (dict with text and table)
                if isinstance(content, dict):
                    # Label bubble for AI
                    st.markdown('<div class="chat-message ai-message">🤖 <strong>AI</strong></div>', unsafe_allow_html=True)
                    # Render markdown content so tables render correctly
                    st.markdown(content["text"])
                    # Display table if available
                    if content.get("table"):
                        table_df = pd.DataFrame(content["table"])
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                else:
                    # Old format (just text)
                    st.markdown('<div class="chat-message ai-message">🤖 <strong>AI</strong></div>', unsafe_allow_html=True)
                    st.markdown(content)
        
        # Quick question buttons with better styling
        st.markdown("### Quick Questions")
        qcol1, qcol2, qcol3 = st.columns(3)
        
        with qcol1:
            if st.button("🏆 Who is the best salesman?", key="btn1"):
                query = "Who is the best salesman based on total sales? Provide specific numbers."
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Show thinking indicator
                with st.spinner("🤔 AI is analyzing your data..."):
                    data_summary = prepare_data_summary(st.session_state.data, st.session_state.insights)
                    response = query_ai(query, data_summary, st.session_state.data, st.session_state.insights)
                    data_table = create_data_table(st.session_state.data, query, st.session_state.insights)
                
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None
                }
                st.session_state.chat_history.append({"role": "assistant", "content": response_data})
                st.rerun()
        
        with qcol2:
            if st.button("📉 Show declining customers", key="btn2"):
                query = "Which customers show declining sales patterns? Identify at-risk customers."
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Show thinking indicator
                with st.spinner("🤔 AI is analyzing your data..."):
                    data_summary = prepare_data_summary(st.session_state.data, st.session_state.insights)
                    response = query_ai(query, data_summary, st.session_state.data, st.session_state.insights)
                    data_table = create_data_table(st.session_state.data, query, st.session_state.insights)
                
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None
                }
                st.session_state.chat_history.append({"role": "assistant", "content": response_data})
                st.rerun()
        
        with qcol3:
            if st.button("🎯 Salesmen needing coaching", key="btn3"):
                query = "Which salesmen need coaching based on their performance? Provide recommendations."
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Show thinking indicator
                with st.spinner("🤔 AI is analyzing your data..."):
                    data_summary = prepare_data_summary(st.session_state.data, st.session_state.insights)
                    response = query_ai(query, data_summary, st.session_state.data, st.session_state.insights)
                    data_table = create_data_table(st.session_state.data, query, st.session_state.insights)
                
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None
                }
                st.session_state.chat_history.append({"role": "assistant", "content": response_data})
                st.rerun()
        
        # Styled chat input container
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--rs-border); margin: 1rem 0;">
            <div style="color: var(--rs-text); font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">💬 Ask Your Question</div>
            <div style="color: var(--rs-text-dim); font-size: 0.9rem;">Type your question about the sales data below</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask a question about your sales data...", key="main_chat")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Show thinking indicator
            with st.spinner("🤔 AI is thinking..."):
                # Get AI response
                data_summary = prepare_data_summary(st.session_state.data, st.session_state.insights)
                response = query_ai(user_input, data_summary, st.session_state.data, st.session_state.insights)
                
                # Create data table if relevant
                data_table = create_data_table(st.session_state.data, user_input, st.session_state.insights)
            
            # Store response with table info
            response_data = {
                "text": response,
                "table": data_table.to_dict('records') if data_table is not None else None
            }
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_data})
            
            st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

# Tab 3: Detailed Analysis
with tab3:
    if st.session_state.data is None:
        st.warning("👆 Please upload your data in the 'Upload Data' tab first")
    else:
        insights = st.session_state.insights
        df = st.session_state.data
        
        # Enhanced Business Performance Metrics
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid var(--rs-border); box-shadow: inset 0 -2px 0 var(--rs-primary);">
            <h2 style="color: var(--rs-text); margin: 0 0 0.5rem 0; font-size: 1.75rem; font-weight: 700;">📊 Business Performance Metrics</h2>
            <p style="color: var(--rs-text-dim); margin: 0; font-size: 1rem;">Key insights and analytics from your sales data</p>
        </div>
        """, unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--topseven-blue); margin: 0 0 8px 0; font-size: 1.5rem;">{insights.get('total_records', 0):,}</h3>
                <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            if 'total_revenue' in insights:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: var(--topseven-orange); margin: 0 0 8px 0; font-size: 1.5rem;">${insights.get('total_revenue', 0):,.0f}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Total Revenue</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: var(--topseven-light-blue); margin: 0 0 8px 0; font-size: 1.5rem;">{len(df)}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Records Loaded</p>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_col3:
            if 'avg_order_value' in insights:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #10b981; margin: 0 0 8px 0; font-size: 1.5rem;">${insights.get('avg_order_value', 0):,.0f}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Avg Order Value</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #8b5cf6; margin: 0 0 8px 0; font-size: 1.5rem;">{len(df.columns)}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Data Columns</p>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_col4:
            if st.session_state.columns and st.session_state.columns['customer']:
                unique_customers = df[st.session_state.columns['customer']].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #ef4444; margin: 0 0 8px 0; font-size: 1.5rem;">{unique_customers}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Unique Customers</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #06b6d4; margin: 0 0 8px 0; font-size: 1.5rem;">{len(df)}</h3>
                    <p style="color: var(--topseven-gray); margin: 0; font-weight: 500;">Data Rows</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance Analytics Charts - MOVED TO TOP
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-primary); margin: 2rem 0 1rem 0;">
            <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">📈 Performance Analytics</h3>
            <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Time-series analysis and trend visualization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create sample time-series data for demonstration
        if st.session_state.columns and 'date' in st.session_state.columns and st.session_state.columns['date']:
            date_col = st.session_state.columns['date']
            if date_col in df.columns:
                # Find the revenue/amount column
                revenue_col = None
                if st.session_state.columns and 'amount' in st.session_state.columns and st.session_state.columns['amount']:
                    revenue_col = st.session_state.columns['amount']
                elif 'amount' in df.columns:
                    revenue_col = 'amount'
                elif 'revenue' in df.columns:
                    revenue_col = 'revenue'
                elif 'value' in df.columns:
                    revenue_col = 'value'
                elif 'price' in df.columns:
                    revenue_col = 'price'
                else:
                    # Try to find any numeric column that could be revenue
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        revenue_col = numeric_cols[0]
                
                if revenue_col and revenue_col in df.columns:
                    # Daily revenue trend
                    daily_revenue = df.groupby(df[date_col].dt.date)[revenue_col].sum().reset_index()
                    daily_revenue.columns = ['Date', 'Revenue']
                    
                    # Create professional line chart
                    fig_line = px.line(
                        daily_revenue,
                        x='Date',
                        y='Revenue',
                        title="Daily Revenue Trend"
                    )
                    fig_line.update_layout(
                        margin=dict(l=8, r=8, t=40, b=8),
                        height=360,
                        hovermode="x unified",
                        showlegend=False
                    )
                    fig_line.update_traces(
                        line=dict(width=2),
                        fill='tonexty',
                        fillcolor='rgba(37, 99, 235, 0.2)'
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("📊 No numeric column found for revenue analysis. Please ensure your data has a revenue/amount column.")
        
        # Sales Performance Metrics
        st.markdown("""
        <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-danger); margin: 2rem 0 1rem 0;">
            <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">🎯 Sales Performance Metrics</h3>
            <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Key performance indicators and business metrics</p>
        </div>
        """, unsafe_allow_html=True)
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            if 'total_revenue' in insights and 'total_records' in insights and insights['total_records'] > 0:
                avg_revenue = insights['total_revenue'] / insights['total_records']
                st.metric(
                    "Average Sale Value",
                    f"${avg_revenue:,.0f}",
                    delta=f"${avg_revenue * 0.05:,.0f}",
                    delta_color="normal"
                )
            else:
                st.metric(
                    "Total Records",
                    f"{len(df):,}",
                    delta="0",
                    delta_color="normal"
                )
        
        with perf_col2:
            if st.session_state.columns and 'customer' in st.session_state.columns and st.session_state.columns['customer']:
                customer_col = st.session_state.columns['customer']
                if customer_col in df.columns:
                    repeat_customers = df[customer_col].value_counts()
                    repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
                    st.metric(
                        "Customer Retention Rate",
                        f"{repeat_rate:.1f}%",
                        delta=f"{repeat_rate * 0.02:.1f}%",
                        delta_color="normal"
                    )
        
        with perf_col3:
            if 'top_salesmen' in insights and insights['top_salesmen']:
                top_salesman_revenue = max(insights['top_salesmen'].values())
                st.metric(
                    "Top Performer Revenue",
                    f"${top_salesman_revenue:,.0f}",
                    delta=f"${top_salesman_revenue * 0.03:,.0f}",
                    delta_color="normal"
                )
        
        # Top Salesmen
        if 'top_salesmen' in insights and insights['top_salesmen']:
            st.markdown("""
            <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-primary); margin: 2rem 0 1rem 0;">
                <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">🏆 Top 5 Salesmen</h3>
                <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Revenue performance by individual sales representatives</p>
            </div>
            """, unsafe_allow_html=True)
            top_salesmen_df = pd.DataFrame([
                {'Salesman': name, 'Revenue': revenue}
                for name, revenue in insights['top_salesmen'].items()
            ])
            st.dataframe(format_df_km(top_salesmen_df), use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_salesmen_df,
                x='Revenue',
                y='Salesman',
                orientation='h',
                title="Top Salesmen by Revenue"
            )
            fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=360, showlegend=False)
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Customers
        if 'top_customers_list' in insights:
            st.markdown("""
            <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-accent); margin: 2rem 0 1rem 0;">
                <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">👥 Top 5 Customers</h3>
                <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Highest revenue generating customer accounts</p>
            </div>
            """, unsafe_allow_html=True)
            top_customers_df = pd.DataFrame(insights['top_customers_list'])
            df_disp = top_customers_df[['name', 'revenue', 'percentage']].copy()
            if 'revenue' in df_disp.columns:
                df_disp['revenue'] = df_disp['revenue'].apply(_abbrev_num)
            if 'percentage' in df_disp.columns:
                df_disp['percentage'] = df_disp['percentage'].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_customers_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Customers by Revenue"
            )
            fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=360, showlegend=False)
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn Risk
        if 'churn_risk' in insights and insights['churn_risk']:
            st.markdown("""
            <div style="background: rgba(245,158,11,0.12); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-warn); margin: 2rem 0 1rem 0;">
                <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">⚠️ Churn Risk Alert</h3>
                <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Customers showing declining engagement patterns</p>
            </div>
            """, unsafe_allow_html=True)
            churn_df = pd.DataFrame(insights['churn_risk'])
            st.dataframe(format_df_km(churn_df), use_container_width=True, hide_index=True)
            
            st.info("💡 **AI Recommendation:** Schedule immediate visits to these customers. Offer special promotions to re-engage them.")
        
        # Top Products
        if 'top_products_list' in insights:
            st.markdown("""
            <div style="background: var(--rs-surface); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--rs-primary); margin: 2rem 0 1rem 0;">
                <h3 style="color: var(--rs-text); margin: 0; font-size: 1.5rem; font-weight: 700;">📦 Top 5 Products</h3>
                <p style="color: var(--rs-text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem;">Best performing products by revenue contribution</p>
            </div>
            """, unsafe_allow_html=True)
            top_products_df = pd.DataFrame(insights['top_products_list'])
            df_prod = top_products_df[['name', 'revenue', 'percentage']].copy()
            if 'revenue' in df_prod.columns:
                df_prod['revenue'] = df_prod['revenue'].apply(_abbrev_num)
            if 'percentage' in df_prod.columns:
                df_prod['percentage'] = df_prod['percentage'].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_prod, use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_products_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Products by Revenue"
            )
            fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=360, showlegend=False)
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        
        # Export functionality
        with st.expander("📥 Export Data"):
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Download Data as CSV",
                    csv,
                    f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                if st.session_state.chat_history:
                    chat_text = "\n\n".join([
                        f"{msg['role'].upper()}: {msg['content']}"
                        for msg in st.session_state.chat_history
                    ])
                    st.download_button(
                        "💬 Download Chat History",
                        chat_text,
                        f"chat_history_{datetime.now().strftime('%Y%m%d')}.txt",
                        "text/plain"
                    )

        insight_cards = [
            {
                'id': 'demand-forecast',
                'title': 'Demand Forecasting',
                'icon': '📈',
                'color': 'linear-gradient(135deg, var(--topseven-blue), var(--topseven-light-blue))',
                'description': 'AI predicts what each customer will order next',
                'details': f"Based on {insights.get('total_records', 0)} transactions, AI can predict inventory needs with {insights.get('forecast_accuracy', 85)}% accuracy, reducing stock-outs by 40%."
            },
            {
                'id': 'route-optimization',
                'title': 'Smart Route Planning',
                'icon': '📍',
                'color': 'linear-gradient(135deg, #10b981, #059669)',
                'description': 'Optimizes daily routes saving 2-3 hours per salesman',
                'details': 'AI considers traffic, visit duration, and customer priority to create optimal routes. Expected time savings: 25-30% per day.'
            },
            {
                'id': 'price-optimization',
                'title': 'Dynamic Pricing',
                'icon': '💰',
                'color': 'linear-gradient(135deg, var(--topseven-orange), #d97706)',
                'description': 'Suggests optimal prices and discounts per customer',
                'details': 'AI analyzes customer price sensitivity and competitor data to maximize both revenue and customer satisfaction.'
            },
            {
                'id': 'churn-prediction',
                'title': 'Churn Risk Detection',
                'icon': '⚠️',
                'color': 'linear-gradient(135deg, #ef4444, #dc2626)',
                'description': 'Identifies customers likely to stop ordering',
                'details': f"{len(insights.get('churn_risk', []))} customers identified as high churn risk. AI recommends immediate outreach." if insights.get('churn_risk') else 'AI monitors ordering patterns to flag at-risk customers before they churn.'
            },
            {
                'id': 'sales-target',
                'title': 'Intelligent Target Setting',
                'icon': '🎯',
                'color': 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                'description': 'Sets realistic, data-driven targets per salesman',
                'details': 'AI considers territory potential, seasonality, and individual performance to set achievable yet challenging targets.'
            },
            {
                'id': 'customer-segmentation',
                'title': 'Customer Segmentation',
                'icon': '👥',
                'color': 'linear-gradient(135deg, #06b6d4, #0891b2)',
                'description': 'Groups customers by behavior and value',
                'details': f"Identified {len(insights.get('top_customers_list', []))} high-value customers accounting for {sum([c.get('percentage', 0) for c in insights.get('top_customers_list', [])]):.1f}% of revenue." if insights.get('top_customers_list') else 'AI segments customers based on value, frequency, and buying patterns.'
            }
        ]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            for i in range(0, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card" onclick="alert('{card['title']}')">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            for i in range(1, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            for i in range(2, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display selected insight details
        if st.session_state.selected_insight:
            selected = next((c for c in insight_cards if c['id'] == st.session_state.selected_insight), None)
            if selected:
                st.markdown(f"""
                <div style="background: #1e293b; border: 2px solid #3b82f6; border-radius: 12px; padding: 24px; margin-top: 24px;">
                    <h3 style="color: {selected['color']}; font-size: 1.5rem; margin-bottom: 12px;">{selected['title']}</h3>
                    <p style="color: #cbd5e1; font-size: 1.125rem;">{selected['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Close Details"):
                    st.session_state.selected_insight = None
                    st.rerun()

# Tab 4: AI Insights
with tab4:
    if st.session_state.data is None:
        st.warning("👆 Please upload your data in the 'Upload Data' tab first")
    else:
        insights = st.session_state.insights
        
        # TopSeven AI Insight cards
        st.markdown("### 🧠 AI-Powered Business Solutions")
        st.markdown("Click on any insight to explore detailed analysis")
        
        col1, col2, col3 = st.columns(3)
        
        insight_cards = [
            {
                'id': 'demand-forecast',
                'title': 'Demand Forecasting',
                'icon': '📈',
                'color': 'linear-gradient(135deg, var(--topseven-blue), var(--topseven-light-blue))',
                'description': 'AI predicts what each customer will order next',
                'details': f"Based on {insights.get('total_records', 0)} transactions, AI can predict inventory needs with {insights.get('forecast_accuracy', 85)}% accuracy, reducing stock-outs by 40%."
            },
            {
                'id': 'route-optimization',
                'title': 'Smart Route Planning',
                'icon': '📍',
                'color': 'linear-gradient(135deg, #10b981, #059669)',
                'description': 'Optimizes daily routes saving 2-3 hours per salesman',
                'details': 'AI considers traffic, visit duration, and customer priority to create optimal routes. Expected time savings: 25-30% per day.'
            },
            {
                'id': 'price-optimization',
                'title': 'Dynamic Pricing',
                'icon': '💰',
                'color': 'linear-gradient(135deg, #f59e0b, #d97706)',
                'description': 'Adjusts prices based on demand and competition',
                'details': 'AI analyzes market conditions, customer behavior, and inventory levels to suggest optimal pricing strategies, potentially increasing revenue by 15-20%.'
            },
            {
                'id': 'inventory-management',
                'title': 'Smart Inventory',
                'icon': '📦',
                'color': 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                'description': 'Predicts stock needs and prevents shortages',
                'details': 'AI forecasts demand patterns and automatically suggests reorder points, reducing stock-outs by 60% and overstock by 40%.'
            },
            {
                'id': 'customer-insights',
                'title': 'Customer Analytics',
                'icon': '👥',
                'color': 'linear-gradient(135deg, #ef4444, #dc2626)',
                'description': 'Deep insights into customer behavior and preferences',
                'details': f"Identified {len(insights.get('top_customers_list', []))} high-value customers accounting for {sum([c.get('percentage', 0) for c in insights.get('top_customers_list', [])]):.1f}% of revenue." if insights.get('top_customers_list') else 'AI segments customers based on value, frequency, and buying patterns.'
            },
            {
                'id': 'sales-forecasting',
                'title': 'Sales Forecasting',
                'icon': '🔮',
                'color': 'linear-gradient(135deg, #06b6d4, #0891b2)',
                'description': 'Predicts future sales with high accuracy',
                'details': 'AI uses historical data, seasonality, and market trends to forecast sales with up to 90% accuracy, helping with planning and resource allocation.'
            },
            {
                'id': 'performance-tracking',
                'title': 'Performance Analytics',
                'icon': '📊',
                'color': 'linear-gradient(135deg, #ec4899, #db2777)',
                'description': 'Real-time tracking of sales performance',
                'details': 'AI monitors KPIs, identifies trends, and provides actionable insights to improve sales team performance and productivity.'
            },
            {
                'id': 'anomaly-detection',
                'title': 'Anomaly Detection',
                'icon': '🚨',
                'color': 'linear-gradient(135deg, #f97316, #ea580c)',
                'description': 'Identifies unusual patterns and potential issues',
                'details': 'AI detects anomalies in sales data, customer behavior, and market conditions, alerting you to potential problems before they impact business.'
            },
            {
                'id': 'customer-segmentation',
                'title': 'Customer Segmentation',
                'icon': '👥',
                'color': 'linear-gradient(135deg, #06b6d4, #0891b2)',
                'description': 'Groups customers by behavior and value',
                'details': f"Identified {len(insights.get('top_customers_list', []))} high-value customers accounting for {sum([c.get('percentage', 0) for c in insights.get('top_customers_list', [])]):.1f}% of revenue." if insights.get('top_customers_list') else 'AI segments customers based on value, frequency, and buying patterns.'
            }
        ]
        
        with col1:
            for i in range(0, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card" onclick="alert('{card['title']}')">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            for i in range(1, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            for i in range(2, len(insight_cards), 3):
                if i < len(insight_cards):
                    card = insight_cards[i]
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display selected insight details
        if st.session_state.selected_insight:
            selected = next((c for c in insight_cards if c['id'] == st.session_state.selected_insight), None)
            if selected:
                st.markdown(f"""
                <div style="background: #1e293b; border: 2px solid #3b82f6; border-radius: 12px; padding: 24px; margin-top: 24px;">
                    <h3 style="color: {selected['color']}; font-size: 1.5rem; margin-bottom: 12px;">{selected['title']}</h3>
                    <p style="color: #cbd5e1; font-size: 1.125rem;">{selected['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Close Details"):
                    st.session_state.selected_insight = None
                    st.rerun()

# Professional TopSeven Footer
st.markdown("---")

# Footer content using Streamlit columns
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### 🚀 Rapid Sales")
    st.markdown("**TOPSEVEN**")
    st.markdown("High-quality software systems delivering enterprise ERP solutions for the Middle East market.")

with col2:
    st.markdown("#### Contact")
    st.markdown("📧 contact@itop7.net")
    st.markdown("🌐 itop7.net")
    st.markdown("📱 +20 150 768 0215")

with col3:
    st.markdown("#### Location")
    st.markdown("13 Khaled Ibn Al-Waleed Street")
    st.markdown("Sheraton Al-Matar")
    st.markdown("Heliopolis, Egypt")

st.markdown("---")
st.markdown("**© 2025 TopSeven. All rights reserved. | Rapid Sales AI Demo**")

