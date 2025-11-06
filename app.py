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
import difflib
import io
import time
from dateutil.relativedelta import relativedelta

# Import services
from services import (
    ColumnDetectionService,
    InsightService,
    AIService,
    DataFormattingService,
    ErrorHandlingService,
    ErrorCategory,
    ConfigService,
    get_config,
)

# Initialize configuration
config = get_config()

# Apply chart configuration
px.defaults.color_discrete_sequence = config.CHART_COLORS
_rapid_layout = config.get_chart_layout()
pio.templates["rapid_dark"] = go.layout.Template(layout=_rapid_layout)
px.defaults.template = "rapid_dark"

# Initialize services
column_detection_service = ColumnDetectionService()
insight_service = InsightService()
ai_service = AIService()
data_formatting_service = DataFormattingService()

# Optional Gemini availability check (for backward compatibility)
USE_GEMINI = config.use_gemini()

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

      /* AI Insight card UX */
      .insight-card { display: grid; grid-template-columns: 56px 1fr; gap: 12px; padding: 14px; border-radius: 12px; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; min-height: 140px; }
      .insight-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,.25); border-color: var(--rs-primary) !important; }
      .insight-icon { width: 56px; height: 56px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 26px; }
      .insight-title { grid-column: 2 / -1; font-weight: 700; margin-bottom: 4px; }
      .insight-desc { grid-column: 2 / -1; color: var(--rs-text-dim); font-size: .95rem; line-height: 1.35; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; word-break: normal; overflow-wrap: anywhere; }
      .insight-actions { grid-column: 2 / -1; margin-top: 8px; }
      .insight-actions button { width: 100%; }

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
if 'override_columns' not in st.session_state:
    st.session_state.override_columns = {}
if 'chat_date_range' not in st.session_state:
    st.session_state.chat_date_range = None
if 'currency_prefix' not in st.session_state:
    st.session_state.currency_prefix = ""
if 'analysis_top_n' not in st.session_state:
    st.session_state.analysis_top_n = config.DEFAULT_TOP_N

# Helper functions for backward compatibility (wrappers around services)
def format_chat_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame for chat display."""
    prefix = st.session_state.get('currency_prefix') or ""
    return data_formatting_service.format_chat_dataframe(df, prefix)

def format_df_km(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame with K/M/B abbreviations - Delegates to DataFormattingService."""
    return data_formatting_service.format_dataframe_km(df)

def _abbrev_num(v):
    """Abbreviate number to K/M/B format - Delegates to DataFormattingService."""
    return data_formatting_service.abbreviate_number(v)

# Old functions moved to services - keeping wrappers for backward compatibility
# Note: All business logic has been moved to service classes
def find_column(df, keywords):
    """Find column by keywords - Delegates to ColumnDetectionService."""
    return column_detection_service.find_column(df, keywords)

@st.cache_data(ttl=config.CACHE_TTL)
def calculate_insights(df, override_columns=None):
    """Calculate insights from data - Delegates to InsightService."""
    return insight_service.calculate_insights(df, override_columns)

def prepare_data_summary(df, insights):
    """Prepare data summary for Gemini - Delegates to InsightService."""
    return insight_service.prepare_data_summary(df, insights)

def create_data_table(df, question, insights):
    """Return a DataFrame relevant to the question - Delegates to InsightService."""
    topn = st.session_state.get('analysis_top_n', 10)
    return insight_service.create_data_table(df, question, insights, topn)

def get_chart_spec(question: str, table: pd.DataFrame):
    if table is None or table.empty:
        return None
    ql = (question or "").lower()
    cols = set(table.columns)
    if {'Salesman', 'Revenue'}.issubset(cols) and len(table) >= 2:
        return {'type': 'bar', 'orientation': 'h', 'x': 'Revenue', 'y': 'Salesman', 'title': 'Top Salesmen by Revenue'}
    if {'Product', 'Revenue'}.issubset(cols) and len(table) >= 2:
        return {'type': 'bar', 'orientation': 'h', 'x': 'Revenue', 'y': 'Product', 'title': 'Top Products by Revenue'}
    if {'Customer', 'Revenue'}.issubset(cols) and len(table) >= 2 and not any(w in ql for w in ['declin', 'at-risk', 'at risk', 'churn', 'losing', 'drop', 'decreas']):
        return {'type': 'bar', 'orientation': 'h', 'x': 'Revenue', 'y': 'Customer', 'title': 'Top Customers by Revenue'}
    if {'Customer', 'Prev 3M', 'Last 3M'}.issubset(cols) and len(table) >= 2:
        return {'type': 'bar_group', 'x': 'Customer', 'series': ['Prev 3M', 'Last 3M'], 'title': 'Customer Spend: Prev 3M vs Last 3M'}
    if {'Month', 'Revenue'}.issubset(cols) and len(table) >= 2:
        return {'type': 'line', 'x': 'Month', 'y': 'Revenue', 'title': 'Monthly Revenue Trend'}
        return None

def build_chart_from_spec(table: pd.DataFrame, spec: dict):
    if not spec:
        return None
    t = spec.get('type')
    if t == 'bar':
        fig = px.bar(table, x=spec.get('x'), y=spec.get('y'), orientation=spec.get('orientation', 'v'), title=spec.get('title'))
        fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=360, showlegend=False)
        fig.update_layout(xaxis=dict(tickformat=',.2f', tickprefix=st.session_state.get('currency_prefix') or ""))
        fig.update_traces(marker_line_width=0, marker_line_color='white')
        return fig
    if t == 'bar_group':
        x = spec.get('x')
        series = spec.get('series', [])
        df_long = table.melt(id_vars=[x], value_vars=series, var_name='Metric', value_name='Value')
        fig = px.bar(df_long, x=x, y='Value', color='Metric', barmode='group', title=spec.get('title'))
        fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=380)
        fig.update_layout(yaxis=dict(tickformat=',.2f', tickprefix=st.session_state.get('currency_prefix') or ""))
        fig.update_traces(marker_line_width=0)
        return fig
    if t == 'line':
        fig = px.line(table, x=spec.get('x'), y=spec.get('y'), title=spec.get('title'))
        fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), height=360, hovermode='x unified', showlegend=False)
        fig.update_layout(yaxis=dict(tickformat=',.2f', tickprefix=st.session_state.get('currency_prefix') or ""))
        fig.update_traces(line=dict(width=2), fill='tonexty', fillcolor='rgba(37, 99, 235, 0.15)')
        return fig
    return None

def query_ai(question, data_summary, df=None, insights=None):
    """
    Query AI - Delegates to AIService.
    If Gemini is available, ask it for a polished write-up.
    Otherwise, build a local, data-grounded answer.
    """
    return ai_service.query(question, data_summary, df, insights)

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
        if st.button("Load included Sample data.xlsx", key="btn_load_sample"):
            try:
                sample_path = "/Volumes/Rabah_SSD/consultation/Rapid Sales/Rapid/Sample data.xlsx"
                xls2 = pd.ExcelFile(sample_path)
                df2 = pd.read_excel(sample_path, sheet_name=xls2.sheet_names[0])
                try:
                    if len(df2.columns) > 0:
                        unnamed = sum(str(c).startswith('Unnamed') for c in df2.columns)
                        if unnamed / max(len(df2.columns), 1) > 0.3:
                            raw_head = pd.read_excel(sample_path, sheet_name=xls2.sheet_names[0], header=None, nrows=5)
                            best_i, best_score = None, -1
                            for i in range(min(5, len(raw_head))):
                                vals = raw_head.iloc[i].astype(str).str.strip().str.lower().tolist()
                                ok = [v for v in vals if v and v not in ('nan', 'none') and not v.startswith('unnamed') and len(v) > 1]
                                score = len(set(ok))
                                if score > best_score:
                                    best_i, best_score = i, score
                            if best_i is not None and best_score > 0:
                                df2 = pd.read_excel(sample_path, sheet_name=xls2.sheet_names[0], header=best_i)
                except Exception:
                    pass
                st.session_state.data = df2
                insights2 = calculate_insights(df2)
                st.session_state.insights = insights2
                st.success("Loaded included Sample data.xlsx")
                st.rerun()
            except Exception as e:
                error_info = ErrorHandlingService.process_error(
                    e,
                    context='load_sample_data',
                    category=ErrorCategory.DATA,
                    user_message="Failed to load sample data. Please try uploading your own file."
                )
                ErrorHandlingService.log_error(error_info)
                st.error(ErrorHandlingService.display_error(error_info))
    
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            xls = pd.ExcelFile(io.BytesIO(bytes_data))
            sheet = st.selectbox("Select sheet", xls.sheet_names, index=0, key="sheet_select")
            with st.spinner("Processing your data... Please wait."):
                df = pd.read_excel(io.BytesIO(bytes_data), sheet_name=sheet)
                try:
                    if len(df.columns) > 0:
                        unnamed = sum(str(c).startswith('Unnamed') for c in df.columns)
                        if unnamed / max(len(df.columns), 1) > 0.3:
                            raw_head = pd.read_excel(io.BytesIO(bytes_data), sheet_name=sheet, header=None, nrows=5)
                            best_i, best_score = None, -1
                            for i in range(min(5, len(raw_head))):
                                vals = raw_head.iloc[i].astype(str).str.strip().str.lower().tolist()
                                ok = [v for v in vals if v and v not in ('nan', 'none') and not v.startswith('unnamed') and len(v) > 1]
                                score = len(set(ok))
                                if score > best_score:
                                    best_i, best_score = i, score
                            if best_i is not None and best_score > 0:
                                df = pd.read_excel(io.BytesIO(bytes_data), sheet_name=sheet, header=best_i)
                except Exception:
                    pass
                st.session_state.data = df
                
                # Schema drift guard: preflight check on load
                from services import DatasetCacheService
                # Get initial column mapping from insights calculation
                insights = calculate_insights(df)
                columns = insights.get('columns', {})
                
                # Validate required columns (early, explicit)
                is_valid, error_msg = DatasetCacheService.validate_required_columns(columns, df)
                if not is_valid:
                    # Stop processing and show column mapping UI with error code
                    error_code = "SCHEMA_MISSING_COLUMN"
                    st.session_state.insights = insights
                    st.session_state.columns = columns
                    st.session_state['schema_error'] = {
                        'code': error_code,
                        'message': error_msg,
                        'columns': columns
                    }
                    st.error(f"❌ Schema validation failed ({error_code}): {error_msg}")
                    st.info("💡 Please configure column mapping below to continue.")
                    # Force column mapping UI to be expanded
                    st.session_state['force_column_mapping'] = True
                else:
                    # Clear any previous schema errors
                    if 'schema_error' in st.session_state:
                        del st.session_state['schema_error']
                    if 'force_column_mapping' in st.session_state:
                        del st.session_state['force_column_mapping']
                    
                    # Store dataset hash for cache invalidation
                    dataset_hash = DatasetCacheService.compute_dataset_hash(df, columns)
                    st.session_state['dataset_hash'] = dataset_hash
                
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
                st.dataframe(format_df_km(df.head(config.MAX_DATA_ROWS_PREVIEW)), use_container_width=True, height=400)
            
            # Column mapping UI (expanded if schema error or forced)
            expand_column_mapping = st.session_state.get('force_column_mapping', False) or st.session_state.get('schema_error') is not None
            with st.expander("🧩 Configure Column Mapping", expanded=expand_column_mapping):
                cols_list = list(df.columns)
                cur = (st.session_state.columns or {})
                def _idx(options, val):
                    try:
                        return options.index(val) if val in options else 0
                    except Exception:
                        return 0
                c1, c2, c3 = st.columns(3)
                with c1:
                    sel_customer = st.selectbox("Customer", [""] + cols_list, index=_idx([""]+cols_list, cur.get('customer')), key="map_customer")
                with c2:
                    amt_options = [""] + cols_list + ["Derive: quantity × price"]
                    default_amt = cur.get('amount') if cur.get('amount') in cols_list else ""
                    sel_amount = st.selectbox("Amount", amt_options, index=_idx(amt_options, default_amt), key="map_amount")
                with c3:
                    sel_salesman = st.selectbox("Salesman", [""] + cols_list, index=_idx([""]+cols_list, cur.get('salesman')), key="map_salesman")
                c4, c5, c6, c7 = st.columns(4)
                with c4:
                    sel_product = st.selectbox("Product", [""] + cols_list, index=_idx([""]+cols_list, cur.get('product')), key="map_product")
                with c5:
                    sel_date = st.selectbox("Date", [""] + cols_list, index=_idx([""]+cols_list, cur.get('date')), key="map_date")
                with c6:
                    sel_qty = st.selectbox("Quantity", [""] + cols_list, index=_idx([""]+cols_list, cur.get('quantity')), key="map_qty")
                with c7:
                    sel_price = st.selectbox("Price", [""] + cols_list, index=_idx([""]+cols_list, cur.get('price')), key="map_price")
                if st.button("Apply Mapping", key="apply_mapping_btn"):
                    overrides = {
                        'customer': sel_customer or None,
                        'amount': ('__DERIVED_AMOUNT__' if sel_amount == 'Derive: quantity × price' else (sel_amount or None)),
                        'salesman': sel_salesman or None,
                        'product': sel_product or None,
                        'date': sel_date or None,
                        'quantity': sel_qty or None,
                        'price': sel_price or None,
                    }
                    st.session_state.override_columns = overrides
                    insights = calculate_insights(df, override_columns=overrides)
                    st.session_state.insights = insights
                    st.session_state.columns = insights.get('columns', {})
                    st.rerun()
                
        except Exception as e:
            error_info = ErrorHandlingService.process_error(
                e,
                context='file_upload',
                category=ErrorCategory.DATA,
                details={'filename': uploaded_file.name if uploaded_file else None}
            )
            ErrorHandlingService.log_error(error_info)
            error_message = ErrorHandlingService.display_error(error_info)
            st.markdown(f"""
            <div style="background: rgba(239,68,68,0.12); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--rs-danger); margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">❌</span>
                    <div>
                        <strong style="color: #FCA5A5; font-size: 1rem;">Error loading file</strong>
                        <p style="color: #F87171; margin: 0.25rem 0 0 0; font-size: 0.875rem;">{error_message}</p>
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
        
        df_tab2 = st.session_state.data
        dtc = (st.session_state.columns or {}).get('date') if st.session_state.columns else None
        if dtc and dtc in df_tab2.columns:
            dts = pd.to_datetime(df_tab2[dtc], errors='coerce')
            dmin, dmax = (dts.min(), dts.max())
            if pd.notna(dmin) and pd.notna(dmax):
                dr = st.date_input("Filter date range (optional)", value=(dmin.date(), dmax.date()), key="chat_date_picker")
                if isinstance(dr, tuple) and len(dr) == 2:
                    st.session_state.chat_date_range = dr
        st.session_state.currency_prefix = st.text_input("Currency prefix (optional)", value=st.session_state.currency_prefix or "", key="currency_prefix_input")
        
        # Advanced Settings (collapsed by default for simplicity)
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.session_state.analysis_top_n = st.slider(
                "Max rows in analysis tables", 
                min_value=5, 
                max_value=50, 
                value=int(st.session_state.analysis_top_n or config.DEFAULT_TOP_N), 
                step=1, 
                key="analysis_top_n_slider",
                help="Controls how many rows are shown in analysis tables (default: 10)"
            )
        
        # Display chat history
        for idx, message in enumerate(st.session_state.chat_history):
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
                        st.dataframe(format_chat_df(table_df), use_container_width=True, hide_index=True)
                    if content.get("chart") and content.get("table"):
                        table_df = pd.DataFrame(content["table"])
                        fig = build_chart_from_spec(table_df, content.get("chart"))
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_chat_{idx}")
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
                
                chart_spec = get_chart_spec(query, data_table) if data_table is not None else None
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None,
                    "chart": chart_spec
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
                
                chart_spec = get_chart_spec(query, data_table) if data_table is not None else None
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None,
                    "chart": chart_spec
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
                
                chart_spec = get_chart_spec(query, data_table) if data_table is not None else None
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None,
                    "chart": chart_spec
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
            
            chart_spec = get_chart_spec(user_input, data_table) if data_table is not None else None
            response_data = {
                "text": response,
                "table": data_table.to_dict('records') if data_table is not None else None,
                "chart": chart_spec
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
                    st.plotly_chart(fig_line, use_container_width=True, key="plot_daily_revenue_line")
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
            st.plotly_chart(fig, use_container_width=True, key="plot_top_salesmen_bar")
        
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
            st.plotly_chart(fig, use_container_width=True, key="plot_top_customers_bar")
        
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
            st.plotly_chart(fig, use_container_width=True, key="plot_top_products_bar")
        
        
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
                    if st.button("View details", key=f"view_{card['id']}_t3_col1"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                    if st.button("View details", key=f"view_{card['id']}_t3_col2"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                    if st.button("View details", key=f"view_{card['id']}_t3_col3"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                
                if st.button("Close Details", key="close_details_t3"):
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
                    <div class="insight-card" >
                        <div class="insight-icon" style="background: {card['color']}">
                            {card['icon']}
                        </div>
                        <div class="insight-title">{card['title']}</div>
                        <div class="insight-desc">{card['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View details", key=f"view_{card['id']}_t4_col1"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                    if st.button("View details", key=f"view_{card['id']}_t4_col2"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                    if st.button("View details", key=f"view_{card['id']}_t4_col3"):
                        st.session_state.selected_insight = card['id']
                        st.rerun()
        
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
                
                if st.button("Close Details", key="close_details_t4"):
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

