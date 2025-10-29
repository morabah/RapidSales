import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Rapid Sales - TopSeven",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching TopSeven branding
st.markdown("""
<style>
    /* TopSeven Brand Colors */
    :root {
        --topseven-blue: #1e40af;
        --topseven-light-blue: #3b82f6;
        --topseven-orange: #f59e0b;
        --topseven-dark: #1e293b;
        --topseven-gray: #64748b;
        --topseven-light-gray: #f1f5f9;
    }
    
    /* Main app styling with TopSeven theme */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Ensure all text has proper contrast */
    * {
        color: inherit;
    }
    
    /* Force dark text on light backgrounds */
    .stApp, .stApp * {
        color: #1e293b !important;
    }
    
    /* Override any light text that might be hard to read */
    .stMarkdown, .stMarkdown * {
        color: #1e293b !important;
    }
    
    /* Ensure headings are dark and visible */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    /* Ensure paragraphs and text are dark */
    p, span, div {
        color: #1e293b !important;
    }
    
    /* Force all Streamlit text elements to be dark and visible */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1e293b !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
        color: #1e293b !important;
    }
    
    /* Ensure metric values are visible */
    .metric-container {
        color: #1e293b !important;
    }
    
    .metric-container * {
        color: #1e293b !important;
    }
    
    /* Ensure all text in cards is visible */
    .insight-card, .metric-card {
        color: #1e293b !important;
    }
    
    .insight-card *, .metric-card * {
        color: #1e293b !important;
    }
    
    /* Ensure chat messages have proper contrast */
    .chat-message {
        color: #1e293b !important;
    }
    
    .chat-message * {
        color: #1e293b !important;
    }
    
    /* Override any light text in Streamlit components */
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        color: #1e293b !important;
    }
    
    /* Ensure sidebar text is visible */
    .css-1d391kg {
        color: #1e293b !important;
    }
    
    .css-1d391kg * {
        color: #1e293b !important;
    }
    
    /* Force all Streamlit text to be dark and visible */
    .stApp .stMarkdown, .stApp .stMarkdown * {
        color: #1e293b !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1e293b !important;
    }
    
    .stApp p, .stApp span, .stApp div {
        color: #1e293b !important;
    }
    
    /* Fix tab content text visibility */
    .stTabs [data-baseweb="tab-panel"] {
        color: #1e293b !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] * {
        color: #1e293b !important;
    }
    
    /* Fix all Streamlit components text */
    .stDataFrame, .stDataFrame * {
        color: #1e293b !important;
    }
    
    .stMetric, .stMetric * {
        color: #1e293b !important;
    }
    
    .stButton, .stButton * {
        color: white !important;
    }
    
    .stExpander, .stExpander * {
        color: #1e293b !important;
    }
    
    .stAlert, .stAlert * {
        color: #1e293b !important;
    }
    
    /* Fix any remaining text visibility issues */
    .stApp * {
        color: #1e293b !important;
    }
    
    /* Override specific Streamlit classes that might have light text */
    .css-1v0mbdj, .css-1v0mbdj * {
        color: #1e293b !important;
    }
    
    .css-1cpxqw2, .css-1cpxqw2 * {
        color: #1e293b !important;
    }
    
    .css-1y4p8pa, .css-1y4p8pa * {
        color: #1e293b !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Header Styling */
    .professional-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-bottom: 3px solid var(--topseven-blue);
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .header-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 2.5rem;
        background: linear-gradient(135deg, var(--topseven-blue), var(--topseven-light-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .logo-text {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--topseven-dark);
        margin: 0;
        line-height: 1.2;
    }
    
    .company-badge {
        background: var(--topseven-orange);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        width: fit-content;
    }
    
    .header-info {
        text-align: right;
        max-width: 400px;
    }
    
    .tagline {
        font-size: 1rem;
        color: var(--topseven-gray);
        margin: 0 0 0.5rem 0;
        font-weight: 500;
        line-height: 1.4;
    }
    
    .subtitle {
        font-size: 0.875rem;
        color: var(--topseven-gray);
        margin: 0;
        opacity: 0.8;
        line-height: 1.3;
    }
    
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            text-align: center;
        }
        
        .header-info {
            text-align: center;
        }
        
        .main-title {
            font-size: 1.75rem;
        }
    }
    
    /* Card styling with TopSeven theme */
    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .insight-card:hover {
        transform: translateY(-4px);
        border-color: var(--topseven-light-blue);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    
    .insight-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .insight-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: var(--topseven-dark);
    }
    
    .insight-desc {
        color: var(--topseven-gray);
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    /* Chat messages with TopSeven styling */
    .chat-message {
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--topseven-blue) 0%, var(--topseven-light-blue) 100%);
        margin-left: auto;
        color: white;
        border-radius: 20px 20px 4px 20px;
    }
    
    .ai-message {
        background: white;
        margin-right: auto;
        color: var(--topseven-dark);
        border: 1px solid #e2e8f0;
        border-radius: 20px 20px 20px 4px;
    }
    
    /* Tab styling with TopSeven theme - Fixed text visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: white;
        padding: 2px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 8px 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        color: var(--topseven-dark) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--topseven-light-gray);
        color: var(--topseven-dark) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--topseven-blue) 0%, var(--topseven-light-blue) 100%);
        color: white !important;
        border-color: var(--topseven-blue);
    }
    
    /* Ensure tab text is always visible */
    .stTabs [data-baseweb="tab"] span {
        color: inherit !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
        margin: 0 !important;
    }
    
    /* Force text visibility on all tab elements */
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }
    
    /* Make sure inactive tabs have dark text */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
        color: #1e293b !important;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) * {
        color: #1e293b !important;
    }
    
    /* Override any Streamlit default tab text styling */
    .stTabs [data-baseweb="tab"] .stMarkdown {
        color: inherit !important;
    }
    
    .stTabs [data-baseweb="tab"] .stMarkdown p {
        color: inherit !important;
        margin: 0 !important;
    }
    
    /* Ensure text is visible on all tab states */
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: #1e293b !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) * {
        color: #1e293b !important;
    }
    
    /* Upload area with TopSeven styling */
    .upload-area {
        border: 2px dashed var(--topseven-gray);
        border-radius: 16px;
        padding: 48px;
        text-align: center;
        transition: all 0.3s ease;
        background: white;
    }
    
    .upload-area:hover {
        border-color: var(--topseven-light-blue);
        background: var(--topseven-light-gray);
    }
    
    /* Buttons with TopSeven styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--topseven-blue) 0%, var(--topseven-light-blue) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Metrics cards styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--topseven-light-blue);
    }
    
    /* Professional Footer Styling */
    .professional-footer {
        background: linear-gradient(135deg, var(--topseven-dark) 0%, #1e293b 100%);
        color: white;
        margin-top: 3rem;
        border-top: 3px solid var(--topseven-blue);
    }
    
    .footer-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .footer-content {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 3rem;
        padding: 3rem 0 2rem 0;
        align-items: start;
    }
    
    .footer-brand {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    .footer-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .footer-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, var(--topseven-light-blue), var(--topseven-orange));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .footer-brand-text h3 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .footer-badge {
        background: var(--topseven-orange);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-top: 0.25rem;
    }
    
    .footer-description {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .footer-links {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
    }
    
    .footer-section h4 {
        color: white;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        border-bottom: 2px solid var(--topseven-orange);
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    .contact-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    
    .contact-icon {
        font-size: 1rem;
        width: 20px;
        text-align: center;
    }
    
    .footer-section p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .footer-section ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .footer-section li {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        padding-left: 1rem;
        position: relative;
    }
    
    .footer-section li:before {
        content: "•";
        color: var(--topseven-orange);
        position: absolute;
        left: 0;
    }
    
    .footer-bottom {
        border-top: 1px solid #334155;
        padding: 1.5rem 0;
    }
    
    .footer-copyright {
        text-align: center;
    }
    
    .footer-copyright p {
        color: #64748b;
        font-size: 0.8rem;
        margin: 0;
    }
    
    @media (max-width: 768px) {
        .footer-content {
            grid-template-columns: 1fr;
            gap: 2rem;
        }
        
        .footer-links {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .footer-container {
            padding: 0 1rem;
        }
    }
    
    /* TopSeven branding elements */
    .topseven-badge {
        background: var(--topseven-orange);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Animated Arrow for Chat Tab */
    .animated-arrow-container {
        position: relative;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border-radius: 12px;
        border: 2px solid var(--topseven-light-blue);
        text-align: center;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .animated-arrow {
        font-size: 2.5rem;
        color: var(--topseven-blue);
        animation: bounceArrow 1.5s infinite;
        line-height: 1;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    @keyframes bounceArrow {
        0%, 100% {
            transform: translateY(0);
            opacity: 1;
        }
        50% {
            transform: translateY(-15px);
            opacity: 0.7;
        }
    }
    
    .arrow-text {
        color: var(--topseven-blue);
        font-weight: 600;
        font-size: 1rem;
        margin: 0;
        animation: pulse 2s infinite;
        display: block;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    /* Hide arrow when on Chat tab */
    .hidden {
        display: none;
    }
    
    /* COMPREHENSIVE TEXT VISIBILITY FIX */
    /* Force main content text to be dark and visible */
    .stApp {
        color: #1e293b !important;
    }
    
    /* Override Streamlit default text colors */
    .stApp .stMarkdown {
        color: #1e293b !important;
    }
    
    .stApp .stMarkdown * {
        color: #1e293b !important;
    }
    
    /* Specific overrides for common Streamlit elements */
    .stMarkdown, .stMarkdown * {
        color: #1e293b !important;
    }
    
    .stDataFrame, .stDataFrame * {
        color: #1e293b !important;
    }
    
    .stMetric, .stMetric * {
        color: #1e293b !important;
    }
    
    .stExpander, .stExpander * {
        color: #1e293b !important;
    }
    
    .stAlert, .stAlert * {
        color: #1e293b !important;
    }
    
    .stSelectbox, .stSelectbox * {
        color: #1e293b !important;
    }
    
    .stTextInput, .stTextInput * {
        color: #1e293b !important;
    }
    
    .stFileUploader, .stFileUploader * {
        color: #1e293b !important;
    }
    
    /* Tab content specifically */
    .stTabs [data-baseweb="tab-panel"] {
        color: #1e293b !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] * {
        color: #1e293b !important;
    }
    
    /* Ensure buttons have white text */
    .stButton, .stButton * {
        color: white !important;
    }
    
    /* Fix text in different background contexts */
    /* White/light backgrounds - dark text */
    .stApp .stMarkdown, .stApp .stMarkdown * {
        color: #1e293b !important;
    }
    
    /* Dark backgrounds - light text */
    .chat-message.user-message, .chat-message.user-message * {
        color: white !important;
    }
    
    .ai-message, .ai-message * {
        color: #1e293b !important;
    }
    
    /* Status messages with colored backgrounds */
    .stSuccess, .stSuccess * {
        color: #065f46 !important;
    }
    
    .stError, .stError * {
        color: #991b1b !important;
    }
    
    .stWarning, .stWarning * {
        color: #92400e !important;
    }
    
    .stInfo, .stInfo * {
        color: #1e40af !important;
    }
    
    /* Override any remaining light text */
    .css-1v0mbdj, .css-1cpxqw2, .css-1y4p8pa, .css-1d391kg {
        color: #1e293b !important;
    }
    
    .css-1v0mbdj *, .css-1cpxqw2 *, .css-1y4p8pa *, .css-1d391kg * {
        color: #1e293b !important;
    }
    
    /* Force all headings to be dark */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    /* Force all paragraphs to be dark */
    p {
        color: #1e293b !important;
    }
    
    /* Force all spans to be dark */
    span {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Helper functions
def find_column(df, keywords):
    """Find column name that matches keywords"""
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword.lower() in col_lower for keyword in keywords):
            return col
    return None

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
        'date': find_column(df, ['date', 'transaction_date', 'order_date']),
        'quantity': find_column(df, ['quantity', 'qty', 'units'])
    }
    
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
    """Create relevant data tables based on the question"""
    if df is None:
        return None
    
    question_lower = question.lower()
    
    # Top salesmen table
    if any(word in question_lower for word in ['salesman', 'sales rep', 'rep', 'best salesman', 'top salesman']):
        if 'top_salesmen' in insights and insights['top_salesmen']:
            top_salesmen_df = pd.DataFrame([
                {'Salesman': name, 'Revenue': f"${revenue:,.0f}"}
                for name, revenue in insights['top_salesmen'].items()
            ])
            return top_salesmen_df
    
    # Top customers table
    if any(word in question_lower for word in ['customer', 'client', 'top customer', 'best customer']):
        if 'top_customers_list' in insights and insights['top_customers_list']:
            top_customers_df = pd.DataFrame(insights['top_customers_list'])
            top_customers_df['revenue'] = top_customers_df['revenue'].apply(lambda x: f"${x:,.0f}")
            return top_customers_df[['name', 'revenue', 'percentage']]
    
    # Top products table
    if any(word in question_lower for word in ['product', 'item', 'top product', 'best product']):
        if 'top_products_list' in insights and insights['top_products_list']:
            top_products_df = pd.DataFrame(insights['top_products_list'])
            top_products_df['revenue'] = top_products_df['revenue'].apply(lambda x: f"${x:,.0f}")
            return top_products_df[['name', 'revenue', 'percentage']]
    
    # Churn risk table
    if any(word in question_lower for word in ['churn', 'risk', 'declining', 'at-risk']):
        if 'churn_risk' in insights and insights['churn_risk']:
            churn_df = pd.DataFrame(insights['churn_risk'])
            return churn_df
    
    return None

def query_ai(question, data_summary, df=None, insights=None):
    """Query Gemini AI with user question"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Ensure data_summary is JSON serializable
        try:
            json_data = json.dumps(data_summary, indent=2, default=str)
        except Exception as json_error:
            # If still having serialization issues, convert everything to strings
            safe_summary = {}
            for key, value in data_summary.items():
                if isinstance(value, (dict, list)):
                    safe_summary[key] = str(value)
                else:
                    safe_summary[key] = value
            json_data = json.dumps(safe_summary, indent=2, default=str)
        
        prompt = f"""You are an expert sales data analyst assistant for Rapid Sales application.

DATA SUMMARY:
{json_data}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the provided sales data to answer the question accurately
2. Provide specific numbers, names, and metrics from the data
3. If calculating something, briefly explain your reasoning
4. Format your answer clearly with bullet points when listing items
5. If the question asks for "best", "top", or "worst", provide rankings with actual values
6. For performance questions, provide actionable insights
7. Be concise but informative
8. Use professional but conversational tone
9. IMPORTANT: When providing lists, rankings, or comparisons, format them as markdown tables for better readability
10. Use markdown table format: | Column1 | Column2 | Column3 | with proper alignment

RESPONSE FORMAT:
- Start with a brief text summary
- Include relevant markdown tables for data presentation
- End with actionable insights or recommendations

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "JSON" in error_msg or "serializable" in error_msg:
            return f"Data processing error: Unable to process the data format. Please check your Excel file format and try again."
        elif "API" in error_msg or "key" in error_msg.lower():
            return f"API Error: {error_msg}\n\nPlease check your Gemini API key configuration."
        else:
            return f"Error processing your request: {error_msg}\n\nPlease try again or contact support if the issue persists."

# Professional TopSeven Header
st.markdown("""
<div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 2rem 1rem; border-bottom: 3px solid #1e40af; margin-bottom: 2rem;">
    <div style="max-width: 1200px; margin: 0 auto;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;">
            <div>
                <h1 style="color: #1e293b; font-size: 2.25rem; font-weight: 700; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.75rem;">
                    <span>🚀</span>
                    <span>Rapid Sales</span>
                </h1>
                <div style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; display: inline-block; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Powered by TopSeven
                </div>
            </div>
            <div style="text-align: right; max-width: 500px;">
                <p style="color: #475569; font-size: 1rem; font-weight: 500; margin: 0 0 0.5rem 0; line-height: 1.5;">
                    High-quality software systems for ERP markets in the Middle East
                </p>
                <p style="color: #64748b; font-size: 0.875rem; margin: 0; line-height: 1.4;">
                    AI-powered sales data analysis and business insights
                </p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Small animated arrow above tabs (only show when data is uploaded)
if st.session_state.data is not None:
    st.markdown("""
    <div style="text-align: center; margin: 0.5rem 0; padding: 0.5rem;">
        <div style="font-size: 1.5rem; color: #1e40af; animation: bounce 1.5s infinite; display: inline-block;">⬇️</div>
        <div style="color: #1e40af; font-weight: 600; font-size: 0.8rem; margin-top: 0.25rem; animation: pulse 2s infinite;">Click Chat with AI</div>
    </div>
    <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
    """, unsafe_allow_html=True)

# Tabs - All 4 tabs should be visible
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "💬 Chat with AI", "🤖 AI Insights", "📊 Detailed Analysis"])

# Debug: Show tab count
st.sidebar.write(f"Total tabs: 4")

# Tab 1: Upload Data
with tab1:
    # Professional Upload Section
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h2 style="color: #1e293b; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">Upload Your Sales Data</h2>
        <p style="color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;">
            Upload your Excel file to unlock AI-powered insights into your sales performance, 
            customer behavior, and business opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload container with better styling
    upload_container = st.container()
    with upload_container:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 2px dashed #cbd5e1; margin-bottom: 1.5rem;">
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📁 Choose an Excel file to upload",
            type=['xlsx', 'xls'],
            help="Supported formats: .xlsx, .xls | Maximum file size: 200MB",
            label_visibility="visible"
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
            <div style="background: #d1fae5; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">✅</span>
                    <div>
                        <strong style="color: #065f46; font-size: 1rem;">Successfully loaded {:,} records</strong>
                        <p style="color: #047857; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                            Your data is ready for analysis. Head to the Chat with AI tab to start exploring!
                        </p>
                    </div>
                </div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
            
            # Data preview in a styled container
            with st.expander("📊 Preview Data (First 20 Rows)", expanded=False):
                st.dataframe(df.head(20), use_container_width=True, height=400)
                
        except Exception as e:
            st.markdown(f"""
            <div style="background: #fee2e2; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444; margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">❌</span>
                    <div>
                        <strong style="color: #991b1b; font-size: 1rem;">Error loading file</strong>
                        <p style="color: #dc2626; margin: 0.25rem 0 0 0; font-size: 0.875rem;">{str(e)}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(f"""
        <div style="background: #dbeafe; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <div>
                    <strong style="color: #1e40af; font-size: 1rem;">Currently loaded: {len(df):,} records</strong>
                    <p style="color: #1e3a8a; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
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
        <div style="background: #fef3c7; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">
            <h4 style="color: #92400e; margin: 0 0 0.75rem 0; font-size: 1.1rem;">💡 Getting Started</h4>
            <div style="color: #78350f; font-size: 0.95rem; line-height: 1.8;">
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
        <div style="background: #fef3c7; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">👆</span>
                <div>
                    <strong style="color: #92400e; font-size: 1rem;">No data uploaded</strong>
                    <p style="color: #78350f; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                        Please upload your data in the 'Upload Data' tab first to start chatting with AI.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1.5rem;">
            <h2 style="color: #1e293b; font-size: 1.75rem; font-weight: 700; margin: 0 0 0.5rem 0;">💬 Ask Questions About Your Data</h2>
            <p style="color: #64748b; font-size: 1rem; margin: 0; line-height: 1.6;">
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
                    # New format with text and table
                    st.markdown(f'<div class="chat-message ai-message">🤖 <strong>AI:</strong><br>{content["text"]}</div>', unsafe_allow_html=True)
                    
                    # Display table if available
                    if content.get("table"):
                        table_df = pd.DataFrame(content["table"])
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                else:
                    # Old format (just text)
                    st.markdown(f'<div class="chat-message ai-message">🤖 <strong>AI:</strong><br>{content}</div>', unsafe_allow_html=True)
        
        # Quick question buttons
        st.markdown("### Quick Questions")
        qcol1, qcol2, qcol3 = st.columns(3)
        
        with qcol1:
            if st.button("🏆 Who is the best salesman?"):
                query = "Who is the best salesman based on total sales? Provide specific numbers."
                st.session_state.chat_history.append({"role": "user", "content": query})
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
            if st.button("📉 Show declining customers"):
                query = "Which customers show declining sales patterns? Identify at-risk customers."
                st.session_state.chat_history.append({"role": "user", "content": query})
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
            if st.button("🎯 Salesmen needing coaching"):
                query = "Which salesmen need coaching based on their performance? Provide recommendations."
                st.session_state.chat_history.append({"role": "user", "content": query})
                data_summary = prepare_data_summary(st.session_state.data, st.session_state.insights)
                response = query_ai(query, data_summary, st.session_state.data, st.session_state.insights)
                data_table = create_data_table(st.session_state.data, query, st.session_state.insights)
                response_data = {
                    "text": response,
                    "table": data_table.to_dict('records') if data_table is not None else None
                }
                st.session_state.chat_history.append({"role": "assistant", "content": response_data})
                st.rerun()
        
        # Chat input
        user_input = st.chat_input("Ask a question about your sales data...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
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

# Tab 3: AI Insights
with tab3:
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

# Tab 4: Detailed Analysis
with tab4:
    if st.session_state.data is None:
        st.warning("👆 Please upload your data in the 'Upload Data' tab first")
    else:
        insights = st.session_state.insights
        df = st.session_state.data
        
        # TopSeven Summary cards
        st.markdown("### 📊 Business Performance Metrics")
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
        
        # Top Salesmen
        if 'top_salesmen' in insights and insights['top_salesmen']:
            st.markdown("### 🏆 Top 5 Salesmen")
            top_salesmen_df = pd.DataFrame([
                {'Salesman': name, 'Revenue': revenue}
                for name, revenue in insights['top_salesmen'].items()
            ])
            st.dataframe(top_salesmen_df, use_container_width=True, hide_index=True)
            
            # Chart
            fig = px.bar(
                top_salesmen_df,
                x='Revenue',
                y='Salesman',
                orientation='h',
                title="Top Salesmen by Revenue",
                color='Revenue',
                color_continuous_scale='blues'
            )
            fig.update_layout(
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Customers
        if 'top_customers_list' in insights:
            st.markdown("### 👥 Top 5 Customers")
            top_customers_df = pd.DataFrame(insights['top_customers_list'])
            st.dataframe(top_customers_df[['name', 'revenue', 'percentage']], use_container_width=True, hide_index=True)
            
            # Chart
            fig = px.bar(
                top_customers_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Customers by Revenue",
                color='revenue',
                color_continuous_scale='greens'
            )
            fig.update_layout(
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn Risk
        if 'churn_risk' in insights and insights['churn_risk']:
            st.markdown("### ⚠️ Churn Risk Alert")
            churn_df = pd.DataFrame(insights['churn_risk'])
            st.dataframe(churn_df, use_container_width=True, hide_index=True)
            
            st.info("💡 **AI Recommendation:** Schedule immediate visits to these customers. Offer special promotions to re-engage them.")
        
        # Top Products
        if 'top_products_list' in insights:
            st.markdown("### 📦 Top 5 Products")
            top_products_df = pd.DataFrame(insights['top_products_list'])
            st.dataframe(top_products_df[['name', 'revenue', 'percentage']], use_container_width=True, hide_index=True)
            
            # Chart
            fig = px.bar(
                top_products_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Products by Revenue",
                color='revenue',
                color_continuous_scale='purples'
            )
            fig.update_layout(
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font_color='#f8fafc'
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

# Professional TopSeven Footer
st.markdown("---")

# Footer content using Streamlit columns
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.markdown("### 🚀 Rapid Sales")
    st.markdown("**Powered by TopSeven**")
    st.markdown("High-quality software systems for ERP markets in the Middle East")

with col2:
    st.markdown("#### 📞 Contact")
    st.markdown("📧 contact@itop7.net")
    st.markdown("🌐 itop7.net")
    st.markdown("📱 +20 150 768 0215")

with col3:
    st.markdown("#### 📍 Address")
    st.markdown("13 Khaled Ibn Al-Waleed Street")
    st.markdown("Sheraton Al-Matar - Heliopolis")

with col4:
    st.markdown("#### 🛠️ Services")
    st.markdown("• Microsoft Dynamics 365")
    st.markdown("• Rapid Sales")
    st.markdown("• Stock Control Management")
    st.markdown("• Self Service")

st.markdown("---")
st.markdown("**© 2025 TopSeven. All rights reserved. | Rapid Sales AI Demo**")

