import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Rapid Sales AI Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match index.html design
st.markdown("""
<style>
    /* Gradient background matching index.html */
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e3a8a, #0f172a);
        color: #f8fafc;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card styling */
    .insight-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        transition: transform 0.2s, border-color 0.2s;
        cursor: pointer;
        margin-bottom: 20px;
    }
    
    .insight-card:hover {
        transform: scale(1.05);
        border-color: #3b82f6;
    }
    
    .insight-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
        font-size: 24px;
    }
    
    .insight-title {
        font-size: 1.25rem;
        font-weight: bold;
        margin-bottom: 8px;
        color: #f8fafc;
    }
    
    .insight-desc {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        max-width: 80%;
    }
    
    .user-message {
        background: #1e40af;
        margin-left: auto;
        color: white;
    }
    
    .ai-message {
        background: #1e293b;
        margin-right: auto;
        color: #f8fafc;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 24px;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 48px;
        text-align: center;
        transition: border-color 0.3s;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
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

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 16px;">
        <span style="font-size: 48px;">🧠</span>
        <h1 style="font-size: 3rem; font-weight: bold; color: #f8fafc; margin: 0;">Rapid Sales AI Demo</h1>
    </div>
    <p style="color: #94a3b8; font-size: 1.125rem;">Discover how AI transforms your sales operations</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "💬 Chat with AI", "🤖 AI Insights", "📊 Detailed Analysis"])

# Tab 1: Upload Data
with tab1:
    st.markdown("### Upload Your Sales Data")
    st.markdown("Upload your Excel file to see AI in action")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your sales data in Excel format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            
            # Calculate insights
            insights = calculate_insights(df)
            st.session_state.insights = insights
            
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    elif st.session_state.data is not None:
        df = st.session_state.data
        st.info(f"📊 Currently loaded: {len(df)} records")
        
        with st.expander("View Data", expanded=False):
            st.dataframe(df, use_container_width=True)

# Tab 2: Chat with AI
with tab2:
    if st.session_state.data is None:
        st.warning("👆 Please upload your data in the 'Upload Data' tab first")
    else:
        st.markdown("### 💬 Ask Questions About Your Data")
        st.markdown("Try asking questions like: *'Who is the best salesman?'* or *'Show me top 5 customers'*")
        
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
        
        # Insight cards matching index.html design
        st.markdown("### AI-Powered Insights")
        
        col1, col2, col3 = st.columns(3)
        
        insight_cards = [
            {
                'id': 'demand-forecast',
                'title': 'Demand Forecasting',
                'icon': '📈',
                'color': '#3b82f6',
                'description': 'AI predicts what each customer will order next',
                'details': f"Based on {insights.get('total_records', 0)} transactions, AI can predict inventory needs with {insights.get('forecast_accuracy', 85)}% accuracy, reducing stock-outs by 40%."
            },
            {
                'id': 'route-optimization',
                'title': 'Smart Route Planning',
                'icon': '📍',
                'color': '#10b981',
                'description': 'Optimizes daily routes saving 2-3 hours per salesman',
                'details': 'AI considers traffic, visit duration, and customer priority to create optimal routes. Expected time savings: 25-30% per day.'
            },
            {
                'id': 'price-optimization',
                'title': 'Dynamic Pricing',
                'icon': '💰',
                'color': '#8b5cf6',
                'description': 'Suggests optimal prices and discounts per customer',
                'details': 'AI analyzes customer price sensitivity and competitor data to maximize both revenue and customer satisfaction.'
            },
            {
                'id': 'churn-prediction',
                'title': 'Churn Risk Detection',
                'icon': '⚠️',
                'color': '#ef4444',
                'description': 'Identifies customers likely to stop ordering',
                'details': f"{len(insights.get('churn_risk', []))} customers identified as high churn risk. AI recommends immediate outreach." if insights.get('churn_risk') else 'AI monitors ordering patterns to flag at-risk customers before they churn.'
            },
            {
                'id': 'sales-target',
                'title': 'Intelligent Target Setting',
                'icon': '🎯',
                'color': '#f59e0b',
                'description': 'Sets realistic, data-driven targets per salesman',
                'details': 'AI considers territory potential, seasonality, and individual performance to set achievable yet challenging targets.'
            },
            {
                'id': 'customer-segmentation',
                'title': 'Customer Segmentation',
                'icon': '👥',
                'color': '#6366f1',
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
        
        # Summary cards
        st.markdown("### 📊 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Records", f"{insights.get('total_records', 0):,}")
        
        with metric_col2:
            if 'total_revenue' in insights:
                st.metric("Total Revenue", f"${insights.get('total_revenue', 0):,.0f}")
            else:
                st.metric("Records Loaded", len(df))
        
        with metric_col3:
            if 'avg_order_value' in insights:
                st.metric("Avg Order Value", f"${insights.get('avg_order_value', 0):,.0f}")
            else:
                st.metric("Columns", len(df.columns))
        
        with metric_col4:
            if st.session_state.columns and st.session_state.columns['customer']:
                unique_customers = df[st.session_state.columns['customer']].nunique()
                st.metric("Unique Customers", unique_customers)
            else:
                st.metric("Rows", len(df))
        
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

