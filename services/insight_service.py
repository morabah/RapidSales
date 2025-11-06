"""
Insight Service
Handles data analysis, insights calculation, and business intelligence operations.
"""

import re
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .column_detection_service import ColumnDetectionService
from .error_handling_service import ErrorHandlingService, ErrorCategory
from .config_service import ConfigService


class InsightService:
    """Service for calculating insights and preparing data for analysis."""
    
    def __init__(self):
        self.column_detector = ColumnDetectionService()
    
    def parse_period(self, text: str) -> tuple[str, int | None] | None:
        """
        Parse period from text query.
        
        Returns:
            Tuple of (period_type, number_of_months) or None
        """
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
        return None
    
    def filter_by_period(
        self,
        df: pd.DataFrame,
        date_col: str | None,
        period: tuple[str, int | None] | None
    ) -> pd.DataFrame:
        """
        Filter DataFrame by time period.
        
        Args:
            df: DataFrame to filter
            date_col: Name of date column
            period: Period tuple from parse_period
            
        Returns:
            Filtered DataFrame
        """
        if not period or not date_col or date_col not in df.columns:
            return df
        
        now = datetime.now().replace(
            hour=23, minute=59, second=59, microsecond=0
        )
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
    
    def calculate_insights(
        self,
        df: pd.DataFrame,
        override_columns: dict[str, str] | None = None
    ) -> dict | None:
        """
        Calculate insights from sales data.
        
        Args:
            df: Input DataFrame
            override_columns: User-provided column mappings
            
        Returns:
            Dictionary of insights or None if invalid input
            
        Raises:
            Re-raises exceptions with context for error handling
        """
        if df is None or len(df) == 0:
            return None
        
        try:
            # Auto-detect columns
            columns = self.column_detector.detect_all_columns(df, override_columns)
            
            # Handle derived amount column
            if override_columns and override_columns.get('amount') == '__DERIVED_AMOUNT__':
                q, p = columns.get('quantity'), columns.get('price')
                if q and p and q in df.columns and p in df.columns:
                    df['_AMOUNT_'] = (
                        pd.to_numeric(df[q], errors='coerce').fillna(0) *
                        pd.to_numeric(df[p], errors='coerce').fillna(0)
                    )
                    columns['amount'] = '_AMOUNT_'
            
            # Parse dates early
            if columns['date'] and columns['date'] in df.columns:
                df[columns['date']] = pd.to_datetime(
                    df[columns['date']], errors='coerce'
                )
            
            # Auto-detect date column if missing
            if not columns['date']:
                best_col, best_ratio = None, 0.0
                for c in df.columns:
                    try:
                        s = pd.to_datetime(df[c], errors='coerce')
                        r = float(s.notna().mean())
                        if r > ConfigService.MIN_DATE_COVERAGE_RATIO and r > best_ratio:
                            best_col, best_ratio = c, r
                    except Exception:
                        pass
                if best_col:
                    columns['date'] = best_col
                    df[best_col] = pd.to_datetime(df[best_col], errors='coerce')
            
            # Derive amount if missing but qty & price exist
            if not columns['amount'] and columns['quantity'] and columns['price']:
                q, p = columns['quantity'], columns['price']
                if q in df.columns and p in df.columns:
                    df['_AMOUNT_'] = (
                        pd.to_numeric(df[q], errors='coerce').fillna(0) *
                        pd.to_numeric(df[p], errors='coerce').fillna(0)
                    )
                    columns['amount'] = '_AMOUNT_'
            
            # Auto-detect amount column if still missing
            if not columns['amount']:
                exclude = {
                    c for c in df.columns
                    if any(t in c.lower() for t in [
                        'qty', 'quantity', 'units', 'unit', 'pcs', 'price',
                        'unit_price', 'rate', 'tax', 'vat', 'discount', 'cost', 'cogs'
                    ])
                }
                best_col, best_sum = None, -1.0
                for c in df.columns:
                    if c in exclude:
                        continue
                    try:
                        s = pd.to_numeric(df[c], errors='coerce').fillna(0)
                        total = float(s.abs().sum())
                        if total > best_sum and s.notna().sum() > 0:
                            best_col, best_sum = c, total
                    except Exception:
                        pass
                if best_col:
                    columns['amount'] = best_col
            
            # Store in session state
            st.session_state.columns = columns
            
            insights = {
                'total_records': len(df),
                'columns': columns
            }
            
            # Calculate revenue metrics
            amount_col = columns['amount']
            vals = None
            if amount_col and amount_col in df.columns:
                ser_amt = df[amount_col]
                if ser_amt.dtype == 'object':
                    cleaned = ser_amt.astype(str).str.replace(
                        r'[^0-9\-\.]+', '', regex=True
                    )
                    vals = pd.to_numeric(cleaned, errors='coerce').fillna(0)
                else:
                    vals = pd.to_numeric(ser_amt, errors='coerce').fillna(0)
            
                insights['total_revenue'] = float(vals.sum())
                insights['avg_order_value'] = float(vals.mean())
                insights['min_order'] = float(vals.min())
                insights['max_order'] = float(vals.max())
            
            # Top salesmen
            salesman_col = columns['salesman']
            if salesman_col and salesman_col in df.columns and amount_col and vals is not None:
                tmp = pd.DataFrame({
                    'key': df[salesman_col],
                    '__amt': vals
                })
                top_salesmen = tmp.groupby('key')['__amt'].sum().sort_values(ascending=False)
                insights['top_salesmen'] = top_salesmen.head(ConfigService.MAX_TOP_ITEMS).to_dict()
            
            # Top customers
            customer_col = columns['customer']
            if customer_col and customer_col in df.columns and amount_col and vals is not None:
                tmp = pd.DataFrame({
                    'key': df[customer_col],
                    '__amt': vals
                })
                top_customers = tmp.groupby('key')['__amt'].sum().sort_values(ascending=False)
                insights['top_customers'] = top_customers.head(ConfigService.MAX_TOP_ITEMS).to_dict()
                
                # Calculate percentages
                total_rev = float(insights.get('total_revenue', 0)) or 1.0
                insights['top_customers_list'] = [
                {
                    'name': name,
                    'revenue': float(revenue),
                    'percentage': round((float(revenue) / total_rev) * 100, ConfigService.DEFAULT_PERCENT_DECIMAL_PLACES)
                }
                for name, revenue in top_customers.head(ConfigService.MAX_TOP_ITEMS).items()
                ]
            
            # Top products
            product_col = columns['product']
            if product_col and product_col in df.columns and amount_col and vals is not None:
                tmp = pd.DataFrame({
                    'key': df[product_col],
                    '__amt': vals
                })
                top_products = tmp.groupby('key')['__amt'].sum().sort_values(ascending=False)
                insights['top_products'] = top_products.head(ConfigService.MAX_TOP_ITEMS).to_dict()
                
                total_rev = float(insights.get('total_revenue', 0)) or 1.0
                insights['top_products_list'] = [
                {
                    'name': name,
                    'revenue': float(revenue),
                    'percentage': round((float(revenue) / total_rev) * 100, ConfigService.DEFAULT_PERCENT_DECIMAL_PLACES)
                }
                for name, revenue in top_products.head(ConfigService.MAX_TOP_ITEMS).items()
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
                    insights['churn_risk'] = sorted(
                        churn_risk, key=lambda x: x['visits']
                    )[:ConfigService.MAX_TOP_ITEMS]
            
            # Forecast accuracy (simulated)
            insights['forecast_accuracy'] = 85
            
            return insights
            
        except Exception as e:
            error_info = ErrorHandlingService.process_error(
                e,
                context='calculate_insights',
                category=ErrorCategory.DATA,
                details={'df_shape': df.shape if df is not None else None}
            )
            ErrorHandlingService.log_error(error_info)
            raise
    
    def prepare_data_summary(
        self,
        df: pd.DataFrame,
        insights: dict | None
    ) -> dict:
        """
        Prepare data summary for AI queries.
        
        Args:
            df: Input DataFrame
            insights: Calculated insights dictionary
            
        Returns:
            Summary dictionary for AI processing
        """
        if df is None or insights is None:
            return {}
        
        summary = {
            'total_records': insights.get('total_records', 0),
            'columns': list(df.columns),
            'column_types': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Add sample data with proper serialization
        sample_df = df.head(ConfigService.MAX_TOP_ITEMS).copy()
        
        # Convert timestamps and other non-serializable objects to strings
        for col in sample_df.columns:
            if sample_df[col].dtype == 'datetime64[ns]':
                sample_df[col] = sample_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif sample_df[col].dtype == 'object':
                sample_df[col] = sample_df[col].astype(str)
        
        summary['sample_data'] = sample_df.to_dict('records')
        summary['column_mapping'] = insights.get('columns', {})
        
        # Add date range if available
        mapping = insights.get('columns', {}) or {}
        dt_col = mapping.get('date')
        if dt_col and dt_col in df.columns:
            dts = pd.to_datetime(df[dt_col], errors='coerce')
            dts = dts.dropna()
            if not dts.empty:
                summary['date_range'] = {
                    'min': str(dts.min().date()),
                    'max': str(dts.max().date())
                }
        
        # Add insights
        if 'total_revenue' in insights:
            summary['total_revenue'] = (
                float(insights['total_revenue'])
                if insights['total_revenue'] is not None else 0
            )
            summary['avg_order_value'] = (
                float(insights['avg_order_value'])
                if insights['avg_order_value'] is not None else 0
            )
        
        # Include ALL salesmen data (not just top performers)
        # This allows AI to answer questions about worst/bottom performers
        if 'top_salesmen' in insights:
            # Store top salesmen
            summary['top_salesmen'] = {
                k: float(v) for k, v in insights['top_salesmen'].items()
            }
        
        # Also calculate and include ALL salesmen if we have the data
        mapping = insights.get('columns', {}) or {}
        salesman_col = mapping.get('salesman')
        amount_col = mapping.get('amount')
        
        if salesman_col and amount_col and salesman_col in df.columns and amount_col in df.columns:
            try:
                # Calculate revenue for ALL salesmen (not just top)
                vals = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
                all_salesmen = df.groupby(salesman_col, dropna=False)[amount_col].sum()
                summary['all_salesmen'] = {
                    k: float(v) for k, v in all_salesmen.items()
                }
            except Exception:
                pass  # Fallback to top_salesmen only if calculation fails
        
        if 'top_customers_list' in insights:
            summary['top_customers'] = insights['top_customers_list']
        
        if 'top_products_list' in insights:
            summary['top_products'] = insights['top_products_list']
        
        if 'churn_risk' in insights:
            summary['churn_risk_customers'] = insights['churn_risk']
        
        return summary
    
    def get_salesmen_performance(
        self,
        df: pd.DataFrame,
        cols: dict,
        *,
        k: int = 10,
        cap: int = 50
    ) -> dict:
        """
        Compute complete salesman performance and return a compact payload.
        
        This method computes the FULL ranking locally and returns:
        - 'top_k' / 'bottom_k' (size k)
        - 'all' when rep_count <= cap (token-safe); otherwise 'rep_count' only
        - Stable tie-breaking by total DESC then name ASC
        
        Args:
            df: DataFrame with sales data
            cols: Column mapping dictionary with 'salesman' and 'amount' keys
            k: Number of top/bottom performers to include (default 10)
            cap: Maximum rep count to include full 'all' data (default 50)
            
        Returns:
            Dictionary with:
            - 'rep_count': Total number of salesmen
            - 'top_k': Top k performers {salesman: {total, count, avg}}
            - 'bottom_k': Bottom k performers {salesman: {total, count, avg}}
            - 'all': Full ranking (only if rep_count <= cap)
        """
        salesman_col = cols.get('salesman')
        amount_col = cols.get('amount')
        
        if not salesman_col or not amount_col:
            return {"rep_count": 0, "top_k": {}, "bottom_k": {}}
        
        if df is None or df.empty:
            return {"rep_count": 0, "top_k": {}, "bottom_k": {}}
        
        # Coerce amount to numeric safely
        ser = df[amount_col]
        if ser.dtype == 'object':
            ser = ser.astype(str).str.replace(r'[^0-9\-\.\,]+', '', regex=True).str.replace(',', '', regex=False)
        amt = pd.to_numeric(ser, errors='coerce').fillna(0.0)
        
        # Build working frame
        tmp = pd.DataFrame({
            'salesman': df[salesman_col].astype(str).fillna(''),
            'amt': amt
        })
        
        # Group and aggregate
        g = tmp.groupby('salesman', dropna=False)['amt'].agg(total='sum', count='count', avg='mean').reset_index()
        
        # Tie-safe sorts
        g_desc = g.sort_values(['total', 'salesman'], ascending=[False, True])
        g_asc = g.sort_values(['total', 'salesman'], ascending=[True, True])
        
        rep_count = int(len(g))
        top_k = g_desc.head(k).set_index('salesman')[['total', 'count', 'avg']].round(ConfigService.DEFAULT_DECIMAL_PLACES).to_dict('index')
        bottom_k = g_asc.head(k).set_index('salesman')[['total', 'count', 'avg']].round(ConfigService.DEFAULT_DECIMAL_PLACES).to_dict('index')
        
        payload = {'rep_count': rep_count, 'top_k': top_k, 'bottom_k': bottom_k}
        
        if rep_count <= cap:
            payload['all'] = g_desc.set_index('salesman')[['total', 'count', 'avg']].round(ConfigService.DEFAULT_DECIMAL_PLACES).to_dict('index')
        
        return payload
    
    def create_data_table(
        self,
        df: pd.DataFrame,
        question: str,
        insights: dict | None,
        topn: int = ConfigService.DEFAULT_TOP_N
    ) -> pd.DataFrame | None:
        """
        Create relevant data table based on question.
        
        Args:
            df: Input DataFrame
            question: Business question
            insights: Insights dictionary
            topn: Number of top results to return
            
        Returns:
            Relevant DataFrame or None
        """
        if df is None:
            return None
        
        ql = (question or "").lower()
        cols = st.session_state.get('columns', {}) or {}
        amt = cols.get('amount')
        sm = cols.get('salesman')
        dt = cols.get('date')
        cust = cols.get('customer')
        prod = cols.get('product')
        
        # Apply date range filter if present
        dr = st.session_state.get('chat_date_range', None)
        if dr and dt and dt in df.columns:
            dcol = pd.to_datetime(df[dt], errors='coerce')
            try:
                start, end = dr
                if start and end:
                    start_ts = pd.to_datetime(start)
                    end_ts = pd.to_datetime(end) + pd.Timedelta(days=1)
                    df = df[(dcol >= start_ts) & (dcol < end_ts)].copy()
            except Exception:
                pass
        
        # Apply period filter if present
        per = self.parse_period(ql)
        if per:
            df = self.filter_by_period(df, dt, per)
        
        # Derive amount if missing
        if not amt or amt not in df.columns:
            q = cols.get('quantity')
            p = cols.get('price')
            if q and p and q in df.columns and p in df.columns:
                qv = pd.to_numeric(df[q], errors='coerce').fillna(0)
                pv = pd.to_numeric(df[p], errors='coerce').fillna(0)
                df = df.assign(__amt=(qv * pv))
                amt = '__amt'
            else:
                num_cols = [
                    c for c in df.columns
                    if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0
                ]
                if num_cols:
                    best_col, best_sum = None, -1.0
                    for c in num_cols:
                        s = pd.to_numeric(df[c], errors='coerce').fillna(0)
                        total = float(s.abs().sum())
                        if total > best_sum:
                            best_col, best_sum = c, total
                    if best_col:
                        amt = best_col
        
        if not amt or amt not in df.columns:
            return None
        
        # Normalize amount to numeric
        vals = pd.to_numeric(df[amt], errors='coerce').fillna(0)
        df = df.assign(__amt=vals)
        
        # Handle salesman questions
        if any(w in ql for w in [
            'salesman', 'sales rep', 'rep', 'best salesman', 'top salesman'
        ]):
            if sm and sm in df.columns:
                out = df.groupby(sm, dropna=False)['__amt'].sum().reset_index()
                out = out.sort_values('__amt', ascending=False).rename(
                    columns={sm: 'Salesman', '__amt': 'Revenue'}
                )
                out['Revenue'] = out['Revenue'].round(2)
                return out.head(topn)
        
        # Handle product questions
        if any(w in ql for w in [
            'product', 'item', 'sku', 'top product', 'best product'
        ]) and prod and prod in df.columns:
            out = df.groupby(prod, dropna=False)['__amt'].sum().reset_index()
            out = out.sort_values('__amt', ascending=False).rename(
                columns={prod: 'Product', '__amt': 'Revenue'}
            )
            out['Revenue'] = out['Revenue'].round(2)
            return out.head(topn)
        
        # Handle declining customers
        if any(w in ql for w in [
            'declin', 'at-risk', 'at risk', 'churn', 'losing', 'drop', 'decreas'
        ]) and cust and dt and cust in df.columns and dt in df.columns:
            roll = pd.DataFrame({
                'cust': df[cust],
                'month_dt': pd.to_datetime(df[dt], errors='coerce').dt.to_period('M'),
                '__amt': vals
            })
            roll = roll[roll['month_dt'].notna()]
            if roll.empty:
                return None
            
            last_months = sorted(roll['month_dt'].dropna().unique())
            if len(last_months) < 4:
                agg = roll.groupby('cust')['__amt'].sum().reset_index().rename(
                    columns={'cust': 'Customer', '__amt': 'Revenue'}
                )
                out = agg.sort_values('Revenue').head(topn)
                out['Revenue'] = out['Revenue'].round(2)
                return out
            
            last3 = last_months[-3:]
            prev3 = last_months[-6:-3]
            msk_last = roll['month_dt'].isin(last3)
            msk_prev = roll['month_dt'].isin(prev3)
            last_df = roll[msk_last].groupby('cust')['__amt'].sum()
            prev_df = roll[msk_prev].groupby('cust')['__amt'].sum()
            idx = set(last_df.index) | set(prev_df.index)
            records = []
            for ckey in idx:
                L = float(last_df.get(ckey, 0.0))
                P = float(prev_df.get(ckey, 0.0))
                if P > 0 and L < P:
                    change = L - P
                    pct = (change / P) * 100.0
                    records.append({
                        'Customer': ckey,
                        'Prev 3M': round(P, 2),
                        'Last 3M': round(L, 2),
                        'Change': round(change, 2),
                        'Change %': round(pct, 1)
                    })
            if records:
                out = pd.DataFrame(records).sort_values(
                    ['Change %', 'Change']
                ).head(topn)
                return out
            recent = roll[msk_last].groupby('cust')['__amt'].sum().reset_index().rename(
                columns={'cust': 'Customer', '__amt': 'Last 3M'}
            )
            out = recent.sort_values('Last 3M').head(topn)
            out['Last 3M'] = out['Last 3M'].round(2)
            return out
        
        # Handle customer questions
        if any(w in ql for w in [
            'customer', 'client', 'account', 'top customer', 'best customer'
        ]) and cust and cust in df.columns:
            out = df.groupby(cust, dropna=False)['__amt'].sum().reset_index()
            out = out.sort_values('__amt', ascending=False).rename(
                columns={cust: 'Customer', '__amt': 'Revenue'}
            )
            out['Revenue'] = out['Revenue'].round(2)
            return out.head(topn)
        
        # Default: monthly rollup if date exists
        if dt and dt in df.columns:
            month = pd.to_datetime(df[dt], errors='coerce').dt.to_period('M').astype(str)
            out = df.groupby(month, dropna=False)['__amt'].sum().reset_index()
            out = out.rename(columns={dt: 'Month', '__amt': 'Revenue'})
            out['Revenue'] = out['Revenue'].round(2)
            out = out.sort_values('Month')
            return out
        
        return None

