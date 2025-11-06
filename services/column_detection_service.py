"""
Column Detection Service
Handles intelligent column detection from Excel files using fuzzy matching.
"""

import re
import difflib
import pandas as pd


class ColumnDetectionService:
    """Service for detecting relevant columns in sales data files."""
    
    # Column keyword mappings
    CUSTOMER_KEYWORDS = [
        'customer', 'client', 'account', 'customer_name', 'client_name',
        'account_name', 'buyer', 'shop', 'store', 'outlet', 'party', 'company'
    ]
    
    AMOUNT_KEYWORDS = [
        'amount', 'total', 'value', 'net', 'net_amount', 'gross_amount',
        'grand_total', 'invoice_amount', 'sales_amount', 'sales_value',
        'net sales', 'line total', 'subtotal', 'total_value', 'revenue', 'sale'
    ]
    
    SALESMAN_KEYWORDS = [
        'salesman', 'sales_person', 'salesperson', 'sales man', 'rep', 'agent',
        'sales rep', 'representative', 'sales representative', 'sales executive',
        'seller', 'owner', 'employee', 'staff', 'rep name', 'agent name',
        'salesman name', 'user'
    ]
    
    PRODUCT_KEYWORDS = [
        'product', 'product_name', 'item', 'item_name', 'sku', 'item code',
        'product code', 'material', 'material name', 'article', 'description',
        'itemcode', 'code', 'brand', 'brandname', 'brand_name', 'brand name'
    ]
    
    DATE_KEYWORDS = [
        'date', 'transaction_date', 'order_date', 'visitdate', 'invoice_date',
        'invoicedate', 'doc_date', 'posting date', 'created_at', 'timestamp',
        'sale date', 'visit date'
    ]
    
    QUANTITY_KEYWORDS = ['quantity', 'qty', 'units', 'unit', 'pcs']
    
    PRICE_KEYWORDS = [
        'price', 'unit_price', 'rate', 'unit rate', 'unitprice',
        'selling_price', 'list_price'
    ]
    
    BLOCKLIST = {
        'salesman', 'sales_person', 'sales man', 'rep', 'agent', 'sales rep'
    }
    
    def find_column(
        self,
        df: pd.DataFrame,
        keywords: list[str]
    ) -> str | None:
        """
        Find column by matching keywords using multiple strategies.
        
        Prefer exact/word-bound matches; fall back to safe substring if needed.
        Returns the first best match.
        
        Args:
            df: DataFrame to search in
            keywords: List of keywords to match against column names
            
        Returns:
            Column name if found, None otherwise
        """
        cols = list(df.columns)
        lower = {c: c.lower() for c in cols}
        
        # Strategy 1: Exact match
        for kw in keywords:
            for c in cols:
                if lower[c] == kw.lower():
                    return c
        
        # Strategy 2: Word-boundary match
        for kw in keywords:
            pat = re.compile(rf'\b{re.escape(kw.lower())}\b')
            for c in cols:
                if pat.search(lower[c]):
                    return c
        
        # Strategy 3: Safe substring match (avoid colliding words)
        for kw in keywords:
            k = kw.lower()
            for c in cols:
                lc = lower[c]
                if k in lc and lc not in self.BLOCKLIST:
                    return c
        
        # Strategy 4: Fuzzy matching with type awareness
        try:
            numeric_kw = {
                'amount', 'sales', 'revenue', 'total', 'value', 'quantity',
                'qty', 'units', 'price', 'unit_price', 'rate', 'net', 'gross',
                'subtotal'
            }
            want_date = any(
                ('date' in kw.lower()) or ('time' in kw.lower())
                for kw in keywords
            )
            want_numeric = any(
                any(t in kw.lower() for t in numeric_kw) for kw in keywords
            )
            
            # Filter candidates by type
            if want_date:
                cand = []
                for c in cols:
                    try:
                        s = pd.to_datetime(df[c], errors='coerce')
                        if s.notna().mean() > 0.6:
                            cand.append(c)
                    except Exception:
                        pass
                if not cand:
                    cand = cols
            elif want_numeric:
                cand = []
                for c in cols:
                    try:
                        s = pd.to_numeric(df[c], errors='coerce')
                        if s.notna().sum() > 0:
                            cand.append(c)
                    except Exception:
                        pass
                if not cand:
                    cand = cols
            else:
                cand = [c for c in cols if df[c].dtype == 'object'] or cols
            
            # Fuzzy match within candidates
            def _norm(s: str) -> str:
                return re.sub(r'[^a-z0-9]+', '', str(s).lower())
            
            best = (None, 0.0)
            for kw in keywords:
                nk = _norm(kw)
                for c in cand:
                    sc = difflib.SequenceMatcher(None, nk, _norm(c)).ratio()
                    if sc > best[1]:
                        best = (c, sc)
            
            if best[0] and best[1] >= 0.72:
                return best[0]
        except Exception:
            pass
        
        return None
    
    def detect_all_columns(
        self,
        df: pd.DataFrame,
        override_columns: dict[str, str] | None = None
    ) -> dict[str, str | None]:
        """
        Detect all relevant columns using keyword mappings.
        
        Args:
            df: DataFrame to analyze
            override_columns: User-provided column mappings to override detection
            
        Returns:
            Dictionary mapping column types to column names
        """
        columns = {
            'customer': self.find_column(df, self.CUSTOMER_KEYWORDS),
            'amount': self.find_column(df, self.AMOUNT_KEYWORDS),
            'salesman': self.find_column(df, self.SALESMAN_KEYWORDS),
            'product': self.find_column(df, self.PRODUCT_KEYWORDS),
            'date': self.find_column(df, self.DATE_KEYWORDS),
            'quantity': self.find_column(df, self.QUANTITY_KEYWORDS),
            'price': self.find_column(df, self.PRICE_KEYWORDS),
        }
        
        # Apply overrides
        if override_columns:
            for k, v in override_columns.items():
                if v and v in df.columns and k in columns:
                    columns[k] = v
        
        return columns


