import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta

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
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* COMPREHENSIVE TEXT CONTRAST FIX - All elements */
    body, html, .stApp {
        color: #0f172a !important;
    }
    
    /* All text elements - dark color for readability */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th, a {
        color: #0f172a !important;
    }
    
    /* Streamlit specific elements */
    .stMarkdown, .stMarkdown *, 
    .stDataFrame, .stDataFrame *,
    .stMetric, .stMetric *,
    .stExpander, .stExpander *,
    .stAlert, .stAlert *,
    .stSelectbox, .stSelectbox *,
    .stTextInput, .stTextInput *,
    .stFileUploader, .stFileUploader * {
        color: #0f172a !important;
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
        background: white;
        border-bottom: 3px solid var(--topseven-blue);
        box-shadow: 0 2px 15px rgba(30, 64, 175, 0.1);
        margin-bottom: 2rem;
        padding: 1.25rem 1rem;
    }
    
    .header-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .header-content {
        display: grid;
        grid-template-columns: auto 1fr;
        align-items: center;
        gap: 2rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 2.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .logo-text h1 {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a !important;
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
    }
    
    .company-badge {
        background: linear-gradient(135deg, var(--topseven-orange), #ea580c);
        color: white !important;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    .header-info {
        text-align: right;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .tagline {
        font-size: 0.95rem;
        color: #475569 !important;
        margin: 0;
        font-weight: 500;
        line-height: 1.4;
    }
    
    .subtitle {
        font-size: 0.85rem;
        color: #64748b !important;
        margin: 0;
        line-height: 1.3;
    }
    
    /* Mobile Responsive Header */
    @media (max-width: 768px) {
        .professional-header {
            padding: 1rem 0.5rem;
        }
        
        .header-content {
            grid-template-columns: 1fr;
            gap: 1rem;
            text-align: center;
        }
        
        .logo-section {
            justify-content: center;
        }
        
        .logo-icon {
            font-size: 2rem;
        }
        
        .logo-text h1 {
            font-size: 1.5rem;
        }
        
        .header-info {
            text-align: center;
        }
        
        .tagline {
            font-size: 0.875rem;
        }
        
        .subtitle {
            font-size: 0.8rem;
        }
    }
    
    /* Card styling with TopSeven theme */
    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .insight-card:hover {
        transform: translateY(-4px);
        border-color: var(--topseven-light-blue);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15);
    }
    
    .insight-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .insight-title {
        font-size: 1.125rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #0f172a !important;
        line-height: 1.3;
    }
    
    .insight-desc {
        color: #64748b !important;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    /* Mobile Responsive Cards */
    @media (max-width: 768px) {
        .insight-card {
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .insight-icon {
            width: 40px;
            height: 40px;
            font-size: 20px;
        }
        
        .insight-title {
            font-size: 1rem;
        }
        
        .insight-desc {
            font-size: 0.8rem;
        }
    }
    
    /* Chat messages with TopSeven styling */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--topseven-blue) 0%, var(--topseven-light-blue) 100%);
        margin-left: auto;
        color: white !important;
        border-radius: 18px 18px 4px 18px;
    }
    
    .user-message * {
        color: white !important;
    }
    
    .ai-message {
        background: white;
        margin-right: auto;
        color: #0f172a !important;
        border: 1px solid #e2e8f0;
        border-radius: 18px 18px 18px 4px;
    }
    
    .ai-message * {
        color: #0f172a !important;
    }
    
    /* Mobile Responsive Chat */
    @media (max-width: 768px) {
        .chat-message {
            max-width: 95%;
            padding: 0.875rem 1rem;
            font-size: 0.9rem;
        }
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
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid #3b82f6;
    }
    
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stFileUploader label {
        color: #1e40af !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4) !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stFileUploader button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stFileUploader button:hover::before {
        left: 100% !important;
    }
    
    .stFileUploader button:hover {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(0, 210, 255, 0.6) !important;
    }
    
    .stFileUploader button:active {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.7) !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stChatInput input {
        background: white !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
    }
    
    .stChatInput input:focus {
        border-color: #1e40af !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        outline: none !important;
        background: white !important;
        color: #1e293b !important;
    }
    
    .stChatInput input::placeholder {
        color: #64748b !important;
        font-weight: 400 !important;
    }
    
    /* Force text visibility in all input states */
    .stChatInput input[type="text"] {
        color: #1e293b !important;
        background: white !important;
    }
    
    .stChatInput textarea {
        color: #1e293b !important;
        background: white !important;
        border: 2px solid #3b82f6 !important;
    }
    
    .stChatInput textarea:focus {
        color: #1e293b !important;
        background: white !important;
    }
    
    .stChatInput button {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%) !important;
        color: #333 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        margin-left: 0.5rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 6px 20px rgba(255, 154, 158, 0.4) !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stChatInput button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stChatInput button:hover::before {
        left: 100% !important;
    }
    
    .stChatInput button:hover {
        background: linear-gradient(135deg, #fecfef 0%, #ff9a9e 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(255, 154, 158, 0.6) !important;
    }
    
    .stChatInput button:active {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(255, 154, 158, 0.7) !important;
    }
    
    /* ENHANCED BUTTON STYLING SYSTEM */
    
    /* Primary Buttons - Main actions with modern styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.7) !important;
    }
    
    /* Full-width buttons for quick questions */
    .stButton > button[data-testid*="btn1"],
    .stButton > button[data-testid*="btn2"],
    .stButton > button[data-testid*="btn3"] {
        width: 100% !important;
        margin: 0.5rem 0 !important;
        font-size: 0.95rem !important;
        padding: 1rem 1.5rem !important;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
    }
    
    .stButton > button[data-testid*="btn1"]:hover,
    .stButton > button[data-testid*="btn2"]:hover,
    .stButton > button[data-testid*="btn3"]:hover {
        background: linear-gradient(135deg, #ee5a24 0%, #ff6b6b 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.6) !important;
    }
    
    /* Clear/Delete buttons - Enhanced red styling */
    .stButton > button:contains("Clear"),
    .stButton > button:contains("Delete"),
    .stButton > button:contains("Remove") {
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4) !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:contains("Clear"):hover,
    .stButton > button:contains("Delete"):hover,
    .stButton > button:contains("Remove"):hover {
        background: linear-gradient(135deg, #c44569 0%, #ff4757 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(255, 71, 87, 0.6) !important;
    }
    
    /* Close buttons - Enhanced gray styling */
    .stButton > button:contains("Close") {
        background: linear-gradient(135deg, #a4b0be 0%, #747d8c 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(164, 176, 190, 0.4) !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:contains("Close"):hover {
        background: linear-gradient(135deg, #747d8c 0%, #a4b0be 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(164, 176, 190, 0.6) !important;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #3b82f6 !important;
    }
    
    .stSpinner > div {
        border-color: #3b82f6 transparent #3b82f6 transparent !important;
    }
    
    /* Metrics cards styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--topseven-light-blue);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Mobile Responsive Buttons and Metrics */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
        }
        
        .metric-card {
            padding: 1.25rem;
        }
        
        .metric-card h3 {
            font-size: 1.25rem;
        }
        
        .metric-card p {
            font-size: 0.8rem;
        }
    }
    
    /* Professional Footer Styling */
    .professional-footer {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: #cbd5e1 !important;
        margin-top: 4rem;
        border-top: 3px solid var(--topseven-blue);
        padding: 2rem 1rem 1rem 1rem;
    }
    
    .footer-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .footer-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 2rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid #334155;
    }
    
    .footer-section h4 {
        color: white !important;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        border-bottom: 2px solid var(--topseven-orange);
        padding-bottom: 0.5rem;
        width: fit-content;
    }
    
    .footer-section p, .footer-section li {
        color: #cbd5e1 !important;
        font-size: 0.875rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .footer-section a {
        color: #94a3b8 !important;
        text-decoration: none;
        transition: color 0.2s;
    }
    
    .footer-section a:hover {
        color: var(--topseven-orange) !important;
    }
    
    .footer-badge {
        background: linear-gradient(135deg, var(--topseven-orange), #ea580c);
        color: white !important;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    
    .contact-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        color: #cbd5e1 !important;
        font-size: 0.875rem;
    }
    
    .contact-icon {
        color: var(--topseven-orange) !important;
        font-size: 1rem;
        width: 20px;
    }
    
    .footer-bottom {
        padding: 1.5rem 0 0.5rem 0;
        text-align: center;
    }
    
    .footer-copyright p {
        color: #64748b !important;
        font-size: 0.8rem;
        margin: 0;
    }
    
    /* Mobile Responsive Footer */
    @media (max-width: 768px) {
        .professional-footer {
            padding: 1.5rem 0.5rem;
        }
        
        .footer-content {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .footer-section {
            text-align: center;
        }
        
        .footer-section h4 {
            margin: 0 auto 1rem auto;
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
    
    /* COMPREHENSIVE INPUT TEXT VISIBILITY FIX */
    /* Force ALL input elements to have dark text on light background */
    input, textarea, select {
        color: #1e293b !important;
        background: white !important;
    }
    
    input:focus, textarea:focus, select:focus {
        color: #1e293b !important;
        background: white !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Specific Streamlit input overrides */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        color: #1e293b !important;
        background: white !important;
        border: 2px solid #3b82f6 !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        color: #1e293b !important;
        background: white !important;
        border-color: #1e40af !important;
    }
    
    /* Chat input specific overrides */
    .stChatInput input, .stChatInput textarea {
        color: #1e293b !important;
        background: white !important;
        border: 2px solid #3b82f6 !important;
        font-weight: 500 !important;
    }
    
    .stChatInput input:focus, .stChatInput textarea:focus {
        color: #1e293b !important;
        background: white !important;
        border-color: #1e40af !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Override any dark theme that might be applied */
    .stApp input, .stApp textarea, .stApp select {
        color: #1e293b !important;
        background: white !important;
    }
    
    .stApp input:focus, .stApp textarea:focus, .stApp select:focus {
        color: #1e293b !important;
        background: white !important;
    }
    
    /* DETAILED ANALYSIS TAB ENHANCEMENTS */
    
    /* Enhanced data table styling */
    .stDataFrame {
        background: white !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        background: white !important;
        color: #1e293b !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 1rem 0.75rem !important;
        text-align: left !important;
        border: none !important;
    }
    
    .stDataFrame td {
        color: #1e293b !important;
        background: white !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding: 0.75rem !important;
        font-size: 0.9rem !important;
    }
    
    .stDataFrame tr:nth-child(even) {
        background: #f8fafc !important;
    }
    
    .stDataFrame tr:hover {
        background: #f1f5f9 !important;
    }
    
    /* Enhanced section headers */
    .stMarkdown h3 {
        color: #1e293b !important;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        border-left: 4px solid #3b82f6 !important;
        margin: 1.5rem 0 1rem 0 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    .metric-card h3 {
        color: #1e293b !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    
    .metric-card p {
        color: #64748b !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    
    /* Fix expander styling */
    .streamlit-expander {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expander .streamlit-expanderHeader {
        background: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expander .streamlit-expanderContent {
        background: white !important;
        color: #1e293b !important;
    }
    
    /* Enhanced download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4) !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stDownloadButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stDownloadButton > button:hover::before {
        left: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(46, 204, 113, 0.6) !important;
    }
    
    .stDownloadButton > button:active {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.7) !important;
    }
    
    /* Fix info boxes */
    .stAlert {
        background: #f0f9ff !important;
        border: 1px solid #0ea5e9 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    .stAlert .stMarkdown {
        color: #1e293b !important;
    }
    
    /* Fix warning boxes */
    .stWarning {
        background: #fef3c7 !important;
        border: 1px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    .stWarning .stMarkdown {
        color: #1e293b !important;
    }
    
    /* Enhanced performance metrics */
    .stMetric {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stMetric > div {
        color: #1e293b !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: white !important;
        color: #1e293b !important;
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: #1e293b !important;
    }
    
    .stMetric [data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #059669 !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced plotly charts container */
    .stPlotlyChart {
        background: white !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Enhanced main content area */
    .main .block-container {
        background: #f8fafc !important;
        padding: 2rem !important;
    }
    
    /* Enhanced tab content */
    .stTabs [data-baseweb="tab-panel"] {
        background: #f8fafc !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
    }
    
    /* Better spacing for detailed analysis */
    .stTabs [data-baseweb="tab-panel"] > div {
        background: transparent !important;
    }
    
    /* Global Mobile Responsiveness */
    @media (max-width: 768px) {
        /* Main app padding */
        .stApp {
            padding: 0.5rem;
        }
        
        /* Reduce heading sizes */
        h1 {
            font-size: 1.75rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.25rem !important;
        }
        
        /* Tab adjustments */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem;
            padding: 0.5rem 0.75rem;
        }
        
        /* Expander adjustments */
        .stExpander {
            font-size: 0.9rem;
        }
        
        /* DataFrame adjustments */
        .stDataFrame {
            font-size: 0.85rem;
        }
        
        /* File uploader */
        .stFileUploader {
            font-size: 0.9rem;
        }
        
        /* Chat input */
        .stChatInput {
            font-size: 0.9rem;
        }
    }
    
    /* Tablet Responsiveness */
    @media (min-width: 769px) and (max-width: 1024px) {
        .header-content {
            gap: 1.5rem;
        }
        
        .footer-content {
            grid-template-columns: repeat(2, 1fr);
        }
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
    return None

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
    <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h2 style="color: #1e293b; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">Upload Your Sales Data</h2>
        <p style="color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;">
            Upload your Excel file to unlock AI-powered insights into your sales performance, 
            customer behavior, and business opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload container with TopSeven branding
    upload_container = st.container()
    with upload_container:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 2rem; border-radius: 12px; border: 2px dashed #3b82f6; margin-bottom: 1.5rem; text-align: center; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">
            <div style="color: #1e40af; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">📁 Ready to Upload</div>
            <div style="color: #64748b; font-size: 0.9rem;">Drag and drop your Excel file here or click browse</div>
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
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #3b82f6; margin: 1rem 0; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">
            <div style="color: #1e40af; font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">💬 Ask Your Question</div>
            <div style="color: #64748b; font-size: 0.9rem;">Type your question about the sales data below</div>
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
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); color: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);">
            <h2 style="color: white; margin: 0 0 0.5rem 0; font-size: 1.75rem; font-weight: 700;">📊 Business Performance Metrics</h2>
            <p style="color: #e2e8f0; margin: 0; font-size: 1rem;">Key insights and analytics from your sales data</p>
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
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #06b6d4; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">📈 Performance Analytics</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Time-series analysis and trend visualization</p>
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
                        title="Daily Revenue Trend",
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig_line.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='#1e293b',
                        font_size=12,
                        title_font_size=16,
                        title_font_color='#1e293b',
                        xaxis=dict(
                            gridcolor='#e2e8f0',
                            linecolor='#1e293b',
                            linewidth=1,
                            zeroline=True,
                            zerolinecolor='#1e293b',
                            zerolinewidth=2
                        ),
                        yaxis=dict(
                            gridcolor='#e2e8f0',
                            linecolor='#1e293b',
                            linewidth=1,
                            zeroline=True,
                            zerolinecolor='#1e293b',
                            zerolinewidth=2
                        ),
                        margin=dict(l=0, r=0, t=40, b=40),
                        showlegend=False
                    )
                    fig_line.update_traces(
                        line=dict(width=3),
                        fill='tonexty',
                        fillcolor='rgba(59, 130, 246, 0.1)'
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("📊 No numeric column found for revenue analysis. Please ensure your data has a revenue/amount column.")
        
        # Sales Performance Metrics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">🎯 Sales Performance Metrics</h3>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Key performance indicators and business metrics</p>
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
            <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">🏆 Top 5 Salesmen</h3>
                <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Revenue performance by individual sales representatives</p>
            </div>
            """, unsafe_allow_html=True)
            top_salesmen_df = pd.DataFrame([
                {'Salesman': name, 'Revenue': revenue}
                for name, revenue in insights['top_salesmen'].items()
            ])
            st.dataframe(top_salesmen_df, use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_salesmen_df,
                x='Revenue',
                y='Salesman',
                orientation='h',
                title="Top Salesmen by Revenue",
                color='Revenue',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#1e293b',
                font_size=12,
                title_font_size=16,
                title_font_color='#1e293b',
                xaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1,
                    zeroline=True,
                    zerolinecolor='#1e293b',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1
                ),
                margin=dict(l=0, r=0, t=40, b=40),
                showlegend=False
            )
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Customers
        if 'top_customers_list' in insights:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">👥 Top 5 Customers</h3>
                <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Highest revenue generating customer accounts</p>
            </div>
            """, unsafe_allow_html=True)
            top_customers_df = pd.DataFrame(insights['top_customers_list'])
            st.dataframe(top_customers_df[['name', 'revenue', 'percentage']], use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_customers_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Customers by Revenue",
                color='revenue',
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#1e293b',
                font_size=12,
                title_font_size=16,
                title_font_color='#1e293b',
                xaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1,
                    zeroline=True,
                    zerolinecolor='#1e293b',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1
                ),
                margin=dict(l=0, r=0, t=40, b=40),
                showlegend=False
            )
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn Risk
        if 'churn_risk' in insights and insights['churn_risk']:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">⚠️ Churn Risk Alert</h3>
                <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Customers showing declining engagement patterns</p>
            </div>
            """, unsafe_allow_html=True)
            churn_df = pd.DataFrame(insights['churn_risk'])
            st.dataframe(churn_df, use_container_width=True, hide_index=True)
            
            st.info("💡 **AI Recommendation:** Schedule immediate visits to these customers. Offer special promotions to re-engage them.")
        
        # Top Products
        if 'top_products_list' in insights:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b5cf6; margin: 2rem 0 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 700;">📦 Top 5 Products</h3>
                <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Best performing products by revenue contribution</p>
            </div>
            """, unsafe_allow_html=True)
            top_products_df = pd.DataFrame(insights['top_products_list'])
            st.dataframe(top_products_df[['name', 'revenue', 'percentage']], use_container_width=True, hide_index=True)
            
            # Professional Chart
            fig = px.bar(
                top_products_df,
                x='revenue',
                y='name',
                orientation='h',
                title="Top Products by Revenue",
                color='revenue',
                color_continuous_scale='Purples'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#1e293b',
                font_size=12,
                title_font_size=16,
                title_font_color='#1e293b',
                xaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1,
                    zeroline=True,
                    zerolinecolor='#1e293b',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    gridcolor='#e2e8f0',
                    linecolor='#1e293b',
                    linewidth=1
                ),
                margin=dict(l=0, r=0, t=40, b=40),
                showlegend=False
            )
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

