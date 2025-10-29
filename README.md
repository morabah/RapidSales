# Rapid Sales AI Demo

AI-powered sales data analysis tool using Google Gemini and Streamlit.

## Features

- 📤 **Excel Upload**: Upload your sales data in Excel format
- 🤖 **AI Insights**: Get automatic insights and analytics
- 💬 **Natural Language Chat**: Ask questions about your data in plain English
- 📊 **Interactive Visualizations**: Charts and graphs for data analysis
- 📥 **Export Data**: Download results and chat history

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key

1. Visit https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### 3. Configure API Key

**For Local Development:**

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

**For Streamlit Cloud Deployment:**

Add the secret in Streamlit Cloud settings (see Deployment section)

### 4. Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment on Streamlit Community Cloud

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`

3. **Add Secrets**
   - Click "Advanced settings"
   - Add secret:
     ```
     GEMINI_API_KEY = "your-actual-api-key"
     ```
   - Click "Deploy"

4. **Your app is live!** 🎉

## Sample Questions to Try

- "Who is the best salesman this month?"
- "Show me top 5 customers by revenue"
- "Which salesmen need coaching?"
- "What's the average order value?"
- "Find customers with declining sales"
- "Which products are underperforming?"
- "Compare performance of all salesmen"
- "What's the revenue trend?"

## Excel File Format

Your Excel file should include columns like:
- **Salesman/Sales Rep**: Name of the salesperson
- **Customer/Client**: Customer name
- **Product/Item**: Product name
- **Amount/Sales/Revenue**: Sales amount (numeric)
- **Date**: Transaction date
- **Quantity**: Units sold (optional)

The app will automatically detect these columns even if they have different names!

## Features Explained

### AI Insights Cards
- **Demand Forecasting**: Predicts customer ordering patterns
- **Smart Route Planning**: Optimizes sales routes
- **Dynamic Pricing**: Suggests optimal pricing
- **Churn Risk Detection**: Identifies at-risk customers
- **Intelligent Target Setting**: Data-driven sales targets
- **Customer Segmentation**: Groups customers by value

### Chat Interface
Ask natural language questions and get AI-powered answers based on your data.

### Detailed Analysis
View comprehensive metrics, top performers, charts, and export your results.

## Requirements

- Python 3.8+
- Streamlit 1.31.0
- Google Generative AI SDK
- Pandas
- OpenPyXL (for Excel support)
- Plotly (for visualizations)

## Cost

- **Streamlit Cloud**: 100% Free
- **Google Gemini API**: Free tier (1,500 requests/day)

## Support

For issues or questions, please check the Streamlit documentation or Google Gemini API documentation.

---

**Rapid Sales AI Demo** - Transforming sales operations with AI

# Rapid Sales AI Demo

AI-powered sales data analysis tool using Google Gemini and Streamlit.

## 🚀 Features

- 📤 **Excel Upload**: Upload your sales data in Excel format
- 🤖 **AI Insights**: Get automatic insights and analytics
- 💬 **Natural Language Chat**: Ask questions about your data in plain English
- 📊 **Interactive Visualizations**: Charts and graphs for data analysis
- 📥 **Export Data**: Download results and chat history
- 🎨 **Professional UI**: Dark theme matching the original design

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key

1. Visit https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### 3. Configure API Key

**For Local Development:**

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

**For Streamlit Cloud Deployment:**

Add the secret in Streamlit Cloud settings (see Deployment section)

### 4. Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deployment on Streamlit Community Cloud

1. **Push code to GitHub** (already done!)
2. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `morabah/RapidSales`
   - Set main file path: `app.py`

3. **Add Secrets**
   - Click "Advanced settings"
   - Add secret:
     ```
     GEMINI_API_KEY = "your-actual-api-key"
     ```
   - Click "Deploy"

4. **Your app is live!** 🎉

## 💬 Sample Questions to Try

- "Who is the best salesman this month?"
- "Show me top 5 customers by revenue"
- "Which salesmen need coaching?"
- "What's the average order value?"
- "Find customers with declining sales"
- "Which products are underperforming?"
- "Compare performance of all salesmen"
- "What's the revenue trend?"

## 📊 Excel File Format

Your Excel file should include columns like:
- **Salesman/Sales Rep**: Name of the salesperson
- **Customer/Client**: Customer name
- **Product/Item**: Product name
- **Amount/Sales/Revenue**: Sales amount (numeric)
- **Date**: Transaction date
- **Quantity**: Units sold (optional)

The app will automatically detect these columns even if they have different names!

## 🎯 Features Explained

### AI Insights Cards
- **Demand Forecasting**: Predicts customer ordering patterns
- **Smart Route Planning**: Optimizes sales routes
- **Dynamic Pricing**: Suggests optimal pricing
- **Churn Risk Detection**: Identifies at-risk customers
- **Intelligent Target Setting**: Data-driven sales targets
- **Customer Segmentation**: Groups customers by value

### Chat Interface
Ask natural language questions and get AI-powered answers based on your data.

### Detailed Analysis
View comprehensive metrics, top performers, charts, and export your results.

## 📋 Requirements

- Python 3.8+
- Streamlit 1.31.0+
- Google Generative AI SDK
- Pandas
- OpenPyXL (for Excel support)
- Plotly (for visualizations)

## 💰 Cost

- **Streamlit Cloud**: 100% Free
- **Google Gemini API**: Free tier (1,500 requests/day)

## 🔗 Links

- **GitHub Repository**: https://github.com/morabah/RapidSales
- **Streamlit Cloud**: Deploy from the repository
- **Google Gemini**: https://makersuite.google.com/app/apikey

## 📞 Support

For issues or questions, please check the Streamlit documentation or Google Gemini API documentation.

---

**Rapid Sales AI Demo** - Transforming sales operations with AI
