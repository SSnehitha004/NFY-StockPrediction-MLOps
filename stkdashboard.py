import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf 
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import time
import warnings
import logging 

import mlflow
import mlflow.pyfunc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
    print(f"DEBUG: Added '{current_script_dir}' to sys.path") 

COMPANY_NAME = 'Infosys' 
st.set_page_config(page_title=f"{COMPANY_NAME} Stock Predictor", layout="wide")

try:
    from utils import (
        fetch_stock_data,
        fetch_news,
        preprocess_and_feature_engineer,
        get_sentiment_context_vader,
        talib, 
        SentimentIntensityAnalyzer 
    )
    st.sidebar.success("Utils loaded successfully.")
except ImportError as e:
    st.error(f"Fatal Error: Could not import from utils.py: {e}")
    st.error("Ensure utils.py is in the same directory and has no errors.")
    st.stop() 
except Exception as e_gen:
    st.error(f"Fatal Error: An unexpected error occurred during utils import: {e_gen}")
    st.stop()

load_dotenv()
NEWS_API_KEY = os.getenv('NEWSAPI_KEY')
TICKER = 'INFY.NS'

NEWS_SOURCES = 'google-news-in,the-times-of-india,the-hindu,business-standard,reuters'
MODEL_REGISTRY_NAME = "StockPredictorInfosys" 

FETCH_STOCK_HISTORY_PREDICT = 90
FETCH_STOCK_HISTORY_CHART = 180
FETCH_NEWS_DAYS_DISPLAY = 3



@st.cache_data(ttl=60 * 15)
def cached_fetch_stock_data_wrapper(ticker, days_history):
    """Cached wrapper for fetch_stock_data from utils."""
    logging.info(f"CACHE_DATA: Running fetch_stock_data for {ticker} ({days_history} days)")
    return fetch_stock_data(ticker, days_history)

@st.cache_data(ttl=60 * 30) 
def cached_fetch_news_wrapper(api_key, query, sources, days_ago):
    """Cached wrapper for fetch_news from utils."""
    logging.info(f"CACHE_DATA: Running fetch_news for {query} ({days_ago} days)")
    return fetch_news(api_key, query, sources, days_ago)


@st.cache_resource(ttl=60*10)
def load_model_from_mlflow_alias(model_name, alias): 
    """Loads the model version associated with the alias from MLflow Model Registry."""
    logging.info(f"CACHE_RESOURCE: Loading model '{model_name}' alias '@{alias}' from MLflow...")
    model_info = {"model": None, "version": "N/A", "run_id": "N/A"}
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model_info["model"] = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model loaded: {model_uri}")
        try:
             client = mlflow.tracking.MlflowClient()
             model_version_details = client.get_model_version_by_alias(model_name, alias)
             model_info["version"] = model_version_details.version
             model_info["run_id"] = model_version_details.run_id
             logging.info(f"Alias '{alias}' points to: Version={model_info['version']}, RunID={model_info['run_id']}")
        except Exception as meta_err:
             logging.warning(f"Could not get model version details for alias '{alias}': {meta_err}")
        return model_info
    except Exception as e:
        logging.error(f"MLflow model loading failed for alias '{alias}': {e}", exc_info=True)
        return model_info 

def run_prediction(model_dict):
    """Fetches data, preprocesses, predicts using MLflow model, gets context."""
    model = model_dict.get("model") 
    if model is None: return {"error": "Model object not available for prediction."}

    with st.spinner(f'Fetching latest data for {TICKER}...'):
        stock_data_raw = cached_fetch_stock_data_wrapper(TICKER, FETCH_STOCK_HISTORY_PREDICT)
        if stock_data_raw.empty: return {"error": f"Failed to fetch sufficient stock data for {TICKER}."}
        last_stock_date = stock_data_raw.index.max().date()

        news_articles_context = cached_fetch_news_wrapper(NEWS_API_KEY, COMPANY_NAME, NEWS_SOURCES, days_ago=FETCH_NEWS_DAYS_DISPLAY)
        context_text = ""
        relevant_context_dates = [last_stock_date, last_stock_date - timedelta(days=1)]
        if news_articles_context:
            texts_for_context = []
            for art in news_articles_context:
                 try:
                     pub_date = pd.to_datetime(art.get('publishedAt')).tz_localize(None).date()
                     if pub_date in relevant_context_dates:
                          title = art.get('title', '')
                          desc = art.get('description', '')
                          texts_for_context.append(f"{title}. {desc}".strip())
                 except: continue
            context_text = " || ".join(texts_for_context)

    with st.spinner('Generating prediction features...'):
        X_latest_full, _, _ = preprocess_and_feature_engineer(stock_data_raw.copy(), pd.DataFrame(), company_name=COMPANY_NAME)
        if X_latest_full is None or X_latest_full.empty: return {"error": "Could not generate features."}
        try: last_features = X_latest_full.iloc[[-1]]; last_feature_date = last_features.index[0].date()
        except IndexError: return {"error": "Could not extract latest features."}
        if last_features.isnull().values.any(): st.warning("NaN values found in latest features."); 

    with st.spinner('Making prediction...'):
        try:
            prediction_result_array = model.predict(last_features)
            prediction = prediction_result_array[0]
            confidence = 0.5 
            predict_proba_available = False
            if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
                 predict_proba_available = True

            if predict_proba_available:
                try:
                    prediction_proba = model._model_impl.predict_proba(last_features)[0]
                    if prediction < len(prediction_proba): confidence = prediction_proba[prediction]
                    else: st.warning(f"Pred index OOB for proba ({prediction} vs len {len(prediction_proba)})")
                except Exception as proba_err: st.warning(f"Could not get prediction probabilities: {proba_err}")
            else: st.warning("Loaded model does not expose 'predict_proba'. Confidence is default 50%.")

            predicted_direction = "Up" if prediction == 1 else "Down"
            prediction_for_date = last_feature_date + timedelta(days=1)
            while prediction_for_date.weekday() >= 5: prediction_for_date += timedelta(days=1)

            vader_summary, vader_highlights = get_sentiment_context_vader(context_text)

            return { "prediction": predicted_direction, "confidence": confidence, "prediction_date": prediction_for_date,
                     "last_feature_date": last_feature_date, "vader_summary": vader_summary, "vader_highlights": vader_highlights, "error": None }
        except Exception as e:
            logging.error(f"Error during prediction calculation: {e}", exc_info=True)
            st.text(f"Features passed to predict:\n{last_features.to_string()}")
            return {"error": f"Prediction calculation failed: {e}"}


st.title(f"{COMPANY_NAME} ({TICKER}) Stock Movement Predictor")
st.caption(f"Using MLflow Model & VADER Context (App Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

st.sidebar.header("Configuration")
selected_alias = st.sidebar.selectbox("Select Model Alias to Load", ["production", "staging", "latest"], index=0) 
st.sidebar.info(f"Attempting to load model: '{MODEL_REGISTRY_NAME}@{selected_alias}'")


model_info = load_model_from_mlflow_alias(MODEL_REGISTRY_NAME, selected_alias)
model_object = model_info.get("model")

if model_object:
    st.sidebar.success(f"Model Loaded: Alias '{selected_alias}' -> Vsn {model_info.get('version', 'N/A')}")
else:
    st.sidebar.error(f"Failed to load model with alias '{selected_alias}'.")
    st.sidebar.info("Ensure model alias exists and MLflow server is running.")

st.header(f"📈 Prediction (Using Model Alias: @{selected_alias})")
if model_object:
    if st.button("Run Prediction", key="predict_button", type="primary"):
        prediction_result = run_prediction(model_info)

        if prediction_result.get("error"):
            st.error(prediction_result["error"])
        else:
            pred_date_str = prediction_result['prediction_date'].strftime('%Y-%m-%d')
            last_feature_date_str = prediction_result['last_feature_date'].strftime('%Y-%m-%d')
            cols = st.columns([1, 1, 2])

            with cols[0]:
                pred_direction = prediction_result["prediction"]
                st.metric(label=f"Predicted Direction ({pred_date_str})", value=pred_direction)
                if pred_direction == "Up": st.success("Indicator: ▲")
                else: st.error("Indicator: ▼")

            with cols[1]:
                st.metric(label="Prediction Confidence", value=f"{prediction_result['confidence']:.2%}")
                st.progress(prediction_result['confidence'])
            with cols[2]:
                st.info(f"Prediction based on data up to: **{last_feature_date_str}**")

            with st.expander("📰 Sentiment Context (VADER)"):
                st.markdown(f"**{prediction_result['vader_summary']}**")
                
                if prediction_result['vader_highlights']:
                    st.markdown("Key Headlines/Snippets:")
                    for h in prediction_result['vader_highlights']:
                        st.caption(f" - {h}") 
                
                else:
                    st.caption("No VADER highlights.")
                st.caption(f"(Analyzed news around {last_feature_date_str})")
    else:
        st.info(f"Click 'Run Prediction' to get forecast using the '@{selected_alias}' model.")
else:
    st.warning(f"Model '{MODEL_REGISTRY_NAME}@{selected_alias}' could not be loaded.")

st.divider() 

# Stock Chart Section
st.header(f"📊 Recent Stock Price ({TICKER})")
with st.spinner(f"Loading chart data..."):
    chart_data = cached_fetch_stock_data_wrapper(TICKER, FETCH_STOCK_HISTORY_CHART)
if not chart_data.empty:
    if 'Close' in chart_data.columns: st.line_chart(chart_data['Close']); st.caption("Adjusted Closing Price")
    else: st.warning("Could not find 'Close' column to plot chart.")
else: st.warning(f"Could not fetch data to display chart for {TICKER}.")

st.divider()

# Recent News Section
st.header(f"📰 Latest News for {COMPANY_NAME}")
with st.spinner(f"Fetching latest news..."):
     latest_news = cached_fetch_news_wrapper(NEWS_API_KEY, COMPANY_NAME, NEWS_SOURCES, days_ago=FETCH_NEWS_DAYS_DISPLAY)
if not latest_news: st.info(f"No recent news found.")
else:
    st.info(f"Displaying top {len(latest_news)} relevant news articles:")
    vader_available = SentimentIntensityAnalyzer is not None
    for article in latest_news:
        with st.expander(f"**{article.get('title', 'N/A')}** - _{article.get('source', {}).get('name', 'N/A')}_"):
            st.write(f"**Published:** {pd.to_datetime(article.get('publishedAt')).strftime('%Y-%m-%d %H:%M')}")
            st.write(article.get('description', 'N/A')); st.markdown(f"[Read More]({article.get('url', '#')})", unsafe_allow_html=True)
            if vader_available:
                 vad_text = f"{article.get('title', '')}. {article.get('description', '')}";
                 if vad_text.strip() and vad_text != '.':
                     try: st.caption(f"VADER Score: {SentimentIntensityAnalyzer().polarity_scores(vad_text)['compound']:.2f}")
                     except Exception: st.caption("VADER Score: Error")

st.markdown("---"); st.caption("Disclaimer: Informational purposes only. Not financial advice.")