import os
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
import joblib
from datetime import date, timedelta
import time
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    import talib
    logging.info("TA-Lib library found.")
except ImportError:
    logging.warning("TA-Lib library not found. TA-Lib indicators will be skipped.")
    talib = None
import nltk
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    logging.info("vaderSentiment library found.")
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        logging.info("Downloading VADER lexicon...")
        try: nltk.download('vader_lexicon')
        except Exception as e: logging.error(f"Could not download VADER lexicon: {e}"); SentimentIntensityAnalyzer = None
except ImportError:
    logging.warning("vaderSentiment library not found. Sentiment context feature will be disabled.")
    SentimentIntensityAnalyzer = None
FINBERT_MODEL = "ProsusAI/finbert"
COMPANY_NAME = 'Infosys'

def fetch_stock_data(ticker, days_history):
    """Fetches historical stock data, flattens MultiIndex immediately, and cleans."""
    logging.info(f"Fetching stock data for {ticker}...")
    end_date = date.today()
    start_date = end_date - timedelta(days=days_history)
    stock_df = pd.DataFrame()
    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        logging.info(f"yfinance download successful for {ticker}. Rows: {len(stock_df)}")
    except Exception as e:
        logging.error(f"Error during yfinance download call for {ticker}: {e}")
        return pd.DataFrame()

    if stock_df.empty:
        logging.warning(f"yfinance download for {ticker} returned an empty DataFrame.")
        return pd.DataFrame()

    logging.debug(f"Columns after download: {stock_df.columns.tolist()}")
    if isinstance(stock_df.columns, pd.MultiIndex):
        logging.info("Detected MultiIndex columns, flattening immediately...")
        stock_df.columns = stock_df.columns.get_level_values(0)
        logging.info(f"Columns after flattening: {stock_df.columns.tolist()}")
        duplicates = stock_df.columns.duplicated(keep='first')
        if duplicates.any():
            duplicated_names = stock_df.columns[duplicates].unique().tolist()
            logging.warning(f"Duplicate columns found AFTER flattening: {duplicated_names}. Keeping first occurrence.")
            stock_df = stock_df.loc[:, ~duplicates]
    else: logging.info("Columns are already flat.")

    required_cols_simple = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    missing_cols = [col for col in required_cols_simple if col not in stock_df.columns]
    if missing_cols:
        logging.error(f"Essential columns missing AFTER flattening for {ticker}: {missing_cols}. Available: {stock_df.columns.tolist()}")
        return pd.DataFrame()
    else:
        if 'Adj Close' in stock_df.columns:
             stock_df['Close'] = stock_df['Adj Close']
        elif 'Close' not in stock_df.columns:
             logging.error("Neither 'Close' nor 'Adj Close' found after flattening.")
             return pd.DataFrame()

        required_cols_simple = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols_final = [col for col in required_cols_simple if col not in stock_df.columns]
        if missing_cols_final:
             logging.error(f"Essential columns missing before NaN drop: {missing_cols_final}")
             return pd.DataFrame()

        logging.info(f"All essential columns found. Performing initial NaN drop...")
        initial_rows = len(stock_df)
        try:
            stock_df = stock_df.dropna(subset=required_cols_simple).copy()
            logging.info(f"Initial NaN drop complete. Removed {initial_rows - len(stock_df)} rows.")
        except Exception as e:
            logging.error(f"Error during initial dropna for {ticker}: {e}")
            return pd.DataFrame()
        if stock_df.empty:
            logging.warning(f"DataFrame empty after initial NaN drop for {ticker}.")
            return pd.DataFrame()

    logging.info(f"Successfully fetched, flattened, and initially cleaned {len(stock_df)} stock data points for {ticker}.")
    return stock_df


def fetch_news(api_key, query, sources, days_ago):
    """Fetches news articles mentioning the query from NewsAPI."""
    logging.info(f"Fetching news for '{query}' (last {days_ago} days)...")
    if not api_key: logging.error("NEWSAPI_KEY not found."); return []
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []
    fetch_end_date = date.today()
    fetch_start_date = fetch_end_date - timedelta(days=days_ago)
    api_limit_date = fetch_end_date - timedelta(days=30)
    if fetch_start_date < api_limit_date:
        logging.warning(f"Adjusting news start date from {fetch_start_date} to {api_limit_date} due to API limits.")
        fetch_start_date = api_limit_date
    logging.info(f"News fetch period: {fetch_start_date.strftime('%Y-%m-%d')} to {fetch_end_date.strftime('%Y-%m-%d')}")

    try:
        page = 1; fetched_count = 0; max_results_per_query = 100
        while fetched_count < max_results_per_query and page <= 5:
            articles = newsapi.get_everything(
                q=query, sources=sources,
                from_param=fetch_start_date.strftime('%Y-%m-%d'), to=fetch_end_date.strftime('%Y-%m-%d'),
                language='en', sort_by='publishedAt',
                page_size=min(100, max_results_per_query - fetched_count), page=page )
            if articles['status'] == 'ok':
                fetched_this_page = len(articles['articles'])
                if fetched_this_page == 0: break
                fetched_count += fetched_this_page; all_articles.extend(articles['articles'])
                if fetched_count >= max_results_per_query: logging.info(f"Reached free tier limit of {max_results_per_query} articles."); break
                page += 1; time.sleep(0.3)
            elif articles.get('code') == 'maximumResultsReached': logging.warning("NewsAPI limit reached."); break
            else: logging.error(f"NewsAPI Error: {articles.get('code')} - {articles.get('message', 'Unknown')}"); break
    except Exception as e: logging.error(f"Error during news fetch loop: {e}")

    logging.info(f"Fetched {len(all_articles)} total articles (before filtering).")
    filtered_articles = [] 
    seen_titles = set()
    for art in all_articles:
        title = art.get('title'); pub_date = art.get('publishedAt'); desc = art.get('description','') or ''
        if not title or title in seen_titles or not pub_date or not art.get('source'): continue
        if COMPANY_NAME.lower() in title.lower() or COMPANY_NAME.lower() in desc.lower():
             filtered_articles.append(art); seen_titles.add(title)
    logging.info(f"Filtered down to {len(filtered_articles)} relevant, unique articles.")
    return filtered_articles

def analyze_sentiment_finbert(articles):
    """Analyzes sentiment using FinBERT. Requires transformers & torch."""
    logging.info("Analyzing sentiment with FinBERT...")
    results = []
    if not articles: return pd.DataFrame(results, columns=['publishedAt', 'sentiment_score', 'sentiment_label', 'text'])

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL) # Use config variable
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading FinBERT model ({FINBERT_MODEL}): {e}")
        return pd.DataFrame(results, columns=['publishedAt', 'sentiment_score', 'sentiment_label', 'text'])

    processed_count = 0; total_articles = len(articles)
    logging.info(f"Starting FinBERT analysis for {total_articles} articles...")
    for i, article in enumerate(articles):
        title = article.get('title', ''); description = article.get('description', '')
        text_to_analyze = f"{title}. {description}".strip()
        if not text_to_analyze or text_to_analyze == '.': continue
        try:
            inputs = tokenizer(text_to_analyze, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad(): outputs = model(**inputs); logits = outputs.logits
            probs = torch.softmax(logits, dim=-1); sentiment_class = torch.argmax(probs, dim=-1).item()
            sentiment_score = probs[0][0].item() - probs[0][1].item() # Pos - Neg
            results.append({
                'publishedAt': pd.to_datetime(article['publishedAt']).tz_localize(None).date(),
                'sentiment_score': sentiment_score,
                'sentiment_label': ['positive', 'negative', 'neutral'][sentiment_class],
                'text': text_to_analyze })
            processed_count += 1
        except Exception as e: logging.warning(f"Could not analyze sentiment for article: '{title[:50]}...'. Error: {e}", exc_info=False)
    logging.info(f"Analyzed sentiment for {processed_count} articles.")
    if not results: return pd.DataFrame(columns=['publishedAt', 'sentiment_score', 'sentiment_label', 'text'])
    return pd.DataFrame(results)


def preprocess_and_feature_engineer(stock_df, sentiment_df, company_name='Infosys'):
    """Preprocesses data and creates features. Assumes stock_df is flattened."""
    logging.info("Preprocessing and Feature Engineering...")
    if stock_df.empty: logging.error("Input stock_df is empty."); return pd.DataFrame(), pd.Series(), pd.DataFrame()
    if isinstance(stock_df.columns, pd.MultiIndex): logging.error("MultiIndex detected in preprocess! Should be flattened earlier."); return pd.DataFrame(), pd.Series(), pd.DataFrame()

    if not isinstance(stock_df.index, pd.DatetimeIndex):
        try:
            date_col_name = None
            if 'Date' in stock_df.columns: date_col_name = 'Date'
            elif stock_df.index.name and 'date' in stock_df.index.name.lower(): date_col_name = stock_df.index.name
            if date_col_name: stock_df.set_index(pd.to_datetime(stock_df[date_col_name]), inplace=True)
            else: raise ValueError("No DatetimeIndex.")
        except Exception as e: logging.error(f"Error setting DatetimeIndex in preprocess: {e}"); return pd.DataFrame(), pd.Series(), pd.DataFrame()
    stock_df.sort_index(inplace=True)

    required_cols_simple = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    if not all(col in stock_df.columns for col in required_cols_simple): logging.error(f"Missing required columns in preprocess: {required_cols_simple}. Found: {stock_df.columns.tolist()}"); return pd.DataFrame(), pd.Series(), pd.DataFrame()
    basic_ta_features = ['SMA_5', 'SMA_20', 'RSI_14']
    talib_features = ['MACD', 'MACD_signal', 'MACD_hist', 'BB_width', 'ATR', 'Stoch_k', 'Stoch_d'] if talib else []
    lagged_features = ['Return_1D', 'Return_5D', 'Volume_Change_1D']
    sentiment_features = ['avg_sentiment_score', 'Sentiment_3D_Avg', 'Sentiment_7D_Avg']
    all_potential_features = basic_ta_features + talib_features + lagged_features + sentiment_features + ['Volume']

    for col in required_cols_simple:
        if not pd.api.types.is_numeric_dtype(stock_df[col]):
            logging.warning(f"Column '{col}' not numeric in preprocess. Attempting conversion.")
            try: stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
            except Exception: stock_df[col] = np.nan
    stock_df.dropna(subset=required_cols_simple, inplace=True)
    if stock_df.empty: logging.error("DataFrame empty after numeric check/dropna in preprocess."); return pd.DataFrame(), pd.Series(), pd.DataFrame()


    logging.info("Calculating Technical Indicators...")
    try:
        close = stock_df['Close']; stock_df['SMA_5'] = close.rolling(window=5).mean(); stock_df['SMA_20'] = close.rolling(window=20).mean()
        delta = close.diff(); gain = delta.where(delta > 0, 0).fillna(0).rolling(window=14).mean(); loss = -delta.where(delta < 0, 0).fillna(0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-6); stock_df['RSI_14'] = 100 - (100 / (1 + rs))
    except Exception as e: logging.warning(f"Error calculating basic TA: {e}"); [stock_df.setdefault(col, np.nan) for col in basic_ta_features]

    if talib:
        try:
            high=stock_df['High']; low=stock_df['Low']; close=stock_df['Close']
            try: stock_df['MACD'], stock_df['MACD_signal'], stock_df['MACD_hist'] = talib.MACD(close)
            except Exception: stock_df[['MACD', 'MACD_signal', 'MACD_hist']] = np.nan
            try: upper, middle, lower = talib.BBANDS(close, timeperiod=20); stock_df['BB_width'] = (upper - lower) / middle.replace(0, 1e-6)
            except Exception: stock_df['BB_width'] = np.nan
            try: stock_df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            except Exception: stock_df['ATR'] = np.nan
            try: stock_df['Stoch_k'], stock_df['Stoch_d'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            except Exception: stock_df[['Stoch_k', 'Stoch_d']] = np.nan
        except Exception as e: logging.warning(f"Error in TA-Lib setup: {e}"); [stock_df.setdefault(col, np.nan) for col in talib_features]
    else: [stock_df.setdefault(col, np.nan) for col in talib_features] # Assign NaN if talib missing

    try:
        close=stock_df['Close']; volume=stock_df['Volume']
        stock_df['Return_1D'] = close.pct_change(1); stock_df['Return_5D'] = close.pct_change(5); stock_df['Volume_Change_1D'] = volume.pct_change(1)
    except Exception as e: logging.warning(f"Error calculating lagged features: {e}"); [stock_df.setdefault(col, np.nan) for col in lagged_features]


    logging.info("Processing Sentiment Data...")
    daily_sentiment_text = pd.DataFrame()
    if not sentiment_df.empty and 'publishedAt' in sentiment_df.columns:
        try:
            sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt']).dt.date
            daily_agg = sentiment_df.groupby('publishedAt').agg(
                avg_sentiment_score=('sentiment_score', 'mean'),
                news_texts=('text', lambda x: ' || '.join(x)) ).reset_index()
            daily_agg.rename(columns={'publishedAt': 'Date_sent'}, inplace=True)
            daily_agg['Date_sent'] = pd.to_datetime(daily_agg['Date_sent']).dt.date

            stock_df_reset = stock_df.reset_index(); stock_df_reset['Date_obj'] = stock_df_reset['Date'].dt.date
            merged_df = pd.merge(stock_df_reset, daily_agg, left_on='Date_obj', right_on='Date_sent', how='left')

            daily_sentiment_text = merged_df[['Date', 'news_texts']].dropna().copy(); daily_sentiment_text.set_index(pd.to_datetime(daily_sentiment_text['Date']), inplace=True); daily_sentiment_text.drop(columns=['Date'], inplace=True) # Index by datetime
            merged_df.set_index('Date', inplace=True); merged_df.drop(columns=['Date_obj', 'Date_sent', 'news_texts'], inplace=True, errors='ignore')
            stock_df = merged_df.sort_index()

            stock_df['avg_sentiment_score'].fillna(0, inplace=True)
            stock_df['Sentiment_3D_Avg'] = stock_df['avg_sentiment_score'].rolling(window=3).mean()
            stock_df['Sentiment_7D_Avg'] = stock_df['avg_sentiment_score'].rolling(window=7).mean()
            logging.info("Sentiment data aggregated and merged.")
        except Exception as e:
            logging.warning(f"Error processing sentiment data: {e}", exc_info=True)

            for col in sentiment_features:
                stock_df[col] = 0
            daily_sentiment_text = pd.DataFrame()
    else:
        logging.info("No sentiment data provided or 'publishedAt' missing.")

        for col in sentiment_features:
             stock_df[col] = 0 
        daily_sentiment_text = pd.DataFrame()


    # Target Variable
    logging.info("Calculating Target Variable...")
    try: stock_df['Target'] = (stock_df['Close'].shift(-1) > stock_df['Close']).astype(int)
    except Exception as e: logging.error(f"Error calculating Target variable: {e}"); stock_df['Target'] = np.nan

    # Final Cleaning
    logging.info("Selecting features and final cleaning...")
    stock_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_present = [f for f in all_potential_features if f in stock_df.columns]
    essential_cols = features_present + ['Target']
    essential_cols = [c for c in essential_cols if c in stock_df.columns]

    initial_rows = len(stock_df); stock_df.dropna(subset=essential_cols, inplace=True)
    logging.info(f"Dropped {initial_rows - len(stock_df)} rows due to NaNs in essential columns.")
    if stock_df.empty: logging.error("DataFrame empty after final dropna."); return pd.DataFrame(), pd.Series(), pd.DataFrame()

    try: y = stock_df['Target'].astype(int)
    except Exception as e: logging.error(f"Error getting Target column after dropna: {e}"); return pd.DataFrame(), pd.Series(), pd.DataFrame()

    final_features = [f for f in features_present if f in stock_df.columns]
    X = stock_df[final_features]
    if not daily_sentiment_text.empty:
        try:
            if not X.index.equals(daily_sentiment_text.index):
                daily_sentiment_text = daily_sentiment_text.reindex(X.index) 
            daily_sentiment_text = daily_sentiment_text[daily_sentiment_text.index.isin(X.index)]
        except Exception as align_err:
            logging.warning(f"Could not align sentiment text index: {align_err}")
            daily_sentiment_text = pd.DataFrame() 


    logging.info(f"Preprocessing complete. Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, daily_sentiment_text


def get_sentiment_context_vader(text):
    """Analyzes text with VADER and returns highlights."""
    if not SentimentIntensityAnalyzer: return "VADER not available.", []
    if not text or not isinstance(text, str) or not text.strip(): return "No text provided.", []

    analyzer = SentimentIntensityAnalyzer(); sentences = text.split(' || '); scored_sentences = []
    valid_sentences_found = 0
    for sentence in sentences:
        sentence = sentence.strip();
        if not sentence or sentence == '.': continue
        try: vs = analyzer.polarity_scores(sentence); scored_sentences.append((vs['compound'], sentence)); valid_sentences_found += 1
        except Exception: pass
    if valid_sentences_found == 0: return "No valid sentences processed.", []
    scored_sentences.sort(key=lambda x: abs(x[0]), reverse=True)
    overall_compound = analyzer.polarity_scores(text)['compound']
    overall_sentiment = "Positive" if overall_compound >= 0.05 else "Negative" if overall_compound <= -0.05 else "Neutral"
    context_summary = f"VADER Overall Sentiment Context: {overall_sentiment} (Compound: {overall_compound:.3f})"
    highlights = [f"({score:.3f}) {sent}" for score, sent in scored_sentences[:3]]
    return context_summary, highlights