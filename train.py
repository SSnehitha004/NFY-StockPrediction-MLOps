
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           f1_score, precision_score, recall_score) # Added specific metrics
from dotenv import load_dotenv
import logging
import warnings
import sys
import argparse 
import time 

import mlflow
import mlflow.sklearn 
from mlflow.tracking import MlflowClient 
from mlflow.models import infer_signature 
from mlflow.exceptions import MlflowException 

try:
    from utils import (
        fetch_stock_data,
        fetch_news,
        analyze_sentiment_finbert,
        preprocess_and_feature_engineer
    )
except ImportError:
    logging.error("Could not import from utils.py. Make sure it's in the Python path.")
    sys.exit(1)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


load_dotenv()
NEWS_API_KEY = os.getenv('NEWSAPI_KEY')
TICKER = 'INFY.NS'
COMPANY_NAME = 'Infosys'
NEWS_SOURCES = 'google-news-in,the-times-of-india,the-hindu,business-standard,reuters'
FETCH_DAYS_HISTORY = 365 * 3
FETCH_NEWS_HISTORY_DAYS = 28 
MLFLOW_EXPERIMENT_NAME = "Stock_Prediction_Infosys" 
MLFLOW_RUN_DESCRIPTION = "Training RandomForest with TA-Lib and FinBERT features"
MODEL_REGISTRY_NAME = "StockPredictorInfosys" 
PRODUCTION_ALIAS = "production" 

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=150, help="Number of trees in the forest")
parser.add_argument("--max_depth", type=int, default=10, help="Max depth of trees")
parser.add_argument("--min_samples_leaf", type=int, default=5, help="Min samples per leaf node")
parser.add_argument("--set_alias", type=str, default=PRODUCTION_ALIAS, help="Alias to set for the new model version (e.g., 'production', 'staging', or 'None')")
args = parser.parse_args()

N_ESTIMATORS = args.n_estimators
MAX_DEPTH = args.max_depth
MIN_SAMPLES_LEAF = args.min_samples_leaf
ALIAS_TO_SET = args.set_alias if args.set_alias.lower() != 'none' else None 


def train_pipeline():
    """Orchestrates the model training process and logs to MLflow."""
    logging.info("--- Starting Training Pipeline ---")


    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(description=MLFLOW_RUN_DESCRIPTION) as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("company_name", COMPANY_NAME)
        mlflow.log_param("fetch_days_history", FETCH_DAYS_HISTORY)
        mlflow.log_param("fetch_news_history_days", FETCH_NEWS_HISTORY_DAYS)


        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_features", "log2")
        mlflow.set_tag("mlflow.runName", f"RF_{TICKER}_{pd.Timestamp.now():%Y%m%d_%H%M}")
        mlflow.set_tag("training_triggered_by", "manual_script")

        # 1. Fetch Data
        logging.info("Fetching data...")
        stock_data = fetch_stock_data(TICKER, FETCH_DAYS_HISTORY)
        if stock_data.empty:
            logging.error("Training pipeline aborted: Failed to fetch stock data.")
            mlflow.set_tag("status", "FAILED_DATA_FETCH")
            return None

        news_articles = fetch_news(NEWS_API_KEY, COMPANY_NAME, NEWS_SOURCES, days_ago=FETCH_NEWS_HISTORY_DAYS)
        sentiment_data = analyze_sentiment_finbert(news_articles)

        # 2. Preprocess & Feature Engineer
        logging.info("Preprocessing and feature engineering...")
        X, y, _ = preprocess_and_feature_engineer(stock_data.copy(), sentiment_data.copy(), company_name=COMPANY_NAME)

        if X.empty or y.empty:
            logging.error("Training pipeline aborted: No data available for training after preprocessing.")
            mlflow.set_tag("status", "FAILED_PREPROCESSING")
            return None
        if X.isnull().values.any():
            logging.warning("NaNs found in final features X before splitting. Check preprocessing logic.")
        logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_training_samples_before_split", len(X))
        mlflow.log_param("feature_names", X.columns.tolist())


        # 3. Split Data using TimeSeriesSplit
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        logging.info(f"Using TimeSeriesSplit with {n_splits} splits.")
        try:
            train_index, test_index = list(tscv.split(X))[-1]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            logging.info(f"Data split: Train size={len(X_train)}, Test size={len(X_test)}")
            mlflow.log_param("test_set_size", len(X_test))
            if len(X_train) == 0 or len(X_test) == 0: raise ValueError("Train or test set empty.")
        except Exception as e:
            logging.error(f"Error during TimeSeriesSplit: {e}. Data length={len(X)}")
            mlflow.set_tag("status", "FAILED_DATA_SPLIT")
            return None

        # 4. Train Model
        logging.info("Training RandomForestClassifier model...")
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=42,
            class_weight='balanced',
            max_features='log2',
            n_jobs=-1
        )
        try:
            model.fit(X_train, y_train)
            logging.info("Model training complete.")
        except Exception as e:
            logging.error(f"Error during model training: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED_TRAINING")
            return None

        # 5. Evaluate Model
        logging.info("Evaluating model...")
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # Prob of positive class

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            report = classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info("Classification Report:\n" + report)


            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })


            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=ax)
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix')
                cm_path = "confusion_matrix.png"; plt.savefig(cm_path); plt.close(fig)
                mlflow.log_artifact(cm_path); os.remove(cm_path)
            except Exception as plot_err: logging.warning(f"Could not log confusion matrix plot: {plot_err}")
            # Classification Report
            report_path = "classification_report.txt"
            with open(report_path, "w") as f: f.write(report)
            mlflow.log_artifact(report_path); os.remove(report_path)
            # Feature Importances
            if hasattr(model, 'feature_importances_'):
                try:
                    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value','Feature'])
                    feature_imp = feature_imp.sort_values(by='Value', ascending=False).reset_index(drop=True)
                    imp_path = "feature_importances.csv"; feature_imp.to_csv(imp_path, index=False)
                    mlflow.log_artifact(imp_path); os.remove(imp_path)
                except Exception as imp_err: logging.warning(f"Could not log feature importances: {imp_err}")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED_EVALUATION")


        # 6. Log Model & Set Alias/Tags
        logging.info("Logging model and setting alias/tags...")
        try:
            signature = infer_signature(X_train, model.predict(X_train))
            # Log the model, which registers it if name is provided
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=MODEL_REGISTRY_NAME
            )
            logging.info(f"Model logged to: {model_info.model_uri}")


            client = MlflowClient()
            time.sleep(5)
            try:
                 latest_versions = client.search_model_versions(f"run_id='{run_id}' and name='{MODEL_REGISTRY_NAME}'")
                 if not latest_versions:
                     logging.warning(f"Could not find version for run {run_id} via search. Getting latest.")
                     latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None"])

                 if latest_versions:
                     model_version = latest_versions[0].version 
                     logging.info(f"Registered as: {MODEL_REGISTRY_NAME}, Version: {model_version}")
                 else:
                      raise MlflowException(f"No versions found for registered model '{MODEL_REGISTRY_NAME}' after logging.")

            except MlflowException as me:
                 logging.error(f"MLflow Exception while getting model version: {me}")
                 raise 
            except Exception as client_e:
                 logging.error(f"Error getting model version using MlflowClient: {client_e}")
                 raise 

            if ALIAS_TO_SET:
                logging.info(f"Setting alias '{ALIAS_TO_SET}' to version {model_version}...")
                client.set_registered_model_alias(name=MODEL_REGISTRY_NAME, alias=ALIAS_TO_SET, version=model_version)
                logging.info(f"Alias '{ALIAS_TO_SET}' set successfully.")
                mlflow.set_tag("alias_set", ALIAS_TO_SET)

            client.set_model_version_tag(name=MODEL_REGISTRY_NAME, version=model_version, key="validation_status", value="pending")
            client.set_model_version_tag(name=MODEL_REGISTRY_NAME, version=model_version, key="data_date_range", value=f"{X.index.min().date()}_to_{X.index.max().date()}")
            logging.info("Tags set for model version.")

            mlflow.set_tag("status", "COMPLETED")

        except Exception as e:
            logging.error(f"Error logging model or setting alias/tags: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED_LOGGING")
            return None

        logging.info("--- Training Pipeline Finished Successfully ---")
        return run_id


if __name__ == "__main__":
    logging.info(f"Executing Training Script for {TICKER}...")
    run_id = train_pipeline()
    if run_id:
        logging.info(f"Training completed. MLflow Run ID: {run_id}")
    else:
        logging.error("Training pipeline failed.")
        sys.exit(1) 

    logging.info("Script execution finished.")