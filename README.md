# Infosys Stock Prediction MLOps Project

This project predicts Infosys (`INFY.NS`) stock price movements using a RandomForestClassifier, technical indicators, and news sentiment analyzed with FinBERT and VADER. It includes a training pipeline, a Streamlit dashboard for real-time predictions, and MLflow for model tracking and registry.

## Features
- **Data Pipeline**: Fetches stock data (`yfinance`) and news (`NewsAPI`), with FinBERT sentiment analysis for training features.
- **Training Pipeline**: Trains a RandomForestClassifier, logs metrics (accuracy, precision, recall, F1-score), and artifacts (confusion matrix, feature importances) to MLflow.
- **Inference Pipeline**: Streamlit dashboard displays predictions, confidence scores, stock price charts, news, and VADER sentiment context.
- **Model Management**: Uses MLflow UI to track experiments and manage model versions.

## Demo
Watch a video of the Streamlit dashboard running locally: <br>
https://github.com/user-attachments/assets/0fe44832-0e29-450a-a0b5-61df9f7879b9

## MLflow UI
MLflow tracks experiments, logs metrics, and manages models. Below are screenshots of the MLflow UI:
- **Experiments**: Showing the "Stock_Prediction_Infosys" experiment.
![Image](https://github.com/user-attachments/assets/67a01841-b2c8-433e-b135-26f7502791e2)

## Project Structure
utils.py: Data fetching, sentiment analysis, and preprocessing <br>
train.py: Model training, evaluation, and MLflow logging. <br>
stkdashboard.py: Streamlit app for predictions and visualizations.
