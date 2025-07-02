# House Price Prediction (California Housing)

This project is a complete end-to-end machine learning pipeline that predicts housing prices in California based on various features. It includes data preprocessing, model training, hyperparameter tuning, evaluation, and deployment with a Flask web application.

## 📁 Project Structure
house-price-prediction/
│
├── app/ # Flask web app
│ ├── templates/ # HTML templates (index.html, result.html)
│ └── app.py # Flask application
│
├── data/ # Raw and processed datasets
│ ├── housing.csv
│ ├── housing_strat_train_set.csv
│ └── housing_strat_test_set.csv
│
├── models/ # Saved models and preprocessors
│ ├── rf_model_v1.pkl
│ └── preprocessor_v1.pkl
│
├── src/ # Source code for data preprocessing, training, etc.
│ ├── train_test_split.py
│ ├── data_preprocessing.py
│ └── model_training.py
│ ├── error_metrics.py
│ ├── fine_tune_best_model.py
│ └── model_evaluation.py

## Features
- Data cleaning and preprocessing
- Feature engineering (e.g., rooms per household)
- Stratified sampling by income category
- Model training: Linear Regression, Decision Tree, Random Forest
- Grid Search for best model tuning
- RMSE & MSE evaluation
- Model deployment via Flask
- Form-based UI for input

## Dependencies
- Python 3.10+
- pandas
- numpy
- scikit-learn
- Flask
- cloudpickle
- joblib

## Credits
Based on exercises and guidance from the book:
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron.

