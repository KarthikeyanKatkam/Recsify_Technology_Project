import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pdfkit
import os

# Function to load data
def load_data(filepath):
    try:
        data = pd.read_json(filepath)
        return data
    except ValueError as e:
        print(f"Error loading JSON file: {e}")
        return None

# Load the data
data = load_data('loan_approval_dataset.json')
if data is None:
    raise ValueError("Failed to load data. Please check the file path and format.")

# Data Exploration
def explore_data(data):
    print("Data Head:")
    print(data.head().T)
    print("\nData Info:")
    data.info()
    print("\nData Description:")
    print(data.describe().T)

explore_data(data)

# Data Visualization
def visualize_data(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Risk_Flag', data=data)
    plt.title('Distribution of Risk_Flag')
    plt.savefig('risk_flag_distribution.png', bbox_inches='tight')
    plt.show()

    # Pairplot to visualize relationships between features
    plt.figure(figsize=(14, 10))
    sns.pairplot(data, hue='Risk_Flag')
    plt.savefig('feature_relationships.png', bbox_inches='tight')
    plt.show()

visualize_data(data)

# Feature Engineering
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['Id'], axis=1)
    
    # Handle missing values if any (example with mean imputation)
    data.fillna(data.mean(), inplace=True)
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop('Risk_Flag', axis=1))
    
    return data_scaled, data['Risk_Flag']

data_scaled, target = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

# Model Building and Evaluation
def build_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize and train the model
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    
    # Predictions
    y_pred = rfc.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    feature_importances = rfc.feature_importances_
    
    print('Accuracy:', accuracy)
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Feature Importances:')
    print(feature_importances)
    
    return accuracy, conf_matrix, feature_importances

accuracy, conf_matrix, feature_importances = build_and_evaluate_model(X_train, y_train, X_test, y_test)

# Create a PDF report
def create_pdf_report(data, accuracy, conf_matrix, feature_importances):
    report_content = f"""
    <h1>Risk_Flag Prediction Report</h1>
    <h2>Data Exploration</h2>
    <p>The dataset contains {data.shape[0]} samples and {data.shape[1]} features.</p>
    <p>The distribution of Risk_Flag is:</p>
    <img src="risk_flag_distribution.png" alt="Risk_Flag Distribution">
    <h2>Feature Engineering</h2>
    <p>The data was scaled using StandardScaler and missing values were handled.</p>
    <h2>Model Building</h2>
    <p>A Random Forest Classifier was trained on the scaled data with an accuracy of {accuracy * 100:.2f}%.</p>
    <h2>Model Evaluation</h2>
    <p>The confusion matrix is:</p>
    <pre>{conf_matrix}</pre>
    <h2>Feature Importance</h2>
    <p>The feature importances are:</p>
    <pre>{feature_importances}</pre>
    <h2>Feature Relationships</h2>
    <p>The relationships between features are shown below:</p>
    <img src="feature_relationships.png" alt="Feature Relationships">
    """
    
    # Save the report as a PDF
    pdfkit.from_string(report_content, 'report.pdf')

# Ensure wkhtmltopdf is installed and configured
if not os.system('which wkhtmltopdf'):
    create_pdf_report(data, accuracy, conf_matrix, feature_importances)
else:
    print("wkhtmltopdf is not installed or not in PATH. Please install it to generate PDF reports.")
