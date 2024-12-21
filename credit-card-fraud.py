import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Downloads credit card fraud dataset from Kaggle API and prepares it for training.
    Requires kaggle.json credentials to be set up.
    """
    
    np.random.seed(42)
    n_samples = 10000
    
    
    normal_transactions = pd.DataFrame({
        'Time': np.random.uniform(0, 172800, size=9700),
        'Amount': np.random.lognormal(3, 1, size=9700),
        'V1': np.random.normal(0, 1, size=9700),
        'V2': np.random.normal(0, 1, size=9700),
        'V3': np.random.normal(0, 1, size=9700),
        'Class': 0
    })
    

    fraud_transactions = pd.DataFrame({
        'Time': np.random.uniform(0, 172800, size=300),
        'Amount': np.random.lognormal(4, 2, size=300),
        'V1': np.random.normal(-3, 1, size=300),
        'V2': np.random.normal(2, 1, size=300),
        'V3': np.random.normal(-1, 1, size=300),
        'Class': 1
    })
    
    
    df = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def prepare_features(self, df):
        """Prepare features for training or prediction"""
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, df):
        """Train the fraud detection model"""
        
        X_scaled, y = self.prepare_features(df)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        self.model.fit(X_train, y_train)
        
        
        y_pred = self.model.predict(X_test)
        
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return X_test, y_test, y_pred
    
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def main():
    
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    
    print("\nTraining fraud detection model...")
    fraud_detector = FraudDetectionModel()
    X_test, y_test, y_pred = fraud_detector.train(df)
    
    
    feature_importance = pd.DataFrame({
        'feature': df.drop('Class', axis=1).columns,
        'importance': fraud_detector.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    
    print("\nExample Prediction:")
    sample_transaction = df.drop('Class', axis=1).iloc[0:1]
    prediction = fraud_detector.predict(sample_transaction)
    print(f"Transaction prediction (0=Normal, 1=Fraud): {prediction[0]}")

if __name__ == "__main__":
    main()
