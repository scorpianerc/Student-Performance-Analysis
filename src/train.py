import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def main():
    # Set tracking URI - gunakan file-based jika tidak bisa connect ke server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    df = pd.read_csv("data/prepared.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        with open("data/model.pkl", "wb") as f:
            pickle.dump(clf, f)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", acc)
        
        # Log model dengan signature dan input example
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(
            clf, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        print(f"Training completed. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
