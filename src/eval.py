import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import json

def main():
    df = pd.read_csv("data/prepared.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    with open("data/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
