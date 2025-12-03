import pandas as pd

def main():
    df = pd.read_csv("data/student_performance_analysis.csv")
    df = df.dropna()
    # Encoding fitur kategorikal
    df = pd.get_dummies(df, drop_first=False)
    df.to_csv("data/prepared.csv", index=False)
    print("Data prepared successfully. Columns:", df.columns.tolist())

if __name__ == "__main__":
    main()
