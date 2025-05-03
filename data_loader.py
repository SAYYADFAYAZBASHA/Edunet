import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

def explore_data(df):
    if df is not None:
        print(df.head())
        print(df.info())
        print(df.describe())
        print(df['Churn'].value_counts())

if __name__ == "__main__":
    data_file = 'customer_churn.csv'
    churn_df = load_data(data_file)
    if churn_df is not None:
        explore_data(churn_df)

