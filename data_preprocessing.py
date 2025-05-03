import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    target_variable = 'Churn'
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    label_encoder = LabelEncoder()
    df[target_variable] = label_encoder.fit_transform(df[target_variable])
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_file = 'customer_churn.csv'
    churn_df = pd.read_csv(data_file)
    if churn_df is not None:
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(churn_df)
