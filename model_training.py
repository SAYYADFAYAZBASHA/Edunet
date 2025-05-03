from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_logistic_regression(preprocessor, X_train, y_train):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(random_state=42))])
    model.fit(X_train, y_train)
    return model

def train_decision_tree(preprocessor, X_train, y_train):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier(random_state=42))])
    model.fit(X_train, y_train)
    return model

def train_random_forest(preprocessor, X_train, y_train):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data

    data_file = 'customer_churn.csv'
    churn_df = load_data(data_file)
    if churn_df is not None:
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(churn_df)
        logistic_model = train_logistic_regression(preprocessor, X_train, y_train)
        decision_tree_model = train_decision_tree(preprocessor, X_train, y_train)
        random_forest_model = train_random_forest(preprocessor, X_train, y_train)
