from data_loader import load_data
from data_preprocessing import preprocess_data
from model_training import train_logistic_regression, train_decision_tree, train_random_forest
from model_evaluation import evaluate_model

def main():
    data_file = 'customer_churn.csv'
    churn_df = load_data(data_file)
    if churn_df is not None:
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(churn_df)
        logistic_model = train_logistic_regression(preprocessor, X_train, y_train)
        decision_tree_model = train_decision_tree(preprocessor, X_train, y_train)
        random_forest_model = train_random_forest(preprocessor, X_train, y_train)
        evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")
        evaluate_model(decision_tree_model, X_test, y_test, "Decision Tree")
        evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

if __name__ == "__main__":
    main()
