from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data
    from model_training import train_logistic_regression, train_decision_tree, train_random_forest

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
