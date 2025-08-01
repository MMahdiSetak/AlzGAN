import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # For saving model


def run():
    # Load data (adjust paths if needed)
    X_train = pd.read_csv('dataset/tabular/train_x.csv')
    y_train = pd.read_csv('dataset/tabular/train_y.csv')['DIAGNOSIS']
    X_val = pd.read_csv('dataset/tabular/val_x.csv')
    y_val = pd.read_csv('dataset/tabular/val_y.csv')['DIAGNOSIS']
    X_test = pd.read_csv('dataset/tabular/test_x.csv')
    y_test = pd.read_csv('dataset/tabular/test_y.csv')['DIAGNOSIS']

    # Define model with class_weight for imbalance
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Simple hyperparameter tuning on val (expand grid as needed)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    # Evaluate on val (for sanity check)
    y_val_pred = best_rf.predict(X_val)
    print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:\n",
          classification_report(y_val, y_val_pred, target_names=['CN (0)', 'MCI (1)', 'AD (2)']))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    # Final eval on test
    y_test_pred = best_rf.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:\n",
          classification_report(y_test, y_test_pred, target_names=['CN (0)', 'MCI (1)', 'AD (2)']))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    # Feature importances (for insight—e.g., MMSE/ADAS often top)
    importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_rf.feature_importances_}).sort_values(
        'Importance', ascending=False)
    print("\nTop Features:\n", importances.head(10))

    # Save model for multimodal fusion later
    joblib.dump(best_rf, 'tabular_rf_model.pkl')
