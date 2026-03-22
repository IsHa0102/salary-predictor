import sys
import os

# Allow importing from src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib


def train():
    print("training started")
    # Load and preprocess data
    df = preprocess_data()

    # Split features and target
    X = df.drop("salary_in_usd", axis=1)
    y = df["salary_in_usd"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)

    print("\n📊 Model Evaluation")
    print(f"MAE: {mae:.2f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/salary_model.pkl")

    print("\n✅ Model saved at: model/salary_model.pkl")

XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05
)
if __name__ == "__main__":
    train()