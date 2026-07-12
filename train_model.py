import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "data/diabetes_prediction_dataset.csv"
)

# Encode categorical columns

le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df["gender"] = le_gender.fit_transform(
    df["gender"]
)

df["smoking_history"] = le_smoking.fit_transform(
    df["smoking_history"]
)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

joblib.dump(
    model,
    "models/diabetes_xgb.pkl"
)

joblib.dump(
    le_gender,
    "models/gender_encoder.pkl"
)

joblib.dump(
    le_smoking,
    "models/smoking_encoder.pkl"
)

print("Model Saved")