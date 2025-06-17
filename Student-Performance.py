# Student Academic - Performance Predictor

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# Fetching dataset in student
student = pd.read_csv("data/student-data.csv")

# Feature Engineering
student['sex'] = student['sex'].map({'M': 1, 'F': 0})
student['schoolsup'] = student['schoolsup'].map({'yes': 1, 'no': 0})
student['famsup'] = student['famsup'].map({'yes': 1, 'no': 0})
student['paid'] = student['paid'].map({'yes': 1, 'no': 0})
student['activities'] = student['activities'].map({'yes': 1, 'no': 0})
student['higher'] = student['higher'].map({'yes': 0, 'no': 1})
student['internet'] = student['internet'].map({'yes': 0, 'no': 1})
student['romantic'] = student['romantic'].map({'yes': 0, 'no': 1})

# ✂️ Split the dataset with Stratified Shuffle Split
student["G3_cat"] = pd.cut(student["G3"],
                           bins=[-1, 5, 10, 15, 20],
                           labels=[0, 1, 2, 3])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(student, student["G3_cat"]):
    strat_train_set = student.loc[train_idx]
    strat_test_set = student.loc[test_idx]

# Removing the column G3_cat after using
for set_ in (strat_train_set, strat_test_set):
    set_.drop("G3_cat", axis=1, inplace=True)

#  🧹 Selecting Final Features
features = [
    "G1", "G2",                                        # Strong predictors
    "score_desire", "score_trend",                     # Engineered high-corr features
    "parent_studytime", "studytime", "Medu", "Fedu",   # Parental & study info
    "travel_study_gap", "failures", "higher",          # Negatively correlated, but meaningful
    "desire_gap"                                       # Engineered negative-corr
]

# ⚙️ Preprocessing
scaler = StandardScaler()
train_scaled = scaler.fit_transform(strat_train_set[features])
test_scaled = scaler.transform(strat_test_set[features])


# 🤖 Training ML Models

model = LinearRegression()
# model = DecisionTreeRegressor()
# model = SVR()
# model = RandomForestRegressor()

model.fit(train_scaled, strat_train_set["G3"])

# Predict
preds = model.predict(test_scaled)

# Evaluation
r2 = r2_score(strat_test_set["G3"], preds)
mse = mean_squared_error(strat_test_set["G3"], preds)
rmse = mse ** 0.5

print("📊 Final Evaluation on Test Data:")
print("R² Score:", r2)
print("RMSE:", rmse)


# Saving the model
dump(model, 'Student_Performance.joblib')

# Loading the model (after saving the model we can load that model in any file and use that model)
# model = load('Real-Estate.joblib')


# Prediction using random data

features_vector = [
    12,     # G1
    13,     # G2
    0.7,    # score_desire
    1.0,    # score_trend (G2 - G1)
    6.0,    # parent_studytime (e.g., avg(3,3) × studytime 2)
    2,      # studytime
    3,      # Medu
    3,      # Fedu
    0,      # travel_study_gap (e.g., travel and study both = 2)
    0,      # failures
    1,      # higher (1 = wants higher education)
    0.35    # desire_gap (1 - G2/20 → 1 - 13/20 = 0.35)
]

# Use the same feature names used during training
feature_names = [
    "G1", "G2", "score_desire", "score_trend", "parent_studytime",
    "studytime", "Medu", "Fedu", "travel_study_gap", "failures",
    "higher", "desire_gap"
]

# Wrap into a DataFrame
input_df = pd.DataFrame([features_vector], columns=feature_names)

# Scale it using StandardScaler
input_scaled = scaler.transform(input_df)

# Predict
predicted_g3 = model.predict(input_scaled)[0]
print(f"📘 Predicted Final Grade (G3): {predicted_g3:.2f}")
