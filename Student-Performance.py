# Student Academic - Performance Predictor

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# student = pd.read_csv("Students_data.csv")
student = pd.read_csv("Students_data.csv", encoding='utf-8', engine='python')

student['Gender'] = student['Gender'].map({'Male': 1, 'Female': 0})
student['Mode'] = student['Mode'].map({'Offline': 1, 'Online': 0})
student['Computer'] = student['Computer'].map({'Yes': 1, 'No': 0})
student['Suspension'] = student['Suspension'].map({'Yes': 1, 'No': 0})
student['AskProblems'] = student['AskProblems'].map({'Yes': 1, 'No': 0})
student['RelationshipStatus'] = student['RelationshipStatus'].map({'Single': 0, 'Relationship': 1, 'Married': 2})
# student = student.get_dummies(student, column=['RelationshipStatus'], drop_first = True)

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(student, student['Suspension']):
    strat_train_set = student.loc[train_index]
    strat_test_set = student.loc[test_index]

student = strat_train_set.copy()

# corr_matrix = student.corr()
# corr_matrix['CGPA'].sort_values(ascending = False)

student["SemAge"] = student['Sem']*student['Age']
student["StudyTimeScore"] = student["DailyHours"] * student["TimesSitStudyDay"]
student["SocialToStudyRatio"] = student["SocialMediaHrs"] / (student["DailyHours"] + 1)

# corr_matrix = student.corr()
# corr_matrix['CGPA'].sort_values(ascending = False)

# from sklearn.feature_selection import mutual_info_regression

X = student.drop('CGPA', axis=1).fillna(0)
y = student['CGPA']

# mi = mutual_info_regression(X, y)
# mi_series = pd.Series(mi, index=X.columns)
# mi_series.sort_values(ascending=False)

# pd.get_dummies(student, drop_first=True)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor()
# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = SVR()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
print("📊 Final Evaluation on Test Data:")
print("R² Score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))

# importances = model.feature_importances_
# plt.barh(X.columns, importances)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.title("Random Forest Feature Importance")
# plt.show()

dump(model, 'Student_Performance.joblib')

model = load('Real-Estate.joblib')

import pandas as pd
import numpy as np

# Example input with same features as training data
sample_input = pd.DataFrame({
    'Gender': [1],
    'Mode': [0],
    'Computer': [1],
    'Suspension': [0],
    'AskProblems': [1],
    'RelationshipStatus': [1],
    'Year': [2022],
    'Age': [21],
    'Sem': [5],
    'DailyHours': [6],
    'TimesSitStudyDay': [3],
    'SocialMediaHrs': [2],
    'Attendance': [80],
    'SkillDevHrs': [3],
    'SemAge': [5 * 21],
    'StudyTimeScore': [12],
    'SocialToStudyRatio': [1.8]
})

# Predict CGPA
predicted_cgpa = model.predict(sample_input)
print("🎯 Predicted CGPA:", predicted_cgpa[0])

