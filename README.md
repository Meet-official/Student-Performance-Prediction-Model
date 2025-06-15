# 🎓 Student Performance Prediction Model

A machine learning project that predicts student CGPA based on academic and lifestyle-related features using Random Forest Regressor.

## 📁 Files Included

- `Student-Performance.py`: Main Python script for data preprocessing, training, and prediction.
- `Students_data.csv`: The dataset containing student records.
- `Student_Performance.joblib`: Saved Random Forest model.
- `requirements.txt`: Python libraries required.
- `Student-Performance.ipynb`: Step-by-step notebook version.

## 🧠 Features Used

- Gender, Mode of Study, Relationship Status
- Daily study hours, Attendance, Age, Semester
- Social media hours, Skill development time
- Composite features like `SemAge`, `StudyTimeScore`, and `SocialToStudyRatio`

## 📊 Models Tried

- ✅ Random Forest Regressor (Best)
- Linear Regression
- Decision Tree
- SVM

## 📈 Model Performance (Random Forest)

- R² Score: 0.12
- MSE: 0.61

## 🚀 How to Run

```bash
pip install -r requirements.txt
python Student-Performance.py
```

## 🧪 Sample Prediction

```
R² Score: 0.10880700649760289
MSE: 0.6190123833507855
```

## 🤖 Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Joblib (for model saving)

Made with ❤️ by **Meet Patel**
