
# 🎓 Student Performance Prediction Model

> 🚀 A machine learning project that predicts a student’s final grade (G3) using past performance, study habits, and engineered features — built with care, research, and precision.

---

## 📌 Overview

This project uses **regression-based machine learning models** to predict the **final academic grade (G3)** of students. It goes beyond basic features by introducing **custom-engineered attributes** that significantly improve prediction accuracy.

---

## 🧠 Models Explored

| Model                   | R² Score | RMSE  |
|------------------------|----------|--------|
| ✅ **Linear Regression**     | `0.86`    | `1.45` |
| Decision Tree Regressor | `0.66`    | `2.28` |
| Support Vector Machine  | `0.76`    | `1.91` |
| Random Forest Regressor | `0.84`    | `1.54` |

➡️ **Final Model Selected**: **Linear Regression** for its simplicity, interpretability, and competitive accuracy.

---

## 🧪 Features Used

The following features were selected after detailed correlation analysis and significance testing:

### 🔹 Raw Features:
- `G1`, `G2`: Past performance (1st and 2nd-period grades)
- `studytime`: Weekly study time
- `failures`: Number of past class failures
- `Medu`, `Fedu`: Mother’s and father’s education levels
- `higher`: Desire for higher education (mapped: yes=0, no=1)

---

### 🔧 Engineered Features (Custom!):

| Feature Name        | Description |
|---------------------|-------------|
| `score_desire`      | Estimated motivation (1 - (failures/3)) × higher desire |
| `score_trend`       | Difference between `G2 - G1` to track improvement or decline |
| `parent_studytime`  | Average of `Medu` and `Fedu` × `studytime` |
| `travel_study_gap`  | Gap between study time and travel time (if both available) |
| `desire_gap`        | How far current grade is from perfect score (1 - G2/20) |

💡 **These engineered features made a measurable improvement in R² score during model testing.**

---

## 📁 Project Structure

```plaintext
student-performance-predictor/
│
├── data/
│   └── student-data.csv                 # Dataset
│
├── models/
│   └── Student_Performance.joblib       # Trained ML model
│
├── notebooks/
│   └── Student Performance.ipynb        # Notebook version
│
├── Student-Performance.py               # Main training & prediction script
├── Outputs from different models.txt    # Main training & prediction script
├── requirements.txt                     # Python package dependencies
└── README.md                            # This beautiful file ✨
```

---

## 🧪 How to Run

### Step-by-step

```bash
# Clone this repo
git clone https://github.com/Meet-official/student-performance-predictor.git
cd student-performance-predictor

# Install dependencies
pip install -r requirements.txt

# Run the model
python Student-Performance.py
```

✅ This will:
- Train the Linear Regression model
- Evaluate R² and RMSE
- Save the model using `joblib`
- Predict a random student's final grade (G3)

---

## 🎯 Sample Prediction Output

```bash
🎯 Predicted Final Grade (G3): 14.14

📊 Final Evaluation on Test Data:
R² Score: 0.86
RMSE: 1.45
```

---

## 📊 Dataset Source

- [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- 649 rows × 33 attributes (Portuguese class)

---

## 💼 Use Cases

- Academic performance prediction
- Education tech platforms
- Dropout risk analysis
- Personalized study recommendation engines

---

## 🛠 Tech Stack

- Python 3.13.1
- Pandas, NumPy
- scikit-learn
- joblib
- Jupyter / VS Code

---

## 📜 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

## ✍️ Author

**Made with ❤ by Meet Patel**

---

## ⭐️ Show Some Love

If you found this useful, consider giving it a ⭐️ on GitHub — it motivates and supports further development!

---
