
# Employee Churn Prediction

This project uses historical HR data and machine learning to predict employee attrition. It includes a predictive model built in Python and an interactive Tableau dashboard for visualizing churn risk by department and job role. The goal is to support HR teams with data-driven insights for proactive retention planning and workforce strategy.

---

## 🧠 Project Summary

- **Problem**: High employee turnover increases costs and disrupts operations.
- **Solution**: Build a predictive model to flag potential attrition risks and visualize trends across departments.
- **Approach**:
  - Preprocessed and cleaned HR datasets
  - Engineered features like tenure, performance, role, and engagement
  - Trained classification models to predict likelihood of churn
  - Visualized churn risk with Tableau dashboards

---

## 📁 Project Structure

```

employee-churn-prediction/
│
├── churn\_model.ipynb       # Jupyter notebook for model training and evaluation
├── data/
│   └── hr\_data.csv         # Sample HR dataset with employee features
├── tableau\_dashboard.twbx  # Tableau workbook for churn risk visualization
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

````

---

## ⚙️ Technologies Used

- Python
  - pandas, scikit-learn, matplotlib, seaborn
- Tableau
  - Interactive visualizations for HR analytics
- Jupyter Notebook

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/employee-churn-prediction.git
cd employee-churn-prediction
````

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Open and run the `churn_model.ipynb` notebook to explore:

   * Data preprocessing
   * Model training and testing
   * Evaluation metrics and feature importance

2. Open `tableau_dashboard.twbx` in Tableau Desktop to interact with the visual dashboard:

   * View churn risk by department, role, or performance
   * Drill down to individual-level insights

---

## 📊 Key Features

* **Attrition Classification**: Predict whether an employee is at risk of leaving.
* **Feature Engineering**: Incorporates tenure, performance ratings, engagement, job role, and satisfaction.
* **Model Evaluation**: Uses accuracy, precision, recall, and confusion matrix to evaluate performance.
* **HR Dashboard**: Offers actionable insights on team-level risk for proactive intervention.

---

## 🔍 Insights & Use Cases

* Identify departments or job roles with elevated churn risk
* Enable early retention planning for high-risk employees
* Support HR strategy with visual and statistical evidence

---

## 📈 Sample KPIs

| Metric                | Description                                  |
| --------------------- | -------------------------------------------- |
| Churn Probability     | Likelihood score output by ML model          |
| High-Risk Departments | Departments with highest predicted attrition |
| Avg Tenure (Churned)  | Average length of stay for churned employees |
| Precision/Recall      | Evaluation metrics for model performance     |

---

## 🛠️ Future Enhancements

* Integrate real-time employee data pipeline
* Deploy model via Flask or FastAPI with a REST endpoint
* Add survey-based engagement features (if available)

---

## 📄 License

This project is intended for academic and research purposes.

---

## 👤 Author

**Anthony Baptiste**
[LinkedIn](https://www.linkedin.com/in/anthony-baptiste00)
[Portfolio](https://antbap23.github.io/portfolio)


