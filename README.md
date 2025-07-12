# zeroCodeML


### 🔍**Core Project Objective**

> Build a system that accepts a dataset and task (like classification, regression, or clustering), then **automatically processes, trains, evaluates, and returns results** with minimal or no user intervention.

### ⚙️ **Workflow**

[User Uploads Dataset + Task]
             ↓
    [Preprocessing Engine]
             ↓
     [Auto Task Verifier]
             ↓
 [Model Selector + Trainer]
             ↓
     [Evaluation + Output]
             ↓
   [Download Model / Report]

### 🧰 **Suggested Tech Stack**

| Component       | Tools/Libs                                          |
| --------------- | --------------------------------------------------- |
| Language        | Python                                              |
| Backend (logic) | `scikit-learn`,`pandas`,`numpy`,`joblib`    |
| AutoML Engines  | `AutoSklearn`,`TPOT`,`H2O AutoML`,`PyCaret` |
| Optional UI     | Streamlit or Flask                                  |
| Visualizations  | `matplotlib`,`seaborn`,`plotly`               |

### 📁 Folder Structure (Simple Version)

auto_ml_pipeline/
│
├── app.py                       # Main script or Streamlit app
├── preprocess.py           # Data cleaning functions
├── model_selector.py    # Logic for choosing/training models
├── evaluator.py              # Metrics and reporting
├── utils.py                      # Helper functions
├── uploads/                   # Uploaded datasets
├── models/                    # Saved models
├── reports/                    # Output reports


### 💡**Example Usage Scenarios**

* A teacher can test models on student datasets for projects.
* A data analyst can test multiple models without writing code.
* Integrate into low-code platforms for startups or dashboards.
