# 🩺 Patient Health Risk Assessment

An AI-powered healthcare application that predicts a patient's diabetes risk using Machine Learning and generates personalized health recommendations using a Large Language Model (LLM). The system combines predictive analytics, explainable AI, and natural language generation to assist users in understanding their health risks.

## 🚀 Features

- Predicts diabetes risk using an XGBoost Machine Learning model.
- Classifies patients into Low, Medium, or High risk categories.
- Provides explainable AI insights using SHAP (SHapley Additive exPlanations).
- Generates personalized health reports using Hugging Face LLM (Qwen2.5-7B-Instruct).
- Interactive and user-friendly Streamlit web interface.
- Stores generated patient reports in an SQLite database.
- Fast and lightweight deployment for demonstration purposes.

## 🏗️ System Architecture

Patient Information
        │
        ▼
 Streamlit Interface
        │
        ▼
 Data Preprocessing
        │
        ▼
 XGBoost Prediction Model
        │
        ├──────────────► SHAP Explainability
        │
        ▼
 Risk Level Prediction
        │
        ▼
 Hugging Face LLM
(Qwen2.5-7B-Instruct)
        │
        ▼
 AI Generated Health Report
        │
        ▼
 SQLite Database Storage


## 🛠️ Technologies Used

- Python
- Streamlit
- XGBoost
- SHAP Explainable AI
- Scikit-learn
- Pandas
- NumPy
- SQLite
- Hugging Face Inference API
- Joblib


## 📂 Project Structure

Patient-Health-Risk-Assessment/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── diabetes_prediction_dataset.csv
│
├── models/
│   ├── diabetes_xgb.pkl
│   ├── gender_encoder.pkl
│   └── smoking_encoder.pkl
│
└── reports/


## 📊 Dataset

The project uses a diabetes prediction dataset containing patient health information, including:

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level

These features are used to predict the probability of diabetes.

## 🤖 Machine Learning Model

The prediction engine is built using an **XGBoost Classifier**, selected for its:

- High prediction accuracy
- Fast inference speed
- Ability to model non-linear relationships
- Strong performance on tabular healthcare datasets

The model outputs a probability score that is mapped into:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

## 🔍 Explainable AI

To improve transparency, the project uses **SHAP (SHapley Additive Explanations)**.

SHAP identifies the most influential features affecting each prediction, allowing users to understand why a particular risk level was assigned.

## 🤖 AI Health Report

After prediction, a personalized health report is generated using the **Hugging Face Inference API** with:

**Model Used**

- Qwen/Qwen2.5-7B-Instruct

The AI summarizes:

- Risk Level
- Health Explanation
- Lifestyle Recommendations
- Doctor Consultation Advice

## ⚙️ Installation

### Clone the repository

bash
git clone https://github.com/sandeshsn-official/Patient-Health-Risk-Assessment.git
cd Patient-Health-Risk-Assessment

### Install dependencies

bash
pip install -r requirements.txt

### Configure Hugging Face Token

Create a `.env` file in the project root:

env
HF_TOKEN=your_huggingface_api_token

### Run the application

bash
streamlit run app.py

## 💻 Usage

1. Enter patient details.
2. Click **Assess Health Risk**.
3. View:
   - Predicted Risk Level
   - Risk Probability
   - Important Risk Factors
   - AI Generated Health Report


## 📈 Results

The application successfully integrates:

- Machine Learning prediction
- Explainable AI
- Large Language Models
- Interactive visualization
- Database storage

to provide an end-to-end intelligent healthcare risk assessment system.

## Screenshots

Home Page

<img width="1882" height="526" alt="image" src="https://github.com/user-attachments/assets/be0e8729-7dab-4c23-a91f-ace21c0ed105" />


### Prediction Result

<img width="1905" height="857" alt="image" src="https://github.com/user-attachments/assets/f3ae7f5f-8b09-46d0-8292-02a9327bc086" />


### SHAP Explanation

<img width="1871" height="826" alt="image" src="https://github.com/user-attachments/assets/b2654f21-ad60-48a8-9db4-d97453597378" />


### AI Generated Report

<img width="1806" height="290" alt="image" src="https://github.com/user-attachments/assets/7be2dcc6-ecc9-4a14-a8bb-e4d095b4c830" />



## 🔮 Future Improvements

- Multi-disease prediction
- PDF report generation
- User authentication
- Cloud deployment
- Electronic Health Record (EHR) integration
- Doctor dashboard
- Medical image analysis support

## 👨‍💻 Author

**Sandesh S N**

M.Tech Data Science Student

Machine Learning • Deep Learning • NLP • Computer Vision

GitHub: https://github.com/sandeshsn-official

## 📜 License

This project is intended for educational and research purposes.
