# 🤖 AI Ticket Intelligent System (Ticket Classifier)

A premium, machine learning-powered system designed to automate IT support workflows. This tool analyzes ticket descriptions to predict the appropriate department, determine technical priority, and recommend the best-suited employee based on historical performance metrics.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blueviolet?style=for-the-badge)

## 🌟 Key Features
- **Hybrid NLP Analysis**: Uses TF-IDF vectorization to understand the context and intent of technical issues.
- **Multi-Output Classification**: Simultaneously predicts **Department** (IT, HR, Finance, etc.) and **Priority** (Critical, High, Medium, Low).
- **Optimization Engine**: Recommends the optimal employee by evaluating:
    - Predicted Resolution Time (Regression)
    - Predicted CSAT/Rating (Regression)
    - Historical Experience & Workload
- **Premium UI**: Modern dashboard built with Streamlit featuring glassmorphism cards, sidebar controls, and real-time AI feedback.

## 🛠️ Tech Stack
- **Frontend**: Streamlit (Python-based Web Framework)
- **Machine Learning**: Scikit-Learn (Random Forest Classifier & Regressor)
- **Data Processing**: Pandas, NumPy
- **Vectorization**: TF-IDF
- **Deployment**: Local Launcher (.bat)

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pintukumar245/Hybrid-Ticket-Prediction-Project-ML-NLP-.git
   cd Hybrid-Ticket-Prediction-Project-ML-NLP-
   ```

2. **Setup Environment**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**:
   If models are not present in the `models/` directory:
   ```bash
   python train_model.py
   ```

5. **Run the Application**:
   Simply run the launcher or use streamlit:
   ```bash
   run_classifier.bat
   # OR
   streamlit run app.py
   ```

## 📊 Dataset
The system is trained on a hybrid dataset combining synthetic support ticket text with structured employee performance data.

## 💼 Resume Showcase
This project demonstrates expertise in:
- End-to-end Machine Learning pipelines.
- Natural Language Processing (NLP) for text classification.
- Developing user-centric AI applications.
- UI/UX design within data-driven environments.

---
Built with ❤️ by Pintu Kumar
