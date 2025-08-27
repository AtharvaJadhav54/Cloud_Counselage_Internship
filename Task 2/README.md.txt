# Task 2: AI-Powered Career Path Prediction System

This project is an interactive web application that uses a machine learning model to predict a suitable career path for students based on their skills and interests.

---

## 🎯 Objective

The goal is to build an end-to-end machine learning system that:
1.  Trains a reliable model on the provided student dataset (`PS2_Dataset.csv`).
2.  Provides a user-friendly web interface for students to input their information.
3.  Delivers a real-time career prediction and suggests relevant courses for professional development.

---

## 🛠️ Methodology & Features

The project follows a complete machine learning workflow:

1.  **Data Cleaning:** The raw dataset is cleaned and standardized. Messy, free-form text in categorical columns is mapped to clean, consistent categories.
2.  **Feature Engineering:** **One-Hot Encoding** is used to convert all categorical features into a numerical format that the model can effectively learn from.
3.  **Model Training:** A **RandomForestClassifier** is trained on the processed data. This model was chosen for its robustness and resistance to overfitting.
4.  **Feature Selection:** To improve performance and prevent overfitting, the system first identifies the **top 10 most influential features** and then trains the final model using only this focused subset of data.
5.  **Streamlit Application:** The entire system is deployed as an interactive web application using Streamlit. The app includes:
    * An intuitive input form with radio buttons and dropdowns.
    * A results section displaying the predicted career path and course recommendations.
    * A dashboard with model insights, including **Feature Importance** and data visualizations.

---

## 🚀 How to Run the Application

1.  **Prerequisites:** Ensure you have Python installed. It is highly recommended to use a virtual environment.
2.  **Install Dependencies:** Install all required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  **File Placement:** Place the script (`app.py`), the `PS2_Dataset.csv` file, and `requirements.txt` in the same directory.
4.  **Execution:** Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
5.  **Access:** Your web browser will automatically open a new tab with the running application.
