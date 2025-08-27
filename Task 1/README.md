# Task 1: Student's Year of Graduation Prediction

This project, contained in a Jupyter Notebook, processes student lead data to calculate their expected year of graduation.

---

## 🎯 Objective

The primary goal of this task is to take a raw dataset of student leads (`Final Lead Data.xlsx`) and, based on their current academic year, calculate the year they are expected to graduate. This is primarily a data processing and feature engineering task.

---

## 🛠️ Methodology

The process follows these key steps:

1.  **Data Loading:** The notebook loads the data directly from the `Final Lead Data.xlsx` file.
2.  **Data Cleaning:**
    * Removes duplicate student entries based on their email address.
    * Consolidates multiple columns for 'College' and 'Branch' into single, clean columns.
    * Intelligently merges the two different "Academic Year" columns into one reliable feature.
3.  **Calculation:** A function calculates the graduation year by adding the remaining years of study (assuming a 4-year program) to the current year.
4.  **Data Visualization:** Interactive charts are generated using Plotly to visualize key aspects of the dataset, such as the distribution of students by gender, academic year, and college.
5.  **Output:** The final, processed data, including the calculated graduation year, is saved to a new Excel file named `Predicted_Graduation_Data.xlsx`. The generated charts are saved as individual `.png` files.

---

## 🚀 How to Run

1.  **Prerequisites:** Ensure you have a Jupyter environment (like Jupyter Lab or VS Code with the Jupyter extension) and the following libraries installed:
    ```bash
    pip install pandas openpyxl plotly kaleido notebook
    ```
2.  **File Placement:** Place the notebook (`Task_1_Student's_Year_of_Graduation_Prediction.ipynb`) and the `Final Lead Data.xlsx` file in the same directory.
3.  **Execution:** Open the notebook and run the cells sequentially from top to bottom.
4.  **Output:** The notebook will generate the `Predicted_Graduation_Data.xlsx` file and save the visualization charts as `.png` image files in the same folder.
