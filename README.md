# Student Performance Prediction AI

![Project banner](assets/banner.png)

Student Performance Prediction AI is a Streamlit-based machine learning dashboard that predicts a student's final score and academic pass/fail risk using academic history, attendance, study habits, support factors, and education context.

The app includes school-board and college/university modes for Indian education workflows, including CBSE, CISCE/ICSE, NIOS, state boards, union territory boards, bachelor degrees, and master degrees.

## Features

- Responsive Streamlit dashboard.
- School board and college/university prediction modes.
- Support for percentage, CGPA out of 10, and CGPA out of 4 inputs.
- Regression model for final score prediction.
- Classification model for pass/fail risk prediction.
- Training pipeline that supports additional Indian board and college datasets.
- CSV template for collecting anonymized Indian academic records.
- GitHub-ready folder structure.

## Project Structure

```text
student-performance-prediction-ai/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- LICENSE
|-- DATA_COLLECTION_GUIDE.md
|
|-- assets/
|   |-- banner.png
|
|-- data/
|   |-- student_data.csv
|
|-- dataset/
|   |-- student-mat.csv
|   |-- student-por.csv
|   |-- indian_student_records_template.csv
|
|-- models/
|   |-- student_model.pkl
|   |-- student_reg_model.pkl
|   |-- student_clf_model.pkl
|   |-- student_model_metadata.pkl
|
|-- notebook/
|   |-- model_training.ipynb
|
|-- screenshots/
|   |-- .gitkeep
|
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- predict.py
|   |-- train.py
|   |-- utils.py
|
|-- Eda/
|   |-- app.py
|   |-- train.py
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The App

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Train The Models

```powershell
python src\train.py
```

The training script saves model artifacts in:

```text
models/
```

It also keeps root-level legacy model files for compatibility with older app versions.

## Dataset

The current base dataset comes from the UCI Student Performance dataset. It includes academic and demographic features such as:

- previous grades
- final grade
- study time
- failures
- absences
- family support
- internet access
- health
- social habits

## Add Indian Board Or College Data

Use this file as the template:

```text
dataset/indian_student_records_template.csv
```

Save real anonymized files using names such as:

```text
dataset/indian_student_records_cbse.csv
dataset/indian_student_records_icse.csv
dataset/indian_student_records_state_boards.csv
dataset/indian_student_records_college.csv
```

The trainer automatically loads files matching:

```text
dataset/indian_student_records*.csv
```

Do not include private student information such as names, roll numbers, phone numbers, email addresses, Aadhaar numbers, or full addresses.

## Model Output

The dashboard returns:

- predicted final score on a 0-20 model scale
- academic risk category
- probability of passing
- model-scope details for converted marks, age, and absences

## Current Accuracy Note

The current model is trained mainly on the available student performance dataset. The app supports Indian board and degree fields, but high accuracy for CBSE, ICSE, state boards, union territory boards, and college degrees requires real anonymized Indian academic records.

## Screenshots

Add screenshots after running the app:

```text
screenshots/dashboard.png
screenshots/prediction.png
screenshots/analytics.png
```

## License

This project is licensed under the MIT License.
