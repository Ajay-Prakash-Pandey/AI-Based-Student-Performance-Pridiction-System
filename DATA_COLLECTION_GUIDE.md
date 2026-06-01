# Data Collection Guide

This project can train on Indian school board and college/university records when you add real rows in the template format.

## Where To Put Data

1. Open `dataset/indian_student_records_template.csv`.
2. Keep the same column names.
3. Save real files as:

```text
dataset/indian_student_records_cbse.csv
dataset/indian_student_records_college.csv
dataset/indian_student_records_state_boards.csv
```

The trainer loads every file matching `dataset/indian_student_records*.csv`, except the template file.

## Minimum Useful Columns

For strong prediction, collect at least:

- `education_level`
- `board_or_system`
- `state_ut`
- `class_or_program`
- `degree_level`
- `degree_name`
- `program_area`
- `scoring_system`
- `age`
- `gender`
- `home_location`
- `study_hours_per_week`
- `past_failed_subjects`
- `absences`
- `max_absences`
- `previous_score`
- `latest_score`
- `final_score`

The most important target column is `final_score`. Without it, the model cannot learn.

## Privacy Rules

Do not store student names, roll numbers, phone numbers, emails, Aadhaar numbers, addresses, or parent contact details. Use anonymous rows only.

## Recommended Data Sources

- Your own school/college marksheet exports after removing personal identifiers.
- Google Forms filled by students with consent.
- Public Kaggle datasets for university CGPA, attendance, and habits.
- UCI Student Performance and UCI higher-education dropout/success datasets.
- Zenodo academic performance datasets for university programmes.
- OpenDataBay/Kaggle CGPA forecasting datasets for college semester prediction.

Public Indian board-level student records are hard to find because student marks are private. For CBSE, ICSE, state, and union-territory board accuracy, collect anonymized records directly from schools, coaching centers, or student surveys.

## Training

After adding real files:

```powershell
python Eda\train.py
```

Then run:

```powershell
python -m streamlit run Eda\app.py
```

## Accuracy Expectation

The model becomes board/degree-aware only after you add rows for those boards and degree programs. A good starting target is at least 500-1000 clean rows per major group, and more if you want separate predictions for many boards and degree types.
