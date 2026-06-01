from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
INDIAN_TEMPLATE_PATH = DATA_DIR / "indian_student_records_template.csv"
INDIAN_DATA_GLOB = "indian_student_records*.csv"
RANDOM_STATE = 42
PASS_MARK = 10


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> pd.DataFrame:
    mat_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
    por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

    mat_df["subject"] = "Mathematics"
    por_df["subject"] = "Portuguese"

    df = pd.concat([mat_df, por_df], ignore_index=True)
    grade_columns = ["G1", "G2", "G3"]
    df[grade_columns] = df[grade_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=grade_columns)
    df["education_level"] = "School Board"
    df["board_or_system"] = "Original UCI Portuguese schools"
    df["state_ut"] = "Not applicable"
    df["class_or_program"] = df["subject"]
    df["degree_level"] = "Not applicable"
    df["degree_name"] = "Not applicable"
    df["program_area"] = df["subject"]
    df["scoring_system"] = "Marks out of 20"
    df["institution_type"] = np.where(df["school"] == "GP", "Urban/large campus", "Small town/rural campus")
    return df


def score_to_20(value: float, scoring_system: str) -> float:
    if pd.isna(value):
        return np.nan
    scoring_system = str(scoring_system).strip().lower()
    if "10" in scoring_system:
        scaled = value * 2
    elif "4" in scoring_system:
        scaled = value * 5
    elif "20" in scoring_system:
        scaled = value
    else:
        scaled = value / 5
    return float(max(0, min(20, scaled)))


def study_bucket(hours_per_week: float) -> int:
    if pd.isna(hours_per_week):
        return 2
    if hours_per_week < 2:
        return 1
    if hours_per_week < 5:
        return 2
    if hours_per_week < 10:
        return 3
    return 4


def load_indian_records() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(DATA_DIR.glob(INDIAN_DATA_GLOB)):
        if csv_path.name == INDIAN_TEMPLATE_PATH.name:
            continue
        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    required = {
        "education_level",
        "board_or_system",
        "state_ut",
        "institution_type",
        "class_or_program",
        "degree_level",
        "degree_name",
        "program_area",
        "scoring_system",
        "age",
        "gender",
        "home_location",
        "family_size",
        "parent_status",
        "mother_guardian_education",
        "father_guardian_education",
        "mother_guardian_job",
        "father_guardian_job",
        "institution_choice_reason",
        "primary_guardian",
        "travel_time",
        "study_hours_per_week",
        "past_failed_subjects",
        "extra_educational_support",
        "family_educational_support",
        "paid_classes_or_coaching",
        "activities",
        "nursery_or_foundation",
        "plans_higher_education",
        "internet",
        "relationship",
        "family_relationship_quality",
        "free_time",
        "social_frequency",
        "workday_alcohol",
        "weekend_alcohol",
        "health",
        "absences",
        "max_absences",
        "previous_score",
        "latest_score",
        "final_score",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "Indian training CSV is missing columns: " + ", ".join(sorted(missing))
        )

    df = pd.DataFrame()
    df["school"] = np.where(raw["institution_type"].str.contains("urban|large", case=False, na=False), "GP", "MS")
    df["sex"] = np.where(raw["gender"].str.lower().str.startswith("f"), "F", "M")
    df["age"] = raw["age"].clip(15, 22)
    df["address"] = np.where(raw["home_location"].str.contains("urban|city", case=False, na=False), "U", "R")
    df["famsize"] = raw["family_size"].fillna("GT3")
    df["Pstatus"] = raw["parent_status"].fillna("T")
    df["Medu"] = raw["mother_guardian_education"].clip(0, 4)
    df["Fedu"] = raw["father_guardian_education"].clip(0, 4)
    df["Mjob"] = raw["mother_guardian_job"].fillna("other")
    df["Fjob"] = raw["father_guardian_job"].fillna("other")
    df["reason"] = raw["institution_choice_reason"].fillna("course")
    df["guardian"] = raw["primary_guardian"].fillna("mother")
    df["traveltime"] = raw["travel_time"].clip(1, 4)
    df["studytime"] = raw["study_hours_per_week"].apply(study_bucket)
    df["failures"] = raw["past_failed_subjects"].clip(0, 3)
    df["schoolsup"] = raw["extra_educational_support"].fillna("no")
    df["famsup"] = raw["family_educational_support"].fillna("yes")
    df["paid"] = raw["paid_classes_or_coaching"].fillna("no")
    df["activities"] = raw["activities"].fillna("yes")
    df["nursery"] = raw["nursery_or_foundation"].fillna("yes")
    df["higher"] = raw["plans_higher_education"].fillna("yes")
    df["internet"] = raw["internet"].fillna("yes")
    df["romantic"] = raw["relationship"].fillna("no")
    df["famrel"] = raw["family_relationship_quality"].clip(1, 5)
    df["freetime"] = raw["free_time"].clip(1, 5)
    df["goout"] = raw["social_frequency"].clip(1, 5)
    df["Dalc"] = raw["workday_alcohol"].clip(1, 5)
    df["Walc"] = raw["weekend_alcohol"].clip(1, 5)
    df["health"] = raw["health"].clip(1, 5)
    df["absences"] = np.round(raw["absences"] * 93 / raw["max_absences"].replace(0, np.nan)).fillna(0).clip(0, 93)
    df["G1"] = [
        score_to_20(score, system)
        for score, system in zip(raw["previous_score"], raw["scoring_system"])
    ]
    df["G2"] = [
        score_to_20(score, system)
        for score, system in zip(raw["latest_score"], raw["scoring_system"])
    ]
    df["G3"] = [
        score_to_20(score, system)
        for score, system in zip(raw["final_score"], raw["scoring_system"])
    ]
    df["subject"] = np.where(
        raw["program_area"].str.contains("math|science|commerce|engineering|technology|medical", case=False, na=False),
        "Mathematics",
        "Portuguese",
    )
    context_columns = [
        "education_level",
        "board_or_system",
        "state_ut",
        "class_or_program",
        "degree_level",
        "degree_name",
        "program_area",
        "scoring_system",
        "institution_type",
    ]
    for column in context_columns:
        df[column] = raw[column].fillna("Unknown")

    return df.dropna(subset=["G1", "G2", "G3"])


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", make_one_hot_encoder())])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_regression_models(preprocessor: ColumnTransformer, X_train, y_train):
    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    best_name = None
    best_mae = float("inf")
    best_pipeline = None

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
        )
        mae = -scores["test_mae"].mean()
        r2 = scores["test_r2"].mean()
        results.append({"model": name, "cv_mae": mae, "cv_r2": r2})
        if mae < best_mae:
            best_name = name
            best_mae = mae
            best_pipeline = pipeline

    best_pipeline.fit(X_train, y_train)
    return best_name, best_pipeline, results


def evaluate_classification_models(preprocessor: ColumnTransformer, X_train, y_train):
    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    best_name = None
    best_score = -float("inf")
    best_pipeline = None

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1",
            },
            n_jobs=1,
        )
        accuracy = scores["test_accuracy"].mean()
        balanced_accuracy = scores["test_balanced_accuracy"].mean()
        f1 = scores["test_f1"].mean()
        results.append(
            {
                "model": name,
                "cv_accuracy": accuracy,
                "cv_balanced_accuracy": balanced_accuracy,
                "cv_f1": f1,
            }
        )
        selection_score = (f1 + balanced_accuracy) / 2
        if selection_score > best_score:
            best_name = name
            best_score = selection_score
            best_pipeline = pipeline

    best_pipeline.fit(X_train, y_train)
    return best_name, best_pipeline, results


def save_artifact(path: Path, artifact) -> None:
    with path.open("wb") as f:
        pickle.dump(artifact, f)


def main() -> None:
    print("Loading student datasets...")
    base_df = load_dataset()
    indian_df = load_indian_records()
    if indian_df.empty:
        df = base_df
        print(f"No Indian training CSVs found matching {DATA_DIR / INDIAN_DATA_GLOB}.")
    else:
        df = pd.concat([base_df, indian_df], ignore_index=True)
        print(f"Loaded {len(indian_df)} Indian training rows.")
    print(f"Loaded {df.shape[0]} total rows and {df.shape[1]} columns.")

    X = df.drop(columns=["G3"])
    y_reg = df["G3"]
    y_clf = np.where(df["G3"] >= PASS_MARK, 1, 0)

    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    preprocessor = build_preprocessor(X)

    print("\nTraining and selecting regression model...")
    reg_name, reg_pipeline, reg_cv_results = evaluate_regression_models(
        preprocessor, X_train, y_train_reg
    )
    reg_preds = reg_pipeline.predict(X_test)
    reg_mae = mean_absolute_error(y_test_reg, reg_preds)
    reg_r2 = r2_score(y_test_reg, reg_preds)

    print("\nTraining and selecting classification model...")
    clf_name, clf_pipeline, clf_cv_results = evaluate_classification_models(
        preprocessor, X_train, y_train_clf
    )
    clf_preds = clf_pipeline.predict(X_test)
    clf_accuracy = accuracy_score(y_test_clf, clf_preds)
    clf_balanced_accuracy = balanced_accuracy_score(y_test_clf, clf_preds)
    clf_f1 = f1_score(y_test_clf, clf_preds)

    metadata = {
        "pass_mark": PASS_MARK,
        "feature_columns": X.columns.tolist(),
        "regression_model": reg_name,
        "classification_model": clf_name,
        "regression_cv_results": reg_cv_results,
        "classification_cv_results": clf_cv_results,
        "holdout_metrics": {
            "regression_mae": reg_mae,
            "regression_r2": reg_r2,
            "classification_accuracy": clf_accuracy,
            "classification_balanced_accuracy": clf_balanced_accuracy,
            "classification_f1": clf_f1,
        },
        "indian_training_rows": int(len(indian_df)),
        "indian_training_pattern": str(DATA_DIR / INDIAN_DATA_GLOB),
    }

    print("\nRegression performance")
    print(f"Selected model: {reg_name}")
    print(f"Holdout MAE: {reg_mae:.2f} marks out of 20")
    print(f"Holdout R2: {reg_r2:.3f}")

    print("\nClassification performance")
    print(f"Selected model: {clf_name}")
    print(f"Holdout accuracy: {clf_accuracy * 100:.2f}%")
    print(f"Holdout balanced accuracy: {clf_balanced_accuracy * 100:.2f}%")
    print(f"Holdout F1: {clf_f1:.3f}")
    print("Confusion matrix [[fail, pass], [fail, pass]]:")
    print(confusion_matrix(y_test_clf, clf_preds))
    print(classification_report(y_test_clf, clf_preds, target_names=["Fail", "Pass"]))

    print("\nSaving production artifacts...")
    MODELS_DIR.mkdir(exist_ok=True)
    save_artifact(BASE_DIR / "student_reg_model.pkl", reg_pipeline)
    save_artifact(BASE_DIR / "student_clf_model.pkl", clf_pipeline)
    save_artifact(BASE_DIR / "student_model_metadata.pkl", metadata)
    save_artifact(MODELS_DIR / "student_reg_model.pkl", reg_pipeline)
    save_artifact(MODELS_DIR / "student_clf_model.pkl", clf_pipeline)
    save_artifact(MODELS_DIR / "student_model.pkl", clf_pipeline)
    save_artifact(MODELS_DIR / "student_model_metadata.pkl", metadata)
    print("Done. Models saved in the project root and models directory.")


if __name__ == "__main__":
    main()
