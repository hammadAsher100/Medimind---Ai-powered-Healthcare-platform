import pandas as pd


def add_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bmi = df.get("bmi", 0)
    age = df.get("age", 0)
    glucose = df.get("glucose", 0)
    df["bmi_category"] = pd.cut(bmi, bins=[0, 18.5, 24.9, 29.9, 10_000], labels=[0, 1, 2, 3]).astype(float)
    df["age_group"] = pd.cut(age, bins=[0, 30, 45, 60, 10_000], labels=[0, 1, 2, 3]).astype(float)
    df["glucose_category"] = pd.cut(glucose, bins=[0, 99, 125, 10_000], labels=[0, 1, 2]).astype(float)
    return df


def add_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cholesterol_ratio"] = df.get("cholesterol", 0) / df.get("resting_bp", 1).replace(0, 1)
    df["thalach_age_interaction"] = df.get("max_heart_rate", 0) / df.get("age", 1).replace(0, 1)
    return df


def add_kidney_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["creatinine_bun_ratio"] = df.get("serum_creatinine", 0) / df.get("blood_urea", 1).replace(0, 1)
    df["anemia_flag"] = df.get("anemia", 0)
    return df


def add_stroke_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hypertension_heart_disease_flag"] = ((df.get("hypertension", 0) == 1) & (df.get("heart_disease", 0) == 1)).astype(int)
    return df


FEATURE_ENGINEERING = {
    "diabetes": add_diabetes_features,
    "heart": add_heart_features,
    "kidney": add_kidney_features,
    "stroke": add_stroke_features,
}
