from pathlib import Path
import json
import pandas as pd


# Expected columns for the Telco Customer Churn dataset
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_schema(df, output_report_name="data_validation.csv"):
    """
    Validate the schema and basic data quality of a churn dataset.

    Returns:
        True  -> validation passed
        False -> validation failed
    """

    print("[INFO] Starting dataset schema validation...")

    errors = []
    warnings = []

    # ---------------------------------------------------------
    # 1. Check dataframe
    # ---------------------------------------------------------
    if df is None:
        errors.append("Dataset is None.")
        return _save_validation_report(
            errors,
            warnings,
            output_report_name
        )

    if not isinstance(df, pd.DataFrame):
        errors.append("Input is not a pandas DataFrame.")
        return _save_validation_report(
            errors,
            warnings,
            output_report_name
        )

    # ---------------------------------------------------------
    # 2. Check empty dataset
    # ---------------------------------------------------------
    if df.empty:
        errors.append("Dataset is empty.")

    # ---------------------------------------------------------
    # 3. Check required columns
    # ---------------------------------------------------------
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 4. Check duplicate customer IDs
    # ---------------------------------------------------------
    if "customerID" in df.columns:

        duplicate_ids = df["customerID"].duplicated().sum()

        if duplicate_ids > 0:
            errors.append(
                f"Found {duplicate_ids} duplicate customerID values."
            )

    # ---------------------------------------------------------
    # 5. Check missing values
    # ---------------------------------------------------------
    missing_values = df.isnull().sum()

    missing_columns_info = {
        column: int(count)
        for column, count in missing_values.items()
        if count > 0
    }

    if missing_columns_info:
        warnings.append(
            f"Missing values detected: {missing_columns_info}"
        )

    # ---------------------------------------------------------
    # 6. Check Churn column
    # ---------------------------------------------------------
    if "Churn" in df.columns:

        valid_churn_values = {
            "Yes",
            "No",
            1,
            0,
            "1",
            "0"
        }

        actual_values = set(
            df["Churn"].dropna().unique()
        )

        invalid_churn_values = [
            value
            for value in actual_values
            if value not in valid_churn_values
        ]

        if invalid_churn_values:
            errors.append(
                f"Invalid Churn values found: "
                f"{invalid_churn_values}"
            )

    # ---------------------------------------------------------
    # 7. Check numerical columns
    # ---------------------------------------------------------
    numerical_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for column in numerical_columns:

        if column not in df.columns:
            continue

        if column == "TotalCharges":

            # TotalCharges sometimes contains blank strings
            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            invalid_count = converted.isna().sum()

            # Don't treat blanks as fatal because the original
            # Telco dataset can contain blank TotalCharges.
            if invalid_count > 0:
                warnings.append(
                    f"{column} contains "
                    f"{invalid_count} non-numeric/blank values."
                )

        else:

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            invalid_count = converted.isna().sum()

            if invalid_count > 0:
                errors.append(
                    f"{column} contains "
                    f"{invalid_count} non-numeric values."
                )

    # ---------------------------------------------------------
    # 8. Check tenure range
    # ---------------------------------------------------------
    if "tenure" in df.columns:

        tenure = pd.to_numeric(
            df["tenure"],
            errors="coerce"
        )

        invalid_tenure = (
            (tenure < 0) |
            (tenure > 100)
        ).sum()

        if invalid_tenure > 0:
            errors.append(
                f"Found {invalid_tenure} invalid tenure values."
            )

    # ---------------------------------------------------------
    # 9. Check MonthlyCharges
    # ---------------------------------------------------------
    if "MonthlyCharges" in df.columns:

        monthly_charges = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce"
        )

        negative_values = (
            monthly_charges < 0
        ).sum()

        if negative_values > 0:
            errors.append(
                f"Found {negative_values} negative "
                f"MonthlyCharges values."
            )

    # ---------------------------------------------------------
    # 10. Check TotalCharges
    # ---------------------------------------------------------
    if "TotalCharges" in df.columns:

        total_charges = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        negative_values = (
            total_charges < 0
        ).sum()

        if negative_values > 0:
            errors.append(
                f"Found {negative_values} negative "
                f"TotalCharges values."
            )

    # ---------------------------------------------------------
    # 11. Print validation result
    # ---------------------------------------------------------
    if errors:

        print("[ERROR] Dataset schema validation failed.")

        for error in errors:
            print(f"[ERROR] {error}")

        if warnings:
            print("[WARNING] Validation warnings:")

            for warning in warnings:
                print(f"[WARNING] {warning}")

        _save_validation_report(
            errors,
            warnings,
            output_report_name
        )

        return False

    print("[SUCCESS] Dataset schema validation passed.")

    if warnings:

        print("[WARNING] Dataset contains warnings:")

        for warning in warnings:
            print(f"[WARNING] {warning}")

    _save_validation_report(
        errors,
        warnings,
        output_report_name
    )

    return True


def _save_validation_report(
    errors,
    warnings,
    output_report_name
):
    """
    Save validation results as JSON.
    """

    report = {
        "validation_status": (
            "passed"
            if not errors
            else "failed"
        ),
        "errors": errors,
        "warnings": warnings,
    }

    # Save inside artifacts directory
    report_dir = Path("artifacts")
    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove .csv extension if supplied
    report_name = Path(
        output_report_name
    ).stem

    report_path = (
        report_dir /
        f"{report_name}.json"
    )

    try:

        with report_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=2
            )

        print(
            f"[INFO] Validation report saved to: "
            f"{report_path}"
        )

    except OSError as error:

        print(
            f"[WARNING] Could not save validation report: "
            f"{error}"
        )

    return not errors
