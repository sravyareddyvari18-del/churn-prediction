import subprocess
import sys


def run_step(description, command):
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)

    result = subprocess.run(
        [sys.executable] + command,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(
            f"\nERROR: {description} failed."
        )


print("=" * 60)
print("LAB 3 - BASELINE ML PIPELINE")
print("=" * 60)


# Step 1: Preprocessing
run_step(
    "STEP 1: DATA PREPROCESSING",
    ["src/preprocess.py"]
)


# Step 2: Train baseline models
run_step(
    "STEP 2: MODEL TRAINING",
    ["src/train.py"]
)


# Step 3: Evaluate models
run_step(
    "STEP 3: MODEL EVALUATION",
    ["src/evaluate.py"]
)


print("\n" + "=" * 60)
print("LAB 3 BASELINE PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)