
import json
import subprocess
import sys
from pathlib import Path

import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"

PREPARE_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "mlops_pipeline"
    / "11_prepare_deployment.py"
)

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
MODEL_PATH = DEPLOYMENT_DIR / "model.pkl"
PREPROCESSOR_PATH = DEPLOYMENT_DIR / "preprocessor.pkl"

K8S_DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "deployment.yaml"

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "deployment_automation_report.json"
)

IMAGE_NAME = "churn-prediction-api"


# ============================================================
# LOAD LIFECYCLE REPORT
# ============================================================

def load_lifecycle_report():

    if not LIFECYCLE_REPORT.exists():
        raise FileNotFoundError(
            f"Lifecycle report not found: {LIFECYCLE_REPORT}"
        )

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# RUN PREPARATION SCRIPT
# ============================================================

def run_preparation():

    if not PREPARE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Preparation script not found: {PREPARE_SCRIPT}"
        )

    print("[INFO] Preparing Production model artifacts...")

    result = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        raise RuntimeError("Deployment preparation failed.")

    print("[SUCCESS] Deployment artifacts prepared.")


# ============================================================
# LOAD DEPLOYMENT METADATA
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Deployment metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD KUBERNETES CONFIGURATION
# ============================================================

def load_kubernetes_config():

    if not K8S_DEPLOYMENT_PATH.exists():
        raise FileNotFoundError(
            f"Kubernetes deployment file not found: "
            f"{K8S_DEPLOYMENT_PATH}"
        )

    with K8S_DEPLOYMENT_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    deployment_name = config["metadata"]["name"]

    app_label = config["metadata"]["labels"]["app"]

    container_name = (
        config["spec"]["template"]["spec"]["containers"][0]["name"]
    )

    return deployment_name, app_label, container_name


# ============================================================
# RUN DOCKER / KUBERNETES COMMANDS
# ============================================================

def run_command(command):

    print(f"[INFO] Running: {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )

    return result


# ============================================================
# SAVE DEPLOYMENT REPORT
# ============================================================

def save_report(report):

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"[SUCCESS] Deployment report saved: {REPORT_PATH}")


# ============================================================
# PRODUCTION DEPLOYMENT
# ============================================================

def deploy_production():

    print("=" * 70)
    print("PRODUCTION DEPLOYMENT")
    print("=" * 70)

    metadata = load_metadata()

    (
        deployment_name,
        app_label,
        container_name
    ) = load_kubernetes_config()

    model_name = metadata["model_name"]

    model_version = metadata["model_version"]

    model_stage = metadata["model_stage"]

    run_id = metadata["run_id"]

    # Verify model stage
    if model_stage != "Production":

        raise RuntimeError(
            f"Deployment blocked. Model stage is "
            f"'{model_stage}', not 'Production'."
        )

    # Verify deployment artifacts
    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():

        raise FileNotFoundError(
            f"Preprocessor artifact not found: "
            f"{PREPROCESSOR_PATH}"
        )

    image_tag = f"{IMAGE_NAME}:model-v{model_version}"

    print(f"[INFO] Model         : {model_name}")
    print(f"[INFO] Model version: {model_version}")
    print(f"[INFO] Run ID       : {run_id}")
    print(f"[INFO] Docker image : {image_tag}")
    print(f"[INFO] Kubernetes   : {deployment_name}")
    print(f"[INFO] Container    : {container_name}")

    # --------------------------------------------------------
    # DOCKER IMAGE CHECK
    # --------------------------------------------------------

    print("[INFO] Checking Docker image...")

    image_check = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    image_exists = image_check.returncode == 0

    if image_exists:

        print(
            f"[INFO] Docker image already exists: {image_tag}"
        )

        print("[INFO] Reusing existing image.")

    else:

        print("[INFO] Docker image not found. Building...")

        run_command([
            "docker",
            "build",
            "-t",
            image_tag,
            "."
        ])

        print("[SUCCESS] Docker image built successfully.")

    # --------------------------------------------------------
    # UPDATE KUBERNETES DEPLOYMENT
    # --------------------------------------------------------

    print("[INFO] Updating Kubernetes deployment...")

    run_command([
        "kubectl",
        "set",
        "image",
        f"deployment/{deployment_name}",
        f"{container_name}={image_tag}"
    ])

    print("[SUCCESS] Kubernetes deployment updated.")

    # --------------------------------------------------------
    # WAIT FOR ROLLOUT
    # --------------------------------------------------------

    print("[INFO] Waiting for Kubernetes rollout...")

    run_command([
        "kubectl",
        "rollout",
        "status",
        f"deployment/{deployment_name}",
        "--timeout=120s"
    ])

    print("[SUCCESS] Kubernetes rollout completed.")

    # --------------------------------------------------------
    # DEPLOYMENT REPORT
    # --------------------------------------------------------

    report = {

        "status": "success",

        "model_name": model_name,

        "model_version": str(model_version),

        "run_id": run_id,

        "docker_image": image_tag,

        "kubernetes_context": "docker-desktop",

        "kubernetes_deployment": deployment_name,

        "kubernetes_container": container_name,

        "app_label": app_label,

        "deployment_bundle": {

            "model": str(
                MODEL_PATH.relative_to(PROJECT_ROOT)
            ),

            "preprocessor": str(
                PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)
            ),

            "metadata": str(
                METADATA_PATH.relative_to(PROJECT_ROOT)
            )
        }
    }

    save_report(report)

    print("=" * 70)
    print("[SUCCESS] Production deployment completed!")
    print("[SUCCESS] Docker image built or reused.")
    print("[SUCCESS] Kubernetes deployment updated.")
    print("[SUCCESS] Kubernetes rollout completed.")
    print("=" * 70)

    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("DEPLOY IF MODEL WAS PROMOTED")
    print("=" * 70)

    report = load_lifecycle_report()

    promoted = report.get("promoted", False)

    candidate_version = report.get("candidate_version")

    production_version = report.get(
        "production_version_after"
    )

    print(f"[INFO] Candidate version : {candidate_version}")
    print(f"[INFO] Promoted          : {promoted}")
    print(f"[INFO] Production version: {production_version}")

    # --------------------------------------------------------
    # SKIP IF MODEL NOT PROMOTED
    # --------------------------------------------------------

    if not promoted:

        print(
            "[INFO] Model was not promoted. "
            "Production deployment skipped."
        )

        return True

    print(
        f"[INFO] Model Version {candidate_version} was promoted."
    )

    # --------------------------------------------------------
    # PREPARE DEPLOYMENT ARTIFACTS
    # --------------------------------------------------------

    run_preparation()

    # --------------------------------------------------------
    # PRODUCTION DEPLOYMENT
    # --------------------------------------------------------

    print("[INFO] Starting Production deployment...")

    deploy_production()

    print("[SUCCESS] Production deployment completed.")

    return True


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        success = main()

        sys.exit(0 if success else 1)

    except Exception as error:

        print(f"[ERROR] Production deployment failed: {error}")

        sys.exit(1)
