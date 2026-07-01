import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DJANGO_DIR = ROOT / "backend" / "django"
FASTAPI_DIR = ROOT / "ai_service"
DJANGO_URL = "http://127.0.0.1:8000/login/"
FASTAPI_URL = "http://127.0.0.1:8001/health"


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DJANGO_SETTINGS_MODULE", "medimind.settings.development")
    env.setdefault("USE_POSTGRES", "False")
    env.setdefault("FASTAPI_URL", "http://127.0.0.1:8001")
    env.setdefault("DISABLE_QDRANT", "true")
    env.setdefault("DISABLE_LLM", "true")
    env.setdefault("DISABLE_ML", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_checked(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"$ {' '.join(args)}")
    completed = subprocess.run(args, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_url(url: str, seconds: int = 30) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except Exception:
            time.sleep(1)
    return False


def seed_preview_account(env: dict[str, str]) -> None:
    script = """
from django.contrib.auth import get_user_model
from health_score.models import HealthScore
from timeline.models import TimelineEvent
from users.models import Allergy, FamilyHistory, MedicalProfile

User = get_user_model()
user, created = User.objects.get_or_create(
    username="hammad",
    defaults={
        "email": "hammad@medimind.local",
        "first_name": "Hammad",
        "last_name": "User",
        "gender": "male",
    },
)
if created:
    user.set_password("MediMind@12345")
    user.save()

MedicalProfile.objects.update_or_create(
    user=user,
    defaults={
        "blood_type": "O+",
        "height_cm": 178,
        "weight_kg": 76,
        "smoking_status": "never",
        "alcohol_use": "none",
        "exercise_frequency": "weekly",
    },
)
Allergy.objects.update_or_create(
    user=user,
    allergen="Penicillin",
    defaults={"severity": "moderate", "notes": "Avoid unless a doctor approves."},
)
FamilyHistory.objects.update_or_create(
    user=user,
    condition="Diabetes",
    relation="Father",
    defaults={"notes": "Type 2"},
)
if not HealthScore.objects.filter(user=user).exists():
    HealthScore.objects.create(
        user=user,
        score=84,
        bmi=23.9,
        sugar_level=96,
        cholesterol=172,
        blood_pressure_systolic=118,
        blood_pressure_diastolic=76,
        lifestyle_score=82,
        strengths=["Healthy BMI", "Normal blood pressure"],
        improvements=["Keep regular exercise"],
    )
if not TimelineEvent.objects.filter(user=user).exists():
    TimelineEvent.objects.create(
        user=user,
        event_type="score_calculated",
        title="Health score calculated",
        description="Sample health score generated for local UI preview.",
        metadata={"source": "run_local"},
    )
print("Preview account ready: username=hammad password=MediMind@12345")
"""
    run_checked([sys.executable, "manage.py", "shell", "-c", script], DJANGO_DIR, env)


def start_process(name: str, args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen | None:
    port = 8001 if name == "FastAPI" else 8000
    if port_open(port):
        print(f"{name} already appears to be running on 127.0.0.1:{port}.")
        return None
    print(f"Starting {name}...")
    return subprocess.Popen(args, cwd=cwd, env=env)


def main() -> int:
    env = local_env()
    print("MediMind AI local preview")
    print("Using SQLite, local preview AI responses, and no Docker.")
    print()

    run_checked([sys.executable, "manage.py", "migrate"], DJANGO_DIR, env)
    seed_preview_account(env)

    processes: list[subprocess.Popen] = []
    fastapi = start_process(
        "FastAPI",
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8001"],
        FASTAPI_DIR,
        env,
    )
    if fastapi:
        processes.append(fastapi)

    django = start_process(
        "Django",
        [sys.executable, "manage.py", "runserver", "127.0.0.1:8000", "--noreload"],
        DJANGO_DIR,
        env,
    )
    if django:
        processes.append(django)

    print()
    print("Waiting for local services...")
    django_ready = wait_for_url(DJANGO_URL, 30)
    fastapi_ready = wait_for_url(FASTAPI_URL, 30)
    print(f"Django:  {'ready' if django_ready else 'not ready'}  {DJANGO_URL}")
    print(f"FastAPI: {'ready' if fastapi_ready else 'not ready'}  {FASTAPI_URL}")
    print()
    print("Login credentials:")
    print("  username: hammad")
    print("  password: MediMind@12345")
    print()

    if django_ready:
        webbrowser.open(DJANGO_URL)

    print("Press Ctrl+C in this terminal to stop services started by this launcher.")
    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    print("A local service stopped. Exiting launcher.")
                    return process.returncode or 0
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping local services...")
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
