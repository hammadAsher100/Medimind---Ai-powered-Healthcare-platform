"""
Shared pytest fixtures for CNN inference and OOD detection tests.

All fixtures are module-scoped (loaded once per test module) for performance.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

# ── Ensure the ai_service directory is on the Python path ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

TESTING_DIR = PROJECT_ROOT / "testing"


@pytest.fixture(scope="module")
def testing_dir() -> Path:
    """Path to the testing directory containing sample images."""
    return TESTING_DIR


@pytest.fixture(scope="module")
def project_root() -> Path:
    return PROJECT_ROOT


# ── Image fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def normal_xray_bytes() -> bytes:
    """Bytes of a valid NORMAL chest X-ray."""
    path = TESTING_DIR / "Chest-Xray-Normal.jpeg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path.read_bytes()


@pytest.fixture(scope="module")
def pneumonia_xray_bytes() -> bytes:
    """Bytes of a valid PNEUMONIA chest X-ray."""
    path = TESTING_DIR / "Chest-Xray-Pneuomia.jpeg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path.read_bytes()


@pytest.fixture(scope="module")
def color_photo_bytes() -> bytes:
    """Bytes simulating a color photo (not an X-ray)."""
    img = Image.new("RGB", (800, 600), (120, 180, 240))
    # Add some color variation to make it look more like a photo
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-30, 30, (400, 600, 3), dtype=np.int16)
    arr[100:500, 100:700, :] = np.clip(arr[100:500, 100:700, :] + noise, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def solid_gray_bytes() -> bytes:
    """Bytes of a solid gray image (low channel variance, like an X-ray)."""
    img = Image.new("L", (224, 224), 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def solid_red_bytes() -> bytes:
    """Bytes of a solid red image (high channel variance, not an X-ray)."""
    img = Image.new("RGB", (224, 224), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def empty_bytes() -> bytes:
    """Empty byte string."""
    return b""


@pytest.fixture(scope="module")
def corrupted_bytes() -> bytes:
    """Random garbage bytes that can't be decoded as an image."""
    return b"\x00\x01\x02\xff\xfe\xfd\xfc\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"


@pytest.fixture(scope="module")
def text_document_bytes() -> bytes:
    """White background with dark text pattern — simulates a document."""
    img = Image.new("L", (800, 600), 245)
    arr = np.array(img)
    # Add dense text-like pattern
    for row in range(0, 600, 15):
        arr[row:row + 3, 50:750] = 30
    for col in range(0, 800, 10):
        arr[30:570, col:col + 1] = 30
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def extreme_aspect_bytes() -> bytes:
    """Very wide image (16:1 aspect ratio), not a chest X-ray shape."""
    img = Image.new("RGB", (1600, 100), (100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def cnn_registry():
    """CNNModelRegistry instance with models loaded."""
    # Save original DISABLE_OOD value before overriding
    _ood_saved = os.environ.pop("DISABLE_OOD", None)
    os.environ["DISABLE_OOD"] = "True"
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "ai_service"))
        from cnn.registry import CNNModelRegistry
        registry = CNNModelRegistry()
        registry.load_all()
    except Exception as exc:
        pytest.skip(f"Could not load CNN registry: {exc}")
    finally:
        # Restore original DISABLE_OOD value to avoid leaking into other tests
        os.environ.pop("DISABLE_OOD", None)
        if _ood_saved is not None:
            os.environ["DISABLE_OOD"] = _ood_saved
    return registry
