"""
Unit tests for the OOD (Out-of-Distribution) detection module.

Tests each image heuristic independently, then the combined
validate_chest_xray function.
"""
from __future__ import annotations

import importlib
import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from cnn.preprocessing import (
    OODImageError,
    check_grayscale_ratio,
    check_aspect_ratio,
    check_edge_density,
    check_brightness_distribution,
    validate_chest_xray,
)


# ── Grayscale ratio tests ────────────────────────────────────────────────

class TestGrayscaleRatio:
    """Test the grayscale ratio check."""

    def test_grayscale_image_high_ratio(self):
        """A truly grayscale image should pass."""
        img = Image.new("L", (224, 224), 128)
        ratio, passes = check_grayscale_ratio(img)
        assert passes, f"Grayscale image failed with ratio={ratio:.4f}"
        assert ratio >= 0.90

    def test_color_image_low_ratio(self):
        """A strongly colored image should fail."""
        # Red and green halves produce high channel variance
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :112, 0] = 255  # Left half: red
        arr[:, 112:, 1] = 255  # Right half: green
        img = Image.fromarray(arr)
        ratio, passes = check_grayscale_ratio(img)
        assert not passes, f"Color image passed with ratio={ratio:.4f}"

    def test_nearly_gray_image_passes(self):
        """An image with very subtle color should pass."""
        arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        arr[:, :, 0] = 130  # Slightly more red
        arr[:, :, 1] = 126  # Slightly less green
        img = Image.fromarray(arr)
        ratio, passes = check_grayscale_ratio(img)
        assert passes, f"Nearly gray image failed with ratio={ratio:.4f}"


# ── Aspect ratio tests ───────────────────────────────────────────────────

class TestAspectRatio:
    """Test the aspect ratio check."""

    def test_portrait_xray_passes(self):
        """A portrait chest X-ray (typical dimension) should pass."""
        img = Image.new("RGB", (1400, 1600), 128)
        ratio, passes = check_aspect_ratio(img)
        assert passes, f"Portrait X-ray failed with ratio={ratio:.4f}"

    def test_landscape_xray_passes(self):
        """A landscape X-ray should pass."""
        img = Image.new("RGB", (1600, 1200), 128)
        ratio, passes = check_aspect_ratio(img)
        assert passes, f"Landscape X-ray failed with ratio={ratio:.4f}"

    def test_extreme_wide_fails(self):
        """A very wide image should fail."""
        img = Image.new("RGB", (3000, 200), 128)
        ratio, passes = check_aspect_ratio(img)
        assert not passes, f"Extreme wide image passed with ratio={ratio:.4f}"

    def test_extreme_tall_fails(self):
        """A very tall image should fail."""
        img = Image.new("RGB", (200, 3000), 128)
        ratio, passes = check_aspect_ratio(img)
        assert not passes, f"Extreme tall image passed with ratio={ratio:.4f}"

    def test_square_image_passes(self):
        """A square image should pass."""
        img = Image.new("RGB", (1024, 1024), 128)
        ratio, passes = check_aspect_ratio(img)
        assert passes, f"Square image failed with ratio={ratio:.4f}"


# ── Edge density tests ───────────────────────────────────────────────────

class TestEdgeDensity:
    """Test the edge density check."""

    def test_smooth_xray_has_low_edge_density(self):
        """An X-ray has smooth transitions, so edge density should be low."""
        # Create a smooth gradient image
        x = np.linspace(0, 1, 224)
        y = np.linspace(0, 1, 224)
        xx, yy = np.meshgrid(x, y)
        arr = (np.sin(xx * 3) * np.cos(yy * 3) * 100 + 128).astype(np.uint8)
        img = Image.fromarray(arr)
        density, passes = check_edge_density(img)
        assert passes, f"Smooth gradient failed with density={density:.4f}"

    def test_text_document_has_high_edge_density(self):
        """A document with sharp text should have high edge density."""
        arr = np.ones((224, 224), dtype=np.uint8) * 245
        # Dense sharp lines (like text)
        arr[::15, 20:200] = 30
        arr[10:200, ::10] = 30
        img = Image.fromarray(arr)
        density, passes = check_edge_density(img)
        assert passes, f"Dense text image should have moderate density, got {density:.4f}"


# ── Brightness distribution tests ────────────────────────────────────────

class TestBrightnessDistribution:
    """Test the brightness distribution check."""

    def test_full_dynamic_range_passes(self):
        """An image using full dynamic range should pass."""
        arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        img = Image.fromarray(arr)
        spread, passes = check_brightness_distribution(img)
        assert passes, f"Full range image failed with spread={spread:.4f}"

    def test_constant_image_fails(self):
        """A constant-value image should fail."""
        arr = np.full((224, 224), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        spread, passes = check_brightness_distribution(img)
        assert not passes, f"Constant image passed with spread={spread:.4f}"

    def test_narrow_range_fails(self):
        """An image with very narrow brightness range should fail."""
        arr = np.random.randint(120, 130, (224, 224), dtype=np.uint8)
        img = Image.fromarray(arr)
        spread, passes = check_brightness_distribution(img)
        assert not passes, f"Narrow range image passed with spread={spread:.4f}"


# ── Combined validate_chest_xray tests ───────────────────────────────────

class TestValidateChestXray:
    """Test the combined validate_chest_xray function."""

    def test_normal_xray_validation_passes(self, normal_xray_bytes):
        """A real NORMAL chest X-ray should pass all checks."""
        result = validate_chest_xray(normal_xray_bytes, filename="normal.jpg", content_type="image/jpeg")
        assert result.get("grayscale_pass", False), "Failed grayscale check"
        assert result.get("aspect_pass", False), "Failed aspect ratio check"
        assert result.get("edge_pass", False), "Failed edge density check"
        assert result.get("histogram_pass", False), "Failed histogram check"
        assert not result["rejected"]

    def test_pneumonia_xray_validation_passes(self, pneumonia_xray_bytes):
        """A real PNEUMONIA chest X-ray should pass all checks."""
        result = validate_chest_xray(pneumonia_xray_bytes, filename="pneumonia.jpg", content_type="image/jpeg")
        assert not result["rejected"]

    def test_color_photo_validation_fails(self, color_photo_bytes):
        """A color photo should be rejected."""
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(color_photo_bytes, filename="photo.jpg", content_type="image/jpeg")

    def test_solid_red_validation_fails(self, solid_red_bytes):
        """A solid red image should be rejected."""
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(solid_red_bytes, filename="red.jpg", content_type="image/jpeg")

    def test_extreme_aspect_validation_fails(self, extreme_aspect_bytes):
        """An image with extreme aspect ratio should be rejected."""
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(extreme_aspect_bytes, filename="wide.jpg", content_type="image/jpeg")

    def test_disabled_ood_skips_all_checks(self, color_photo_bytes):
        """When DISABLE_OOD=true, validation returns immediately."""
        os.environ["DISABLE_OOD"] = "True"
        try:
            result = validate_chest_xray(color_photo_bytes, filename="photo.jpg", content_type="image/jpeg")
            assert result.get("ood_skipped", False)
        finally:
            os.environ.pop("DISABLE_OOD", None)

    def test_empty_bytes_returns_ood_skipped(self, empty_bytes):
        """Empty bytes should cause OOD to skip (let normal validation handle)."""
        result = validate_chest_xray(empty_bytes, filename="empty.jpg", content_type="image/jpeg")
        # Should return with ood_skipped rather than crashing
        assert result.get("ood_skipped", False)

    def test_corrupted_bytes_returns_ood_skipped(self, corrupted_bytes):
        """Corrupted bytes should cause OOD to skip gracefully."""
        result = validate_chest_xray(corrupted_bytes, filename="bad.jpg", content_type="image/jpeg")
        assert result.get("ood_skipped", False)


# ── OOD + registry integration test ───────────────────────────────────────

class TestOODIntegration:
    """Test OOD check integrated with the CNN registry.

    These tests require TensorFlow/Keras and a loaded model.
    """

    @pytest.fixture(autouse=True)
    def check_deps(self):
        import importlib
        if importlib.util.find_spec("keras") is None:
            pytest.skip("keras not available — skipping registry-dependent tests")
        # Also check if model file exists
        from pathlib import Path
        import sys
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root / "ai_service"))
        from cnn.config import get_cnn_model_configs
        config = get_cnn_model_configs()["pneumonia_xray"]
        if not config.model_path.exists():
            pytest.skip(f"Model not found at {config.model_path}")

    def test_registry_rejects_color_photo(self, color_photo_bytes, cnn_registry):
        """The CNN registry should reject non-X-ray images via OOD."""
        os.environ.pop("DISABLE_OOD", None)
        try:
            with pytest.raises(OODImageError):
                cnn_registry.predict(
                    "pneumonia_xray", color_photo_bytes,
                    filename="photo.jpg", content_type="image/jpeg"
                )
        finally:
            os.environ["DISABLE_OOD"] = "True"

    def test_registry_accepts_xray_with_ood_enabled(self, normal_xray_bytes, cnn_registry):
        """With OOD enabled, valid X-rays should still pass."""
        os.environ.pop("DISABLE_OOD", None)
        try:
            result = cnn_registry.predict(
                "pneumonia_xray", normal_xray_bytes,
                filename="normal.jpg", content_type="image/jpeg"
            )
            assert result["predicted_class"] in ("NORMAL", "PNEUMONIA")
        finally:
            os.environ["DISABLE_OOD"] = "True"
