"""
Integration tests for the CNN pneumonia detection pipeline.

Tests the full inference chain: image upload → preprocessing → prediction.
Also validates OOD rejection, error handling, and graceful degradation.

Tests that require TensorFlow/Keras are skipped unless keras is installed.
"""
from __future__ import annotations

import importlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cnn.preprocessing import (
    ImageValidationError,
    OODImageError,
    PreprocessedImage,
    preprocess_image,
    validate_image_upload,
    validate_chest_xray,
)
from cnn.config import get_cnn_model_configs
from cnn.registry import CNNModelRegistry

_HAS_KERAS = importlib.util.find_spec("keras") is not None


# ── Upload validation tests ───────────────────────────────────────────────

class TestImageUpload:
    """Test the validate_image_upload function."""

    def test_empty_file_raises(self):
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_upload(b"", "test.jpg", "image/jpeg")

    def test_oversized_file_raises(self):
        big = b"\x00" * (20 * 1024 * 1024 + 1)
        with pytest.raises(ImageValidationError, match="20 MB"):
            validate_image_upload(big, "test.jpg", "image/jpeg")

    def test_invalid_content_type_raises(self):
        with pytest.raises(ImageValidationError, match="Only JPEG.*supported"):
            validate_image_upload(b"test", "test.jpg", "image/gif")

    def test_invalid_extension_raises(self):
        with pytest.raises(ImageValidationError, match="file extension"):
            validate_image_upload(b"test", "test.bmp", None)

    def test_valid_upload_passes(self):
        # Should not raise
        validate_image_upload(b"test_data", "test.jpg", "image/jpeg")

    def test_valid_upload_no_content_type_passes(self):
        validate_image_upload(b"test_data", "test.png", None)


# ── Preprocessing tests ──────────────────────────────────────────────────

class TestPreprocessing:
    """Test the preprocess_image function."""

    config = get_cnn_model_configs()["pneumonia_xray"]

    def test_preprocessing_produces_correct_shape(self, normal_xray_bytes):
        result = preprocess_image(normal_xray_bytes, self.config, "test.jpg", "image/jpeg")
        assert isinstance(result, PreprocessedImage)
        assert result.batch.shape == (1, 224, 224, 3)
        assert result.batch.dtype == np.float32

    def test_preprocessing_normalized(self, normal_xray_bytes):
        result = preprocess_image(normal_xray_bytes, self.config)
        assert result.batch.min() >= 0.0
        assert result.batch.max() <= 1.0

    def test_preprocessing_metadata_includes_dimensions(self, normal_xray_bytes):
        result = preprocess_image(normal_xray_bytes, self.config, "test.jpg", "image/jpeg")
        assert result.metadata["original_image"]["width"] > 0
        assert result.metadata["original_image"]["height"] > 0
        assert result.metadata["filename"] == "test.jpg"

    def test_corrupted_file_raises(self, corrupted_bytes):
        with pytest.raises(ImageValidationError, match="read as an image"):
            preprocess_image(corrupted_bytes, self.config)

    def test_consistency_across_calls(self, normal_xray_bytes):
        """Preprocessing the same image twice should give the same result."""
        r1 = preprocess_image(normal_xray_bytes, self.config)
        r2 = preprocess_image(normal_xray_bytes, self.config)
        np.testing.assert_array_almost_equal(r1.batch, r2.batch)


# ── CNN Prediction tests (require keras) ──────────────────────────────────

@pytest.mark.skipif(not _HAS_KERAS, reason="keras not available")
class TestCNNPrediction:
    """Test the full CNN prediction pipeline."""

    @pytest.fixture(scope="class")
    def registry(self):
        reg = CNNModelRegistry()
        reg.load_all()
        return reg

    def test_model_loaded(self, registry):
        assert "pneumonia_xray" in registry.models, "Model was not loaded"
        status = registry.status()
        assert status["loaded_count"] > 0

    def test_normal_xray_classifies_as_normal(self, registry, normal_xray_bytes):
        result = registry.predict(
            "pneumonia_xray", normal_xray_bytes,
            filename="normal_test.jpg", content_type="image/jpeg"
        )
        assert result["predicted_class"] == "NORMAL"
        assert result["confidence"] > 0.0
        assert result["confidence_percentage"] > 0.0

    def test_pneumonia_xray_classifies_as_pneumonia(self, registry, pneumonia_xray_bytes):
        result = registry.predict(
            "pneumonia_xray", pneumonia_xray_bytes,
            filename="pneumonia_test.jpg", content_type="image/jpeg"
        )
        assert result["predicted_class"] == "PNEUMONIA"
        assert result["confidence"] > 0.0

    def test_prediction_has_full_report(self, registry, normal_xray_bytes):
        result = registry.predict(
            "pneumonia_xray", normal_xray_bytes,
            filename="report_test.jpg", content_type="image/jpeg"
        )
        assert "diagnosis" in result
        assert "assessment_summary" in result
        assert "clinical_recommendations" in result
        assert "formatted_report" in result
        assert "probabilities" in result
        assert "NORMAL" in result["probabilities"]
        assert "PNEUMONIA" in result["probabilities"]

    def test_prediction_probabilities_sum_to_one(self, registry, normal_xray_bytes):
        result = registry.predict(
            "pneumonia_xray", normal_xray_bytes
        )
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-5

    def test_confidence_interpretation_present(self, registry, normal_xray_bytes):
        result = registry.predict(
            "pneumonia_xray", normal_xray_bytes
        )
        assert "confidence_interpretation" in result
        assert "label" in result["confidence_interpretation"]
        assert "percentage" in result["confidence_interpretation"]


# ── OOD rejection tests ───────────────────────────────────────────────────

class TestOODRejection:
    """Test that non-chest-X-ray images are rejected."""

    def test_color_photo_rejected(self, color_photo_bytes):
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(color_photo_bytes, filename="photo.jpg", content_type="image/jpeg")

    def test_solid_red_rejected(self, solid_red_bytes):
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(solid_red_bytes, filename="red.jpg", content_type="image/jpeg")

    def test_extreme_aspect_ratio_rejected(self, extreme_aspect_bytes):
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(extreme_aspect_bytes, filename="wide.jpg", content_type="image/jpeg")

    def test_text_document_rejected(self, text_document_bytes):
        with pytest.raises(OODImageError, match="chest X-ray"):
            validate_chest_xray(text_document_bytes, filename="doc.png", content_type="image/png")

    def test_normal_xray_not_rejected(self, normal_xray_bytes):
        # Should not raise OODImageError
        result = validate_chest_xray(normal_xray_bytes, filename="normal.jpg", content_type="image/jpeg")
        assert "rejected" in result
        assert not result["rejected"]

    def test_pneumonia_xray_not_rejected(self, pneumonia_xray_bytes):
        result = validate_chest_xray(pneumonia_xray_bytes, filename="pneumonia.jpg", content_type="image/jpeg")
        assert not result["rejected"]

    def test_disabled_ood_skips_check(self, color_photo_bytes):
        """When DISABLE_OOD=true, OOD check should be skipped."""
        os.environ["DISABLE_OOD"] = "True"
        try:
            result = validate_chest_xray(color_photo_bytes, filename="photo.jpg", content_type="image/jpeg")
            assert result.get("ood_skipped", False)
        finally:
            os.environ.pop("DISABLE_OOD", None)


# ── Graceful degradation tests ────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_KERAS, reason="keras not available")
class TestGracefulDegradation:
    """Test that the pipeline degrades gracefully when components are missing."""

    def test_unknown_model_id_raises_key_error(self, cnn_registry):
        with pytest.raises(KeyError):
            cnn_registry.predict("nonexistent_model", b"test")

    def test_empty_image_raises_validation_error(self, cnn_registry):
        with pytest.raises(ImageValidationError, match="empty"):
            cnn_registry.predict("pneumonia_xray", b"")


# ── Edge case tests ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases for the CNN inference pipeline."""

    def test_grayscale_image_still_processed(self, solid_gray_bytes):
        """A grayscale image (like an X-ray) should still pass preprocessing."""
        config = get_cnn_model_configs()["pneumonia_xray"]
        result = preprocess_image(solid_gray_bytes, config, "gray.jpg", "image/jpeg")
        assert result.batch.shape == (1, 224, 224, 3)

    def test_png_image_processed(self, normal_xray_bytes):
        """PNG images should work."""
        # Convert test JPEG to PNG bytes
        img = Image.open(io.BytesIO(normal_xray_bytes))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        config = get_cnn_model_configs()["pneumonia_xray"]
        result = preprocess_image(png_bytes, config, "test.png", "image/png")
        assert result.batch.shape == (1, 224, 224, 3)
