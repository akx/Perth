import numpy as np
import pytest

from perth import DummyWatermarker
from perth.utils import calculate_audio_metrics

SAMPLE_RATE = 44100


@pytest.fixture
def test_audio():
    t = np.linspace(0, 1, SAMPLE_RATE)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def watermarker():
    return DummyWatermarker()


def test_apply_watermark(watermarker, test_audio):
    watermarked = watermarker.apply_watermark(test_audio, sample_rate=SAMPLE_RATE)
    assert watermarked.shape == test_audio.shape


def test_get_watermark(watermarker, test_audio):
    watermarked = watermarker.apply_watermark(test_audio, sample_rate=SAMPLE_RATE)
    watermark = watermarker.get_watermark(watermarked, sample_rate=SAMPLE_RATE)
    assert isinstance(watermark, np.ndarray)
    assert len(watermark) == 32


def test_custom_watermark_length(watermarker, test_audio):
    watermarked = watermarker.apply_watermark(test_audio, sample_rate=SAMPLE_RATE)
    custom_length = 64
    watermark = watermarker.get_watermark(
        watermarked, sample_rate=SAMPLE_RATE, watermark_length=custom_length
    )
    assert len(watermark) == custom_length


def test_calculate_metrics():
    t = np.linspace(0, 1)
    original = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, len(original))
    modified = original + noise
    metrics = calculate_audio_metrics(original, modified)

    assert "snr" in metrics
    assert "mse" in metrics
    assert "psnr" in metrics

    assert metrics["snr"] > 0
    assert metrics["mse"] > 0
    assert metrics["mse"] < 0.1
    assert metrics["psnr"] > 0
