"""
Tests for the custom event API endpoint and related functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import (
    router,
    _ensure_custom_memory_file,
    _load_custom_memory,
    _save_custom_memory,
    _build_custom_anomaly_async,
)


@pytest.fixture
def custom_memory_temp_dir():
    """Create a temporary directory for custom memory files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def custom_memory_file(custom_memory_temp_dir):
    """Create a temporary custom memory file."""
    custom_path = custom_memory_temp_dir / "custom_anomaly_memory.json"
    return custom_path


@pytest.fixture
def mock_runner():
    """Create a mock runner with necessary VAD properties."""
    runner = Mock()
    runner.vad = Mock()
    runner.vad.device = "cpu"
    runner.vad.rtvad_root = "/mock/rtvad"
    runner.vad.reload_memory = Mock()
    return runner


@pytest.fixture
def app_with_runner(mock_runner):
    """Create FastAPI app with mock runner in state."""
    app = FastAPI()
    app.include_router(router)
    app.state.runner = mock_runner
    return app


@pytest.fixture
def test_client(app_with_runner):
    """Create a FastAPI TestClient."""
    return TestClient(app_with_runner)


@pytest.fixture
def app_with_no_runner():
    """Create FastAPI app without a runner."""
    app = FastAPI()
    app.include_router(router)
    app.state.runner = None
    return app


@pytest.fixture
def test_client_no_runner(app_with_no_runner):
    """Create a TestClient without runner."""
    return TestClient(app_with_no_runner)


@pytest.mark.unit
class TestCustomMemoryOperations:
    """Tests for custom memory file operations."""

    def test_ensure_custom_memory_file_creates_new(self, custom_memory_file):
        """Test creating a new custom memory file."""
        assert not custom_memory_file.exists()
        
        _ensure_custom_memory_file(custom_memory_file)
        
        assert custom_memory_file.exists()
        data = json.loads(custom_memory_file.read_text())
        assert data == {"custom_anomalies": []}

    def test_ensure_custom_memory_file_preserves_existing(self, custom_memory_file):
        """Test that existing file is not overwritten."""
        # Create file with some data
        initial_data = {"custom_anomalies": [{"id": "test-1", "text": "test event"}]}
        custom_memory_file.parent.mkdir(parents=True, exist_ok=True)
        custom_memory_file.write_text(json.dumps(initial_data))
        
        _ensure_custom_memory_file(custom_memory_file)
        
        data = json.loads(custom_memory_file.read_text())
        assert data == initial_data

    def test_load_custom_memory_empty(self, custom_memory_file):
        """Test loading empty custom memory."""
        items = _load_custom_memory(custom_memory_file)
        
        assert items == []
        assert custom_memory_file.exists()

    def test_load_custom_memory_with_data(self, custom_memory_file):
        """Test loading custom memory with existing data."""
        test_items = [
            {"id": "1", "text": "person falling", "embedding": [0.1, 0.2]},
            {"id": "2", "text": "unattended bag", "embedding": [0.3, 0.4]},
        ]
        _save_custom_memory(custom_memory_file, test_items)
        
        items = _load_custom_memory(custom_memory_file)
        
        assert items == test_items
        assert len(items) == 2

    def test_save_and_load_custom_memory(self, custom_memory_file):
        """Test saving and loading custom memory."""
        test_items = [
            {
                "id": "uuid-1",
                "text": "unauthorized entry",
                "category": "user_defined_anomaly",
                "label": 1,
                "embedding": [0.1, 0.2, 0.3],
                "created_at": 1234567890.0,
            }
        ]
        
        _save_custom_memory(custom_memory_file, test_items)
        loaded_items = _load_custom_memory(custom_memory_file)
        
        assert loaded_items == test_items
        assert loaded_items[0]["text"] == "unauthorized entry"
        assert loaded_items[0]["category"] == "user_defined_anomaly"


@pytest.mark.unit
class TestCustomAnomalyEndpoint:
    """Tests for /pipeline/custom-anomaly endpoint."""

    def test_post_custom_anomaly_success(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test successful custom anomaly creation."""
        # Mock the rtvad_root path
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": "person climbing without harness"}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["added"] is True
        assert data["text"] == "person climbing without harness"
        assert data["status"] == "building_embedding"

    def test_post_custom_anomaly_empty_text(self, test_client):
        """Test that empty text is rejected."""
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": ""}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Text is required" in data.get("error", "")

    def test_post_custom_anomaly_whitespace_only(self, test_client):
        """Test that whitespace-only text is rejected."""
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": "   \n\t  "}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Text is required" in data.get("error", "")

    def test_post_custom_anomaly_missing_text(self, test_client):
        """Test that missing text field is rejected."""
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Text is required" in data.get("error", "")

    def test_post_custom_anomaly_no_runner(self, test_client_no_runner):
        """Test that endpoint fails gracefully when runner is not available."""
        response = test_client_no_runner.post(
            "/pipeline/custom-anomaly",
            json={"text": "test event"}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "Pipeline is not running" in data.get("error", "")

    def test_post_custom_anomaly_duplicate(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test that duplicate anomalies are rejected."""
        import time
        
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        custom_path = custom_memory_temp_dir / "custom_anomaly_memory.json"
        
        # Pre-populate the memory with an existing event
        _save_custom_memory(custom_path, [
            {
                "id": "pre-existing-1",
                "text": "person falling",
                "embedding": [0.1, 0.2],
                "created_at": 1234567890.0,
            }
        ])
        
        # Try to add duplicate
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": "person falling"}
        )
        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data.get("error", "").lower()

    def test_post_custom_anomaly_duplicate_case_insensitive(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test that duplicate detection is case-insensitive."""
        import time
        
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        custom_path = custom_memory_temp_dir / "custom_anomaly_memory.json"
        
        # Pre-populate with uppercase version
        _save_custom_memory(custom_path, [
            {
                "id": "pre-existing-1",
                "text": "Person Falling",
                "embedding": [0.1, 0.2],
                "created_at": 1234567890.0,
            }
        ])
        
        # Try to add with different case
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": "person falling"}
        )
        assert response.status_code == 409

    def test_post_custom_anomaly_invalid_json(self, test_client):
        """Test that invalid JSON is handled."""
        response = test_client.post(
            "/pipeline/custom-anomaly",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # FastAPI returns 422 for validation errors, but the route catches JSON parsing and returns 500
        assert response.status_code in [400, 422, 500]

    def test_post_custom_anomaly_special_characters(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test creating anomaly with special characters."""
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        
        text = "person entering área restrita 🚫"
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": text}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["text"] == text

    def test_post_custom_anomaly_long_text(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test creating anomaly with long text."""
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        
        # Create a long but reasonable text
        text = "A person is climbing a ladder without proper safety equipment or harness"
        response = test_client.post(
            "/pipeline/custom-anomaly",
            json={"text": text}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["text"] == text


@pytest.mark.integration
class TestCustomAnomalyIntegration:
    """Integration tests for custom anomaly functionality."""

    def test_custom_anomaly_workflow(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test complete workflow of adding multiple custom anomalies."""
        import time
        
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        
        event_texts = [
            "person falling",
            "unattended bag",
            "unauthorized entry",
        ]
        
        # Add each event
        for text in event_texts:
            response = test_client.post(
                "/pipeline/custom-anomaly",
                json={"text": text}
            )
            assert response.status_code == 202
            assert response.json()["text"] == text
        
        # Now manually add them to memory to simulate async completion
        custom_path = custom_memory_temp_dir / "custom_anomaly_memory.json"
        items = [
            {
                "id": str(i),
                "text": text,
                "embedding": [0.1 * i, 0.2 * i],
                "created_at": 1234567890.0 + i,
            }
            for i, text in enumerate(event_texts)
        ]
        _save_custom_memory(custom_path, items)
        
        # Verify all were added (by checking subsequent adds fail as duplicates)
        for text in event_texts:
            response = test_client.post(
                "/pipeline/custom-anomaly",
                json={"text": text}
            )
            assert response.status_code == 409

    def test_memory_persistence(self, custom_memory_file):
        """Test that custom events persist across loads."""
        items1 = [
            {
                "id": "1",
                "text": "test event 1",
                "embedding": [0.1, 0.2],
                "created_at": 1234567890.0,
            }
        ]
        
        _save_custom_memory(custom_memory_file, items1)
        
        # Load it back
        loaded = _load_custom_memory(custom_memory_file)
        assert len(loaded) == 1
        assert loaded[0]["text"] == "test event 1"
        
        # Add another
        items2 = loaded + [
            {
                "id": "2",
                "text": "test event 2",
                "embedding": [0.3, 0.4],
                "created_at": 1234567891.0,
            }
        ]
        _save_custom_memory(custom_memory_file, items2)
        
        # Load again
        loaded2 = _load_custom_memory(custom_memory_file)
        assert len(loaded2) == 2
        assert loaded2[1]["text"] == "test event 2"

    def test_concurrent_duplicate_check(self, test_client, mock_runner, custom_memory_temp_dir, monkeypatch):
        """Test that concurrent requests with same event are handled."""
        import time
        import threading
        
        monkeypatch.setattr(mock_runner.vad, "rtvad_root", str(custom_memory_temp_dir))
        
        results = []
        
        def add_event():
            response = test_client.post(
                "/pipeline/custom-anomaly",
                json={"text": "concurrent test event"}
            )
            results.append(response.status_code)
        
        # Send two concurrent requests with same text
        t1 = threading.Thread(target=add_event)
        t2 = threading.Thread(target=add_event)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Both should return 202 initially (no duplicate at first check)
        # The async thread handles actual deduplication
        assert len(results) == 2
        assert all(code == 202 for code in results)
