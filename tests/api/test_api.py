# /home/ubuntu/OpenAgent/tests/api/test_api.py

import pytest
from fastapi.testclient import TestClient
import uuid
import time

# Ensure the app.api can be imported. This might require adjusting PYTHONPATH or project structure if run from root.
# For now, assume it can be imported if tests are run from the project root or with proper setup.
from app.api import app, mock_tasks_db, mock_checkpoint_system # Import the FastAPI app and mock dbs for manipulation

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def reset_mock_dbs_before_each_test():
    """Clears mock databases before each test to ensure isolation."""
    mock_tasks_db.clear()
    mock_checkpoint_system["checkpoints"].clear()
    yield # Test runs here
    # No cleanup needed after as it will be cleared before the next test

# --- Tests for Task Endpoints --- #

def test_submit_task_success(client):
    task_payload = {
        "task_description": "Test task for submission",
        "agent_model_preference": "auto",
        "autonomy_level": "supervised"
    }
    response = client.post("/task", json=task_payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    assert data["message"] == "Task submitted successfully and is pending execution."
    assert data["task_id"] in mock_tasks_db
    assert mock_tasks_db[data["task_id"]]["description"] == task_payload["task_description"]

def test_submit_task_intervention_trigger(client):
    task_payload = {
        "task_description": "This is an intervention test task",
        "agent_model_preference": "deepseek-v3",
        "autonomy_level": "supervised"
    }
    response = client.post("/task", json=task_payload)
    assert response.status_code == 200
    data = response.json()
    task_id = data["task_id"]
    assert task_id in mock_tasks_db
    assert mock_tasks_db[task_id]["status"] == "requires_intervention"
    assert mock_tasks_db[task_id]["requires_intervention"] is True
    assert mock_tasks_db[task_id]["checkpoint_id"] is not None
    assert mock_tasks_db[task_id]["checkpoint_id"] in mock_checkpoint_system["checkpoints"]

def test_get_task_status_found(client):
    # First, submit a task to have something to query
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    mock_tasks_db[task_id] = {
        "id": task_id, "description": "Status test task", "status": "pending",
        "current_step": 0, "total_steps": 3, "updated_at": time.time(),
        "requires_intervention": False, "checkpoint_id": None
    }

    response = client.get(f"/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "pending" # Initial status
    # The mock API progresses the task on GET, so it might be "running" now
    # Let's check the DB directly for the *next* expected state after one GET
    assert mock_tasks_db[task_id]["status"] == "running"
    assert mock_tasks_db[task_id]["current_step"] == 1

def test_get_task_status_not_found(client):
    non_existent_task_id = "task_does_not_exist"
    response = client.get(f"/status/{non_existent_task_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}

def test_get_task_status_with_intervention_data(client):
    task_id = f"task_intervention_status_{uuid.uuid4().hex[:8]}"
    checkpoint_id = f"ckpt_intervention_status_{uuid.uuid4().hex[:4]}"
    mock_tasks_db[task_id] = {
        "id": task_id, "description": "Intervention status test", "status": "requires_intervention",
        "current_step": 1, "total_steps": 3, "updated_at": time.time(),
        "requires_intervention": True, "checkpoint_id": checkpoint_id
    }
    mock_checkpoint_system["checkpoints"][checkpoint_id] = {
        "task_id": task_id, "step_id": "step_01", "reason": "Test intervention reason",
        "current_plan_or_state": {"action": "pause"}, "status": "pending_review"
    }

    response = client.get(f"/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "requires_intervention"
    assert data["requires_intervention"] is True
    assert data["checkpoint_data"] is not None
    assert data["checkpoint_data"]["checkpoint_id"] == checkpoint_id
    assert data["checkpoint_data"]["reason"] == "Test intervention reason"

# --- Tests for Intervention Endpoints --- #

def test_handle_intervention_success_proceed(client):
    task_id = f"task_intervene_proceed_{uuid.uuid4().hex[:8]}"
    checkpoint_id = f"ckpt_intervene_proceed_{uuid.uuid4().hex[:4]}"
    mock_tasks_db[task_id] = {
        "id": task_id, "description": "Intervention proceed test", "status": "requires_intervention",
        "current_step": 1, "total_steps": 3, "updated_at": time.time(),
        "requires_intervention": True, "checkpoint_id": checkpoint_id
    }
    mock_checkpoint_system["checkpoints"][checkpoint_id] = {
        "task_id": task_id, "step_id": "step_01", "reason": "Test intervention",
        "current_plan_or_state": {}, "status": "pending_review"
    }

    intervention_payload = {"feedback": "All good, proceed.", "action": "proceed"}
    response = client.post(f"/intervention/{checkpoint_id}", json=intervention_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["checkpoint_id"] == checkpoint_id
    assert data["status"] == "resolved"
    assert mock_checkpoint_system["checkpoints"][checkpoint_id]["status"] == "resolved"
    assert mock_tasks_db[task_id]["status"] == "running" # Task resumes
    assert mock_tasks_db[task_id]["requires_intervention"] is False

def test_handle_intervention_success_abort(client):
    task_id = f"task_intervene_abort_{uuid.uuid4().hex[:8]}"
    checkpoint_id = f"ckpt_intervene_abort_{uuid.uuid4().hex[:4]}"
    mock_tasks_db[task_id] = {"id": task_id, "status": "requires_intervention", "checkpoint_id": checkpoint_id, "requires_intervention": True}
    mock_checkpoint_system["checkpoints"][checkpoint_id] = {"task_id": task_id, "status": "pending_review"}

    intervention_payload = {"feedback": "Stop this task.", "action": "abort"}
    response = client.post(f"/intervention/{checkpoint_id}", json=intervention_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "aborted"
    assert mock_checkpoint_system["checkpoints"][checkpoint_id]["status"] == "aborted"
    assert mock_tasks_db[task_id]["status"] == "aborted"

def test_handle_intervention_checkpoint_not_found(client):
    intervention_payload = {"feedback": "Doesn't matter.", "action": "proceed"}
    response = client.post("/intervention/ckpt_does_not_exist", json=intervention_payload)
    assert response.status_code == 404
    assert response.json() == {"detail": "Checkpoint not found"}

def test_handle_intervention_not_pending_review(client):
    checkpoint_id = f"ckpt_intervene_not_pending_{uuid.uuid4().hex[:4]}"
    mock_checkpoint_system["checkpoints"][checkpoint_id] = {"task_id": "some_task", "status": "resolved"} # Already resolved

    intervention_payload = {"feedback": "Too late.", "action": "proceed"}
    response = client.post(f"/intervention/{checkpoint_id}", json=intervention_payload)
    assert response.status_code == 200 # API handles this gracefully
    data = response.json()
    assert data["message"] == "Intervention not applicable or already processed."
    assert data["status"] == "resolved"

# --- Tests for Metrics Endpoint --- #

def test_get_system_metrics(client):
    # Mock the metrics data that the API's mock_metrics_tracker would return
    expected_metrics = {"model_A": {"calls": 10, "avg_latency_ms": 200}}
    # If the API's mock_metrics_tracker.get_summary can be patched or set:
    # For this test, we assume the mock_metrics_tracker in api.py is static for the test run.
    
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "timestamp" in data
    assert data["metrics"] == expected_metrics # Relies on the static mock in api.py

# To run: pytest tests/api/test_api.py

