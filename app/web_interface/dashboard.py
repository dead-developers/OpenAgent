# app/web_interface/dashboard.py

import streamlit as st
import requests # For API calls to the agent system
import json
import time
import os

# --- Configuration ---
# In a real application, these would come from a config file or environment variables
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000") # Assuming FastAPI runs on port 8000
REFRESH_INTERVAL = 5 # Seconds

# --- Helper Functions ---
def submit_task_to_agent(task_description: str, agent_model: str, autonomy_level: str) -> Optional[str]:
    """Submits a task to the agent API and returns the task ID."""
    try:
        payload = {
            "task_description": task_description,
            "agent_model_preference": agent_model, # e.g., "deepseek-v3", "deepseek-r1", "auto"
            "autonomy_level": autonomy_level # e.g., "full", "supervised", "manual"
        }
        response = requests.post(f"{AGENT_API_URL}/task", json=payload, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        task_id = response.json().get("task_id")
        st.session_state.last_api_error = None
        return task_id
    except requests.exceptions.RequestException as e:
        st.error(f"API Error submitting task: {e}")
        st.session_state.last_api_error = str(e)
        return None

def get_task_status_from_agent(task_id: str) -> Optional[dict]:
    """Retrieves the status of a task from the agent API."""
    if not task_id:
        return None
    try:
        response = requests.get(f"{AGENT_API_URL}/status/{task_id}", timeout=5)
        response.raise_for_status()
        st.session_state.last_api_error = None
        return response.json()
    except requests.exceptions.RequestException as e:
        # Only show error if it's a new one or task is active
        if st.session_state.get("active_task_id") == task_id or st.session_state.last_api_error != str(e):
            st.error(f"API Error getting status for {task_id}: {e}")
        st.session_state.last_api_error = str(e)
        return None

def handle_intervention_from_agent(checkpoint_id: str, user_feedback: str, action: str) -> bool:
    """Sends intervention feedback to the agent API."""
    try:
        payload = {"feedback": user_feedback, "action": action} # action: "proceed", "modify", "abort"
        response = requests.post(f"{AGENT_API_URL}/intervention/{checkpoint_id}", json=payload, timeout=10)
        response.raise_for_status()
        st.session_state.last_api_error = None
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"API Error handling intervention for {checkpoint_id}: {e}")
        st.session_state.last_api_error = str(e)
        return False

def get_metrics_from_agent() -> Optional[dict]:
    """Retrieves performance metrics from the agent API."""
    try:
        response = requests.get(f"{AGENT_API_URL}/metrics", timeout=5)
        response.raise_for_status()
        st.session_state.last_api_error = None
        return response.json()
    except requests.exceptions.RequestException as e:
        if st.session_state.last_api_error != str(e):
             st.error(f"API Error getting metrics: {e}")
        st.session_state.last_api_error = str(e)
        return None

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="OpenAgent Dashboard")

# Initialize session state variables
if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None
if "task_status_history" not in st.session_state:
    st.session_state.task_status_history = {}
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "last_api_error" not in st.session_state:
    st.session_state.last_api_error = None
if "current_checkpoint" not in st.session_state:
    st.session_state.current_checkpoint = None # Store checkpoint data if intervention is needed

# --- Sidebar: Control Panel ---
st.sidebar.title("OpenAgent Control Panel")

task_description = st.sidebar.text_area("Enter Task Description:", height=150, key="task_desc_input")
agent_model_options = ["auto", "deepseek-v3", "deepseek-r1"] # As per spec
agent_model_preference = st.sidebar.selectbox("Preferred Agent Model:", agent_model_options, key="agent_model_select")
autonomy_level_options = ["full", "supervised", "manual"] # As per spec
autonomy_level = st.sidebar.selectbox("Autonomy Level:", autonomy_level_options, key="autonomy_select")

if st.sidebar.button("Submit Task", key="submit_task_btn"):
    if task_description:
        task_id = submit_task_to_agent(task_description, agent_model_preference, autonomy_level)
        if task_id:
            st.session_state.active_task_id = task_id
            st.session_state.task_status_history[task_id] = [] # Initialize history for this task
            st.session_state.current_checkpoint = None # Clear any old checkpoint
            st.sidebar.success(f"Task submitted! ID: {task_id}")
            # Clear the input field after submission
            # This requires a bit of a workaround in Streamlit if not using forms
    else:
        st.sidebar.warning("Please enter a task description.")

st.sidebar.markdown("---_)
st.sidebar.subheader("Active Task")
if st.session_state.active_task_id:
    st.sidebar.write(f"Current Task ID: {st.session_state.active_task_id}")
    if st.sidebar.button("Clear Active Task", key="clear_task_btn"):
        st.session_state.active_task_id = None
        st.session_state.current_checkpoint = None
else:
    st.sidebar.write("No active task.")

# --- Main Area: Task Status & Metrics ---
st.title("Agent Status Dashboard")

status_col, metrics_col = st.columns(2)

# --- Task Status Section ---
with status_col:
    st.header("Task Progress")
    status_placeholder = st.empty()

    if st.session_state.active_task_id and st.session_state.current_checkpoint:
        st.subheader("Intervention Required!")
        checkpoint_data = st.session_state.current_checkpoint
        st.write(f"**Checkpoint ID:** {checkpoint_data.get(	checkpoint_id	)}")
        st.write(f"**Reason:** {checkpoint_data.get(	reason	)}")
        st.write("**Current State/Plan:**")
        st.json(checkpoint_data.get(	current_plan_or_state	, {}))
        
        feedback_text = st.text_area("Your Feedback/Instructions:", key="intervention_feedback")
        intervention_action = st.selectbox("Action:", ["proceed", "modify_plan", "abort"], key="intervention_action")

        if st.button("Submit Intervention", key="submit_intervention_btn"):
            if feedback_text or intervention_action == "abort" or intervention_action == "proceed": # Feedback might not be needed for abort/proceed
                success = handle_intervention_from_agent(checkpoint_data["checkpoint_id"], feedback_text, intervention_action)
                if success:
                    st.success("Intervention submitted.")
                    st.session_state.current_checkpoint = None # Clear checkpoint after handling
                    # Force a refresh of status
            else:
                st.warning("Please provide feedback or select an action.")

# --- Metrics Section ---
with metrics_col:
    st.header("Performance Metrics")
    metrics_placeholder = st.empty()

# --- Auto-Updating Loop ---
if __name__ == "__main__": # Ensure this runs only when script is executed directly
    while True:
        current_task_id = st.session_state.get("active_task_id")
        task_display_content = "No active task or not submitted yet."

        if current_task_id and not st.session_state.current_checkpoint:
            status_data = get_task_status_from_agent(current_task_id)
            if status_data:
                st.session_state.task_status_history[current_task_id].append(status_data)
                # Display latest status
                task_display_content = f"**Task ID:** {current_task_id}\n"
                task_display_content += f"**Status:** {status_data.get(	status	, 	N/A	)}\n"
                task_display_content += f"**Details:** {status_data.get(	details	, 	N/A	)}\n"
                task_display_content += f"**Last Update:** {time.strftime(	%Y-%m-%d %H:%M:%S	, time.localtime(status_data.get(	timestamp	, time.time())))}\n"
                
                if "current_step" in status_data and "total_steps" in status_data:
                    progress = (status_data["current_step"] / status_data["total_steps"]) * 100 if status_data["total_steps"] > 0 else 0
                    task_display_content += f"**Progress:** {status_data[	current_step	]}/{status_data[	total_steps	]} ({progress:.2f}%)\n"

                if status_data.get(	requires_intervention	) and status_data.get(	checkpoint_data	):
                    st.session_state.current_checkpoint = status_data["checkpoint_data"]
                    # Rerun to display intervention UI immediately
                    st.experimental_rerun()
                
                if status_data.get(	status	) in ["completed", "failed", "aborted"]:
                    st.session_state.active_task_id = None # Clear active task if finished
            else:
                task_display_content = f"**Task ID:** {current_task_id}\nStatus: Error fetching status or task not found."
        elif st.session_state.current_checkpoint:
            task_display_content = "Awaiting user intervention (see details above)."
        
        with status_placeholder.container():
            st.markdown(task_display_content)
            if current_task_id and st.session_state.task_status_history.get(current_task_id):
                st.subheader("Status History")
                # Display last 5 history items for brevity
                history_to_show = st.session_state.task_status_history[current_task_id][-5:]
                for i, entry in enumerate(reversed(history_to_show)):
                    with st.expander(f"Update {len(history_to_show) - i} ({time.strftime(	%H:%M:%S	, time.localtime(entry.get(	timestamp	, time.time())))})"):
                        st.json(entry)
        
        # Update Metrics
        metrics_data = get_metrics_from_agent()
        if metrics_data:
            st.session_state.metrics_history.append(metrics_data)
            # Keep last 100 metrics for display, for example
            if len(st.session_state.metrics_history) > 100:
                st.session_state.metrics_history.pop(0)
            
            with metrics_placeholder.container():
                st.json(metrics_data) # Simple JSON display for now
                # Could add charts here based on metrics_history
        else:
            with metrics_placeholder.container():
                st.write("Could not retrieve metrics or no metrics available.")

        time.sleep(REFRESH_INTERVAL)
        # Only rerun if there's an active task or checkpoint to avoid constant reruns when idle
        if st.session_state.active_task_id or st.session_state.current_checkpoint:
             st.experimental_rerun() # Rerun to update the UI elements
        elif not st.session_state.active_task_id and not st.session_state.current_checkpoint and st.session_state.last_api_error:
            # If idle but there was an API error, allow one more rerun to clear it if connection restores
            pass # Let the loop run once more to see if API error clears

