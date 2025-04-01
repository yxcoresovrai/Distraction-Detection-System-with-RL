# env/feedback_ingest.py

import streamlit as st
from core.logger import log_all
from datetime import datetime

st.title("📥 Feedback Ingestor")

if "goal" not in st.session_state:
    st.session_state.goal = ""

if st.session_state.goal == "":
    goal = st.text_input("Set your goal for this session:")
    if goal:
        st.session_state.goal = goal
        st.session_state.start_time = datetime.now().isoformat()
        st.success(f"Session started for goal: {goal}")
else:
    st.write(f"Current Goal: {st.session_state.goal}")
    completed = st.radio("Did you complete the goal?", ["Yes", "No"])
    reason = ""
    if completed == "No":
        reason = st.text_input("Why not?")

    if st.button("End Session"):
        log_all(
            goal_text=st.session_state.goal,
            start_time=st.session_state.start_time,
            completed=(completed == "Yes"),
            reason=reason
        )
        st.success("Session ended and logged.")
        st.session_state.goal = ""
