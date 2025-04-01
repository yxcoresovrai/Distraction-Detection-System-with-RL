# core/logger.py

import csv
import os
from datetime import datetime

DATA_DIR = "data"

def log_distraction(event_type: str, source: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "distractions.csv")
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), event_type, source])


def log_goal(goal_text: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "goals.csv")
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), goal_text])


def log_session(goal_text: str, start_time: str, completed: bool, reason: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "session_summary.csv")
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([goal_text, start_time, datetime.now().isoformat(), completed, reason])


def log_all(goal_text: str, start_time: str, completed: bool, reason: str):
    log_session(goal_text, start_time, completed, reason)
    if not completed:
        log_goal(goal_text)