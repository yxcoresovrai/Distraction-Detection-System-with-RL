# QuantumMindAI – Sovereign Cognitive System 

## Overview
**QuantumMindAI** is a local-first, offline-only framework for building human-aligned, distraction-aware AI agents capable of self-reflection, short-term memory, symbolic decision-making, and real-time feedback adaptation.

This isn't just a SaaS application – it's a step toward better reasoning in local-first cognitive systems. Everything runs locally, with no external API dependencies, ensuring privacy, resilience, and total control.

---

## Features

1. **Cognitive Kernel** (`/core/cognitive_kernel.py`)
   - A symbolic reasoning module with:
     - **Short-term memory** (deque)
     - **Goal stack** for hierarchical tasks
     - **Rulebase** for direct action decisions based on context
     - **Reflection** method for introspective logging

2. **Reason Classifier** (`inference.py`)
   - A locally fine-tuned **DistilBert** model that classifies user feedback into reasons like `"tired"`, `"unclear"`, or `"distracted"`.
   - All model weights and label encoders stored offline in `./failure_classifier`.

3. **Distraction Detection** (`/agents/distract_detector.py`)
   - Computer Vision pipeline (YOLOv8 + Mediapipe) to flag phone usage, no face detected, or high distraction.
   - Logs events locally to `distractions.csv` whenever it sees a flagged item.

4. **RL Environment** (`/agents/env_agent.py`)
   - Simulated environment (FocusEnvironment) that mimics user states:
     - `time_of_day` (morning, afternoon, evening)
     - `distraction_level` (low, medium, high)
     - `last_feedback` (`tired`, `distracted`, etc.)
   - Provides reward signals to the agent for chosen actions.

5. **Rule-Based Agent** (`/agents/rule_agent.py`)
   - Example agent that picks actions (`popup`, `break`, `pomodoro`, `rewrite_goal`) based on the kernel’s rules and environment states.

6. **Logging + Memory** (`/core/logger.py`)
   - All events, sessions, and goals are appended to `.csv` logs within `/data/`.
   - These logs feed back into short-term memory and reflection cycles.

7. **Runner** (`runner.py`)
   - The main orchestration script:
     - Initializes environment + cognitive kernel.
     - Grabs or simulates user feedback.
     - Uses `inference.py` to classify feedback.
     - Feeds results into kernel for final action.
     - Logs each loop.

8. **Local-Only**
   - No cloud or API calls.
   - Models + data stored in local `./models` or `./failure_classifier`.
   - Strong emphasis on offline sovereignty.

---

## Why local-first cognitive models?
- **Privacy:** Freed from corporate or governmental data scrapes
- **Fine Tuning:** Can Fine Tune to match user's goals

---

## File Structure

```plaintext
QuantumMindAI/
├── core/
│   ├── cognitive_kernel.py         # Short-term memory, goal stack, reflection
│   ├── logger.py                   # CSV logging for events, goals, sessions
│   └── reason_classifier.py        # (Example: inference.py merged)
│
├── agents/
│   ├── rule_agent.py               # Symbolic decision agent
│   ├── env_agent.py                # FocusEnvironment (toy RL)
│   ├── distract_detector.py        # YOLOv8 + Mediapipe real-time
│   └── ppo_agent.py                # (Future stub for RL)
│
├── env/
│   ├── webcam_loop.py              # Live camera feed
│   ├── feedback_ingest.py          # (Optional user input pipeline)
│
├── data/
│   ├── distractions.csv
│   ├── goals.csv
│   ├── session_summary.csv
│   └── failure_reasons.csv         # Train DistilBert
│
├── models/
│   └── failure_classifier/         # DistilBert model + tokenizer
├── runner.py                       # Orchestrator
├── redirector.py                   # Popup & interventions
└── README.md
```

---

## Setup & Run
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have [PyTorch](https://pytorch.org/), [transformers](https://github.com/huggingface/transformers), [mediapipe](https://google.github.io/mediapipe/), [ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8, etc.

2. **Train the Reason Classifier** (optional, if you want to retrain):
   ```bash
   python distil_bert.py
   ```
   This outputs a local model in `./failure_classifier`.

3. **Run the Agent**
   ```bash
   python runner.py
   ```
   - Simulates environment steps
   - Classifies random user feedback
   - Logs short-term memory events
   - Takes symbolic actions

4. **Distraction Detector**
   ```bash
   python agents/distract_detector.py
   ```
   - Opens webcam
   - Detects phone usage or no face
   - Logs to `distractions.csv`
   - Pops up a motivational overlay


## Contact
- **Author**: Team Soverign Cognitive Interface
- **License**: TBD (Open-Source or Private). For unstoppable human alignment.

