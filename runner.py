# runner.py — Sovereign Orchestrator with PPO Scratch Agent

from agents.env_agent import FocusEnvironment
from core.inference import predict_reason
from agents.distract_detector import detect
from agents.ppo_scratch import PPOAgentScratch
import random
import threading
import time

# Simulated user feedback samples (until live UI or loop is added)
feedback_examples = [
    "I was too tired to do anything.",
    "Got stuck checking YouTube again.",
    "I wasn’t clear what I was trying to finish.",
    "Lost focus halfway through.",
    "Kept jumping tabs."
]

def run_distraction_detection():
    detect()

def run_cognition_loop():
    env = FocusEnvironment()
    agent = PPOAgentScratch()

    # Optional: Load model if you added save/load functionality
    # agent.load_model("models/ppo_scratch.pt")

    state = env.reset()
    print("[SYSTEM] Initial state:", state)

    i = 0
    try:
        while True:
            user_feedback = random.choice(feedback_examples)
            reason = predict_reason(user_feedback)
            state['last_feedback'] = reason

            encoded_state = agent.encode_state(state)
            action, _, _ = agent.select_action(encoded_state)

            next_state, reward, _, _ = env.step(action)

            print(f"\n[LOOP {i+1}] PPO_Action: {action} | Feedback: {reason} | Distraction: {state['distraction_level']} | Reward: {reward}")
            
            state = next_state
            i += 1
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[EXIT] Sovereign PPO Loop terminated by user.")

def main():
    detector_thread = threading.Thread(target=run_distraction_detection)
    detector_thread.daemon = True
    detector_thread.start()

    run_cognition_loop()

if __name__ == "__main__":
    main()
