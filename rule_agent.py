# agents/rule_agent.py

from core.cognitive_kernel import CognitiveKernel
from agents.env_agent import FocusEnvironment
from core.inference import predict_reason

class RuleAgent:
    def __init__(self):
        self.kernel = CognitiveKernel()
        self.env = FocusEnvironment()

    def run_step(self, feedback_text):
        reason = predict_reason(feedback_text)
        state = self.env.state
        action = self.kernel.decide_action(feedback=reason, distraction=state['distraction_level'])

        next_state, reward, _, _ = self.env.step(action)
        self.kernel.log_event(action, reason, state['distraction_level'])

        print(f"[Agent Step] Feedback: '{feedback_text}' | Reason: {reason} | Action: {action} | Reward: {reward}")
        return action, reward, next_state

    def reflect_memory(self):
        self.kernel.reflect()

if __name__ == "__main__":
    agent = RuleAgent()
    sample_feedback = [
        "I'm too tired today.",
        "I kept checking my phone.",
        "I didn't know what to do."
    ]

    for feedback in sample_feedback:
        agent.run_step(feedback)

    agent.reflect_memory()
