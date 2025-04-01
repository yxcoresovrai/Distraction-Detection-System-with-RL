# core/cognitive_kernel.py

from collections import deque
import datetime

class CognitiveKernel:
    def __init__(self):
        self.short_term_memory = deque(maxlen=15)  # recent events
        self.goal_stack = []
        self.rulebase = self._init_rulebase()

    def _init_rulebase(self):
        return {
            ("tired",): "break",
            ("unclear",): "rewrite_goal",
            ("distracted",): "popup",
            ("low",): "pomodoro",
            ("medium",): "pomodoro",
            ("high",): "popup"
        }

    def log_event(self, action, feedback, distraction):
        self.short_term_memory.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "feedback": feedback,
            "distraction": distraction
        })

    def push_goal(self, goal):
        if goal not in self.goal_stack:
            self.goal_stack.append(goal)

    def pop_goal(self):
        return self.goal_stack.pop() if self.goal_stack else None

    def reflect(self):
        print("\n[🧠 REFLECTION] Recent Memory Log:")
        for m in self.short_term_memory:
            print(f"{m['timestamp']} | Action: {m['action']} | Feedback: {m['feedback']} | Distraction: {m['distraction']}")

    def decide_action(self, feedback=None, distraction="medium"):
        key = (feedback,) if feedback else (distraction,)
        return self.rulebase.get(key, "pomodoro")
