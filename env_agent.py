# agents/env_agent.py

import random

class FocusEnvironment:
    def __init__(self):
        self.states = {
            'time_of_day': ["morning", "afternoon", "evening"],
            'distraction_level': ["low", "medium", "high"],
            'last_feedback': ["none", "tired", "distracted", "unclear"],
            'last_action': ["none", "popup", "break", "pomodoro", "rewrite_goal"]
        }
        self.reset()

    def reset(self):
        self.state = {
            'time_of_day': random.choice(self.states['time_of_day']),
            'distraction_level': 'medium',
            'last_feedback': "none",
            'last_action': "none"
        }
        self.done = False
        return self.state

    def step(self, action):
        reward = 0

        if action == "break" and self.state['last_feedback'] == "tired":
            reward = 1
        elif action == "popup" and self.state['distraction_level'] == "high":
            reward = 1
        elif action == "rewrite_goal" and self.state['last_feedback'] == "unclear":
            reward = 1
        elif action == "pomodoro" and self.state['distraction_level'] == "medium":
            reward = 1
        else:
            reward = -1

        self.state['time_of_day'] = random.choice(self.states['time_of_day'])
        self.state['distraction_level'] = random.choice(self.states['distraction_level'])
        self.state['last_feedback'] = random.choice(self.states['last_feedback'])
        self.state['last_action'] = action

        return self.state, reward, self.done, {}

if __name__ == "__main__":
    env = FocusEnvironment()
    state = env.reset()
    print("Initial state:", state)
    for _ in range(5):
        action = random.choice(env.states['last_action'][1:])
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action} -> Reward: {reward}, Next State: {next_state}")
