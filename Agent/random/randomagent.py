import random

class RandomAgent:
    def __init__(self):
        pass
    
    def choose_action(self):
        return random.choice(["hit", "stay"])
        