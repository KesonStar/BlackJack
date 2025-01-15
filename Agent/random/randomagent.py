import random


import sys
sys.path.append("../..")
from Game.BlackJack import BlackJack


class RandomAgent:
    def __init__(self):
        pass
    
    def choose_action(self, game: BlackJack):
        return random.choice(["hit", "stay"])
        