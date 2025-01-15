import sys
sys.path.append("../..")
from Game.BlackJack import BlackJack



class BaseAgent:
    def choose_action(self, game: BlackJack):
        if game.get_playervalue() < 17:
            return "hit"
        else:
            return "stay"