import random
import numpy as np




import sys
sys.path.append("../..")
from Game.BlackJack import BlackJack
from tqdm import trange



class MDPAgent:
    def __init__(self, gamma=1.0, eps=0.0001, is_train = False):
        # self.game = BlackJack(mode=game_mode)
        
        self.gamma = gamma 
        self.eps = eps # Threshold for convergence
        self.value_table = np.zeros((32, 12, 2)) # Player sum, dealer card, has_usable_ace
        if is_train:
            self.policy = np.zeros((32, 12, 2), dtype=int) # Policy: 0 (stay) or 1 (hit)
        else:
            self.policy = self.load_policy("Agent/MDP/policy.npy")
            
        self.action_space = [0, 1] # 0 (stay) or 1 (hit)
    @staticmethod
    def has_usable_ace(hand):
        """Check if the hand has a usable ace."""
        value, ace = 0, False
        for card in hand:
            card_number = card["number"]
            value += min(
                10, int(card_number) if card_number not in ["J", "Q", "K", "A"] else 11
            )
            ace |= card_number == "A"
        return int(ace and value + 10 <= 21)
    
    def state(self, game):
        player_sum = game.get_playervalue()
        dealer_card_num = game.total_value(game.dealer_hand[:1])
        has_usable_ace = self.has_usable_ace(game.player_hand)
        return (player_sum, dealer_card_num, has_usable_ace) 

    def calculate_state_value(self, player_sum, dealer_showing, usable_ace, action):
        if action == 0:  # stick
            return self._evaluate_stick(player_sum, dealer_showing)  # Compare player sum and dealer showing card
        else:  # hit
            return 0 if player_sum <= 21 else -1  # Reward for hitting

    def evaluate_actions(self, player_sum, dealer_showing, usable_ace):
        actions_values = np.zeros(2) # 0 (stay) or 1 (hit)
        for action in range(len(self.action_space)):
            actions_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action)
        return np.max(actions_values)
    
    def extract_policy(self):
        for player_sum in range(1, 32):
            for dealer_showing in range(1, 12):
                for usable_ace in range(2):
                    action_values = np.zeros(len(self.action_space))
                    for action in range(len(self.action_space)):
                        action_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action)
                    self.policy[player_sum, dealer_showing, usable_ace] = np.argmax(action_values)
    
    
    def value_iteration(self):
        iterations = 0
        while True:
            delta = 0
            for player_sum in range(1, 32):
                for dealer_showing in range(1, 11):
                    for usable_ace in range(2):
                        v_old = self.value_table[player_sum, dealer_showing, usable_ace]
                        v_new = self.evaluate_actions(player_sum, dealer_showing, usable_ace)
                        self.value_table[player_sum, dealer_showing, usable_ace] = v_new
                        delta = max(delta, abs(v_old - v_new))
                        # print(delta)
            if delta < self.eps:
                break
            else:
                # print(f'{iterations}: {delta}')
                iterations += 1
        self.extract_policy()
        
    def _evaluate_stick(self, player_sum, dealer_showing):
        dealer_probs = self._calculate_dealer_probabilities(dealer_showing)
        expected_reward = 0

        for dealer_sum, prob in dealer_probs.items():
            if dealer_sum == 'bust':
                expected_reward += prob
            elif dealer_sum != 'bust' and int(dealer_sum) > player_sum:
                expected_reward -= prob
            elif dealer_sum != 'bust' and int(dealer_sum) < player_sum:
                expected_reward += prob

        return expected_reward

    def _calculate_dealer_probabilities(self, dealer_showing):
        card_probabilities = {1: 1/13, 2: 1/13, 3: 1/13, 4: 1/13, 5: 1/13, 6: 1/13, 7: 1/13, 8: 1/13, 9: 1/13, 10: 4/13}
        dealer_probabilities = {}

        for card_value, prob in card_probabilities.items():
            dealer_sum = dealer_showing + card_value
            if dealer_sum < 21:
                if dealer_sum in dealer_probabilities:
                    dealer_probabilities[dealer_sum] += prob
                else:
                    dealer_probabilities[dealer_sum] = prob

        return dealer_probabilities

    def save_policy(self, filename = "policy.npy"):
        np.save(filename, self.policy)
        
    def load_policy(self, filename = "policy.npy"):
        policy = np.load(filename)
        return policy
    
    
    def choose_action(self, game):
        if self.policy[self.state(game)] == 0:
            return 'stay'
        else:
            return 'hit'
    
    
if __name__ == "__main__":
    agent = MDPAgent(is_train=True)
    agent.value_iteration()
    agent.save_policy()
    
    

