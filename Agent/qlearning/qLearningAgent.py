import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../..")
from Game.BlackJack import BlackJack


class qLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, is_train = False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        if is_train:
            self.Q = np.zeros((33, 12, 2, 2)) # Player sum, dealer card, action
        else:
            self.load_q_table("Agent/qlearning/Q_traditional.npy")
            
    def choose_action(self, game, is_train=False):
        player_sum = game.get_playervalue()
        dealer_card = game.total_value(game.dealer_hand[:1])
        usable_ace = self.has_usable_ace(game.player_hand)
        # print(player_sum, dealer_card, usable_ace)
        if is_train:
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(["hit", "stay"])
            else:
                return "hit" if self.Q[player_sum, dealer_card, usable_ace, 0] > self.Q[player_sum, dealer_card, usable_ace, 1] else "stay"

        else:
            return "hit" if self.Q[player_sum, dealer_card, usable_ace, 0] > self.Q[player_sum, dealer_card, usable_ace, 1] else "stay"
            

    def update(
        self,
        player_sum,
        dealer_card,
        usable_ace,
        action,
        reward,
        new_player_sum,
        new_dealer_card,
        new_usable_ace,
    ):
        action_idx = 0 if action == "hit" else 1
        old_value = self.Q[player_sum, dealer_card, usable_ace, action_idx]
        future_max = np.max(self.Q[new_player_sum, new_dealer_card, new_usable_ace])
        self.Q[player_sum, dealer_card, usable_ace, action_idx] = (
            old_value + self.alpha * (reward + self.gamma * future_max - old_value)
        )

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

    def train(self, episodes):

        for _ in tqdm(range(episodes)):
            game = BlackJack("novel")
            game.start()

            # if ep % one_percent == 0:
            #     progress = (ep / episodes) * 100
            #     print(f"Training progress: {progress:.2f}%")

            dealer_card = game.total_value(game.dealer_hand[:1])
            status = "continue"

            while status == "continue":
                player_sum = game.get_playervalue()
                usable_ace = self.has_usable_ace(game.player_hand)
                action = self.choose_action(game, is_train=True)
                game.player_action(action)
                
                status = game.get_status()
                new_player_sum = game.get_playervalue()
                new_usable_ace = self.has_usable_ace(game.player_hand)

                reward = 0  # Intermediate reward, only final matters
                assert status in ["continue", "player_blackjack", "player_bust", "stay"]
                
                if status == "player_blackjack":
                    reward = 1
                    # print("Player got a blackjack")
                elif status == "player_bust":
                    reward = -1
                    # print("Player bust")

                if reward != 0:
                    self.update(
                        player_sum,
                        dealer_card,
                        usable_ace,
                        action,
                        reward,
                        new_player_sum,
                        dealer_card,
                        new_usable_ace)

                if action == "stay":
                    break

            game.dealer_action("basic")
            final_result = game.game_result()
            final_reward =  1 if final_result == "win" else (-1 if final_result == "lose" else 0)
            self.update(
                player_sum,
                dealer_card,
                usable_ace,
                action,
                final_reward,
                new_player_sum,
                dealer_card,
                new_usable_ace,
            )


    def save_q_table(self, filename = "Q_traditional.npy"):
        np.save(filename, self.Q)

    def load_q_table(self, filename):
        self.Q = np.load(filename)
        
    def play(self):
        game = BlackJack("novel")
        game.start()

        status = "continue"
        while status == "continue":
            action = self.choose_action(game)
            status = game.player_action(action)
            
            if action == "stay":
                break
                
        if status == "continue" or "stay":
            game.dealer_action("basic")
            # print("Dealer has:", game.format_cards(game.dealer_hand), game.get_dealervalue())
        # print("Player has:", game.format_cards(game.player_hand), game.get_playervalue())
        final_result = game.game_result()
        
        # print("*"*20,"\n\n")
        return final_result
    
    
if __name__ == "__main__":
    rounds = 100000
    agent = qLearningAgent(is_train=True)
    agent.train(1000000)
    agent.save_q_table("Q_traditional_2.npy")
