import numpy as np
import pandas as pd

import sys
sys.path.append("../..")
from Game.BlackJack import BlackJack

class NaiveBayesAgent:
    def __init__(self, is_train = False):
        if is_train:
            self.class_priors = {}
            self.feature_likelihoods = {}
            self.classes = None
        else:
            self.class_priors, self.feature_likelihoods, self.classes = self.load_params()

        

    def train(self, filepath = '../../Data/blackjack_simulation_data.csv'):
        df = pd.read_csv(filepath)
        X = df[['player_total_initial','has_ace', 'dealer_card_num']]
        y = df['correct_action']
        
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            self.feature_likelihoods[cls] = (np.sum(X_cls, axis=0) + 1) / (len(X_cls) + 2)  # 拉普拉斯平滑

    def _predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                prior = self.class_priors[cls]
                likelihood = np.prod(self.feature_likelihoods[cls] ** x * (1 - self.feature_likelihoods[cls]) ** (1 - x))
                posteriors[cls] = prior * likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)
    
    def choose_action(self, game: BlackJack):
        has_ace = 1 if game.player_hand[0]['number'] == 'A' or game.player_hand[1]['number'] == 'A' else 0
        dealer_hand_value = game.total_value(game.dealer_hand[:1])

        action = self._predict(np.array([[game.get_playervalue(), has_ace, dealer_hand_value]]))
        return 'hit' if action[0] == 1 else 'stay'
    
    def save_params(self, filepath = 'bayes_'):
        np.save(filepath + 'class_priors.npy', self.class_priors)
        np.save(filepath + 'feature_likelihoods.npy', self.feature_likelihoods)
        np.save(filepath + 'classes.npy', self.classes)
        
    def load_params(self, filepath = 'Agent/bayes/bayes_'):
        class_priors = np.load(filepath + 'class_priors.npy', allow_pickle=True).item()
        feature_likelihoods = np.load(filepath + 'feature_likelihoods.npy', allow_pickle=True).item()
        classes = np.load(filepath + 'classes.npy', allow_pickle=True)
        
        return class_priors, feature_likelihoods, classes
    

if __name__ == '__main__':
    agent = NaiveBayesAgent(is_train=True)
    agent.train()
    agent.save_params()

        
    



