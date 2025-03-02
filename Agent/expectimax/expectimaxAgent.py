
import sys
sys.path.append("../..")
from Game.BlackJack import BlackJack


class ExpectimaxAgent:
    """
    Player that implements an expectimax policy for choosing actions
    """
    def __init__(self):
        # Init player parent
        self.avgVal = float(4*(1+2+3+4+5+6+7+8+9+10+10+10+10))/float(52)

    def choose_action(self, game: BlackJack):
        """
        GetAction for UserPlayer prints valid actions and asks for input for the action to take
        input: gameState of current game
        returns: action to take
        """

        """
        Get the expected dealer hand value using the dealers current hand and
        the average value of a deck
        """
        averageValue = self.avgVal
        dVal = game.get_dealervalue()
        while dVal < 17:
            dVal += averageValue
        handVal = game.get_playervalue()

        # Overall function that tests for an end state and returns a score
        def overall(self, state, agent, pStay, dealerVal, bet):
            if pStay or state > 21:
                pBust = state > 21
                if pBust and dealerVal <= 21:
                    return [-bet]
                elif dealerVal > 21 and not pBust:
                    return [bet]
                elif dealerVal > state:
                    return [-bet]
                elif state > dealerVal:
                    return [bet]
                else:
                    return [0]

            if agent == 0:
                return maxVal(self, state, agent, pStay, dealerVal, bet)
            else:
                return expVal(self, state, agent, pStay, dealerVal, bet)

        # if its the agents turn, then take the maximum of the possible actions
        def maxVal(self, state, agent, pStay, dealerVal, bet):

            # initialize a highschore (in a list so we can add optimal action)
            highScore = [float('-inf')]

            # get possible actions for first turn
            actions = ['hit', 'stay']

            # keep the max action score of the actions
            for action in actions:
                nS = generateSuccessor(state, agent, pStay, bet, action)
                score = overall(self, nS[0], 1, nS[1], dealerVal, nS[2])
                if score[0] >= highScore[0]:
                    highScore = [score[0], action]
            return highScore

        # for all subsequent turns, take the expected val of possible actions
        def expVal(self, state, agent, pStay, dealerVal, bet):

            highScore = [0]
            actions = ['hit', 'stay']

            # accumulate expected val of actions
            for action in actions:
                nS = generateSuccessor(dealerVal, agent, pStay, bet, action)
                score = overall(self, state, 0, pStay, nS[0], nS[2])
                highScore[0] += float(score[0])/float(len(actions))
                # print(state.getPlayerHands() == og, '****')
            return highScore

        # depending on the action, update the card val, Stay status, and bet
        def generateSuccessor(state, agent, pStay, bet, action):
            
            averageValue = self.avgVal 
            if agent == 0:
                if action == 'hit':
                    newState = state + averageValue
                    newStay = pStay
                    newBet = bet
                if action == 'stay':
                    newStay = True
                    newState = state
                    newBet = bet
                return [newState, newStay, newBet]
            else:
                if action == 'hit':
                    return [state + averageValue, pStay, bet]
                if action == 'stay':
                    return [state, pStay, bet]
                

        return overall(self, handVal, 0, False, dVal, 10)[1] ## [1] here is the action to take defined in maxVal