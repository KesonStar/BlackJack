import random

class BlackJack:
    heart = "\u2665"
    spade = "\u2660"
    diamond = "\u2666"
    club = "\u2663"

    suits = {
        "diamonds": diamond,
        "hearts": heart,
        "spades": spade,
        "clubs": club
    }
    
    
    def __init__(self, mode = "traditional"):
        self.deck = self.generate_deck()
        random.shuffle(self.deck)
        self.player_hand = []
        self.dealer_hand = []
        if mode not in ["traditional", "novel"]:
            raise ValueError("Invalid game mode")   
        
        self.card_count = {
                      '2': 0,
                      '3': 0,
                      '4': 0,
                      '5': 0,
                      '6': 0,
                      '7': 0,
                      '8': 0,
                      '9': 0,
                      '10': 0,
                      'A': 0}
        self.mode = mode
        self.status = "continue"

    @staticmethod
    def format_cards(cards):
        """Format a list of cards into a readable string."""
        
        result = ""
        for card in cards:
            suit = BlackJack.suits[card["suit"]]
            result += f"{card['number']}{suit} "
        
        return result.strip()
    
    def generate_deck(self):
        """Generate a full deck of cards with all suits and numbers."""
        
        numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        deck = [{'number': number, 'suit': suit} for number in numbers for suit in suits]
        return deck
    
    def total_value(self, hand: list):
        """Compute the total value of a hand, taking aces into account."""
        
        value = 0
        aces = 0
        for card in hand:
            if card['number'] in ['J', 'Q', 'K']:
                value += 10
            elif card['number'] == 'A':
                value += 11
                aces += 1
            else:
                value += int(card['number'])
                
        while value > 21 and aces != 0:
            value -= 10
            aces -= 1
        return value
    
    def draw_card(self):
        """Draw a card from the deck and update the card count."""
        
        card = self.deck.pop()
        if card["number"] in ['J', 'Q', 'K']: 
            self.card_count["10"] += 1
        else:
            self.card_count[card["number"]] += 1
        return card

    def player_action(self, action):
        """Process the player's action, either hit or stay."""
        
        if action == "hit":
            self.player_hand.append(self.draw_card())
            return self.update_status()
        elif action == "stay":
            return self.update_status()
    
    def dealer_action(self, strategy: str = "basic"):
        """Make the dealer perform actions based on the specified strategy."""
        
        if strategy == "basic":
            while self.total_value(self.dealer_hand) <= 17:
                self.dealer_hand.append(self.draw_card())
        elif strategy == "greedy":
            while self.total_value(self.dealer_hand) < 21:
                self.dealer_hand.append(self.draw_card())
        elif strategy == "random":
            while random.choice([True, False]):
                self.dealer_hand.append(self.draw_card())

    
    def update_status(self, status = "continue"):
        """Update the game status based on the player's hand value."""
        
        player_value = self.get_playervalue()
        if player_value > 21:
            self.status = "player_bust"
        elif player_value == 21:
            self.status = "player_blackjack"
        else:
            self.status = status
        return self.status
    
    def get_status(self):
        return self.status
    
    

    def get_dealervalue(self):
        """Get the total value of the dealer's hand."""
        return self.total_value(self.dealer_hand)

    def get_playervalue(self):
        """Get the total value of the player's hand."""
        return self.total_value(self.player_hand)
    
    
    """
    一共两种gamemode，分别是对于player bust后的不同判断方式
    traditional：Player bust后直接判lose
    novel：Player bust后看Dealer，若dealer也bust则算draw
    """
    def game_result(self):
        """Determine the result of the game based on player and dealer hand values."""

        dealer_value = self.get_dealervalue()
        player_value = self.get_playervalue()
        
        if self.mode == "traditional":
            if player_value > 21:
                return "lose"
            elif dealer_value > 21 or player_value > dealer_value:
                return "win"
            elif player_value == dealer_value:
                return "draw"
            else:
                return "lose"
        elif self.mode == "novel":
            if player_value > 21 and dealer_value <= 21:
                return "lose"
            elif dealer_value > 21 or player_value > dealer_value:
                return "win"
            elif player_value == dealer_value or (player_value > 21 and dealer_value > 21):
                return "draw"
            else:
                return "lose"

        
    def start(self):
        """Start a new round by dealing two cards each to the player and dealer."""
        
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.update_status() # init the status as continue
    
    def reset(self):
        """Reset the game state and shuffle a new deck."""
        
        self.deck = self.generate_deck()
        random.shuffle(self.deck)
        self.player_hand = []
        self.dealer_hand = []        
        self.status = "continue"
        self.card_count = {
                      '2': 0,
                      '3': 0,
                      '4': 0,
                      '5': 0,
                      '6': 0,
                      '7': 0,
                      '8': 0,
                      '9': 0,
                      '10': 0,
                      'A': 0}
        
        
        
    
    def play(self, player_action: str, dealer_strategy: str, output = False):
        """Simulate a turn of the game with the specified player action and dealer strategy."""
        
        self.dealer_action(dealer_strategy)
        self.player_action(player_action)
        if output:
            print("Dealer has:", game.format_cards(game.dealer_hand), game.total_value(game.dealer_hand))
        return self.status
        
        
        
if __name__ == "__main__":
    game = BlackJack()
    for round in range(5):
        game.start()
        print("Dealer shows:", game.format_cards(game.dealer_hand[:1]))
        while game.status == "continue":
            print(game.format_cards(game.player_hand), game.total_value(game.player_hand))
            action = input("Enter an action (hit/stay): ")
            game.play(action, "basic")
            if action == "stay":
                break
        print(f"dealer's hand: {game.format_cards(game.dealer_hand)} {game.get_dealervalue()}  player's hand: {game.format_cards(game.player_hand)} {game.get_playervalue()}")
        print(game.game_result())
        print(f"card count: {game.card_count}")
        print("================================")
        game.reset()
