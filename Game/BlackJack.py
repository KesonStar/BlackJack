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
        result = ""
        for card in cards:
            suit = BlackJack.suits[card["suit"]]
            result += f"{card['number']}{suit} "
        
        return result.strip()
    
    def generate_deck(self):
        numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        deck = [{'number': number, 'suit': suit} for number in numbers for suit in suits]
        return deck
    
    # Compute the value of hand
    def total_value(self, hand: list):
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
        card = self.deck.pop()
        if card["number"] in ['J', 'Q', 'K']: 
            self.card_count["10"] += 1
        else:
            self.card_count[card["number"]] += 1
        return card

    def player_action(self, action):
        if action == "hit":
            self.player_hand.append(self.draw_card())
            self.update_status()
        elif action == "stay":
            self.update_status("stay")
    
    def dealer_action(self, strategy: str = "basic"):
        
        if strategy == "basic":
            while self.total_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
        elif strategy == "greedy":
            while self.total_value(self.dealer_hand) < 21:
                self.dealer_hand.append(self.draw_card())
        elif strategy == "random":
            while random.choice([True, False]):
                self.dealer_hand.append(self.draw_card())

    
    def update_status(self, status = "continue"):
        player_value = self.get_playervalue()
        if player_value > 21:
            self.status = "player_bust"
        elif player_value == 21:
            self.status = "player_blackjack"
        else:
            self.status = status
    

    """
    一共两种gamemode，分别是对于player bust后的不同判断方式
    traditional：Player bust后直接判lose
    novel：Player bust后看Dealer，若dealer也bust则算draw
    """

    def get_dealervalue(self):
        return self.total_value(self.dealer_hand)

    def get_playervalue(self):
        return self.total_value(self.player_hand)
    
    def game_result(self):

        dealer_value = self.get_dealervalue()
        player_value = self.get_playervalue()

        if player_value > 21:
            if self.mode == "traditional" or self.mode == "novel" and dealer_value <= 21:
                return "lose"
            elif self.mode == "novel" and dealer_value > 21:
                return "draw"
            
        elif dealer_value > 21 or player_value > dealer_value:
            return "win"
        elif player_value == dealer_value:
            return "draw"
        else:
            return "loss"
        
    def start(self):
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.update_status()
    
    def reset(self):
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
        print(f"dealer's hand: {game.format_cards(game.dealer_hand)} {game.get_dealervalue()}  player's hand: {game.format_cards(game.player_hand)} {game.get_playervalue()}")
        print(game.game_result())
        print(f"card count: {game.card_count}")
        print("================================")
        game.reset()
    
    
    

        
   
    
    

        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    