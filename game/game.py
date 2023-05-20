import numpy as np
from cards import EKCards, EKCardTypes


class EKGame:

    def __init__(self) -> None:
        
        self.cards = EKCards()
    
    def reset(self, num_players: int):
        self.cards.reset(num_players)