import numpy as np

class EKCardTypes:
    """ Enum specifying which integers correspond to what cards """
    
    # Card index definitions:
    EXPL_KITTEN = 0
    DEFUSE      = 1
    ATTACK      = 2
    FAVOR       = 3
    NOPE        = 4
    SHUFFLE     = 5
    SKIP        = 6
    SEE_FUTURE  = 7

    CAT_A       = 8
    CAT_B       = 9
    CAT_C       = 10
    CAT_D       = 11
    CAT_E       = 12

    # Total number of different card types:
    NUM_TYPES   = 13


class EKCards:
    """
    This class represents all cards in the game, and has methods to move them
    around in a safe manner.

    Attributes:
        total_deck (ndarray): Specifies how many cards of what type are in the
        deck (i.e. a count for each of the 13 different types). Can be used for
        card counting.
        DECK_IDX (int): index of the deck. Needed for (known/random)_pick
        DISCARD_PILE_IDX (int): index of the discard pile
        FIRST_PLAYER_IDX (int): index of the first player. All other players
        come after that, so you can do `FIRST_PLAYER_IDX + n`
    """

    DECK_IDX            = 0         # defining indeces into self.cards:
    DISCARD_PILE_IDX    = 1
    FIRST_PLAYER_IDX    = 2

    INIT_NUM_CARDS_TO_DEAL = 7      # 8 if you count the defuse!
    INIT_DECK_ORDERED_LEN = 7       # We can know at most the top 7 cards of deck


    def reset(self, num_players: int):
        """
        Resets the deck to a completely new game.

        Args:
            num_players (int): The number of players to put in the new game.
            Minimum is 2, maximum is 5.
        """
        
        if num_players < 2 or num_players > 5:
            raise RuntimeError("Number of players must be between 2 and 5")

        self.num_players = num_players
        
        # Applying the rules for how many exploding kittens and defuses to put in:
        expl_kits = num_players - 1
        extra_defuses = 2 if num_players <= 3 else 6 - num_players
        total_defuses = num_players + extra_defuses

        # The total number of cards of each type in the game:
        self.total_deck = np.array([
            expl_kits,      # EXPL_KITTEN
            total_defuses,  # DEFUSE
            4,              # ATTACK
            4,              # FAVOR
            5,              # NOPE
            4,              # SHUFFLE
            4,              # SKIP
            5,              # SEE_FUTURE

            4,              # CAT_A
            4,              # CAT_B
            4,              # CAT_C
            4,              # CAT_D
            4               # CAT_E
        ])
        
        # 2 more stacks than number of players: 1 for deck and  1 for discard pile:
        num_stacks = num_players + 2
        self.cards = np.zeros([num_stacks, EKCardTypes.NUM_TYPES], dtype=np.int64)

        # "known_map[3, 4, 5] == 2" means that player 3 knows that stack 4 has 2
        # cards of type 5. ❗*IMPORTANT* note that this does not tell what
        # anyone knows about him/herself.
        self.known_map = \
            np.zeros([num_players, num_stacks, EKCardTypes.NUM_TYPES], dtype=np.int64)

        # Sometimes it is known what cards are where in the deck. We must
        # account for this, by keeping a list of known cards. The following
        # array does this. -1 means unknown, otherwise we use values from the
        # Cards enum. Index 0 is the next card to be drawn. We keep track of
        # only the num_players amount of cards at the top of the stack.
        self.deck_ordered = -1 * np.ones(EKCards.INIT_DECK_ORDERED_LEN, dtype=np.int64)

        self.deal_cards(extra_defuses)


    def get_state(self, player_idx: int):
        """
        Get the state for a specific player.

        Args:
            player_idx (int): index of the player (from 0 to N - 1) to get a
            state representation for.
        
        Returns:
            A numpy array of dimensions [N + 2, 14], containing what you know
            about each of the N players in the game, as well as what you know
            about the deck and the discard pile. In addition giving the number
            of known cards per type (13), it gives the total number of cards.
            index 0 gives deck, index 1 gives discard pile, index 2 gives cards
            of player before you, index 3 gives your own cards, and the rest
            gives known cards of all players after you.
        """

        shape = self.cards.shape
        ret = np.zeros([shape[0], shape[1] + 1], dtype=np.int64)

        # You can see how many cards other players have:
        ret[:, 0] = np.sum(self.cards, axis = 1)

        # And you might know some more stuff about other players:
        ret[:, 1:] = self.known_map[player_idx]

        # About yourself you know everything, ofc
        ret[EKCards.FIRST_PLAYER_IDX + player_idx, 1:] = \
            self.cards[EKCards.FIRST_PLAYER_IDX + player_idx]

        # And the same goes for the discard pile:
        ret[EKCards.DISCARD_PILE_IDX, 1:] = \
            self.cards[EKCards.DISCARD_PILE_IDX]

        # Now, rotate, such that you are always the second player:
        ret[EKCards.FIRST_PLAYER_IDX:] = \
            np.roll(ret[EKCards.FIRST_PLAYER_IDX:], -player_idx + 1, axis = 0)

        return ret
    
    
    def deal_cards(self, extra_defuses: int):
        """
        Dealing cards to players, according to the manual of the game.
        """
        
        # Step 1, 2 and 3: remove all exploding kittens and most defuses:
        self.cards[EKCards.DECK_IDX] = self.total_deck
        self.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = 0
        self.cards[EKCards.DECK_IDX, EKCardTypes.DEFUSE] = extra_defuses

        # Dealing cards:
        for p_idx in range(self.num_players):
            # Step 2 again: deal a defuse to each player:
            self.cards[EKCards.FIRST_PLAYER_IDX + p_idx, EKCardTypes.DEFUSE] = 1
            
            # Step 4: deal 7 cards to each player. Everyone now has 8 total
            for _ in range(EKCards.INIT_NUM_CARDS_TO_DEAL):
                self.random_pick(EKCards.DECK_IDX,
                                 EKCards.FIRST_PLAYER_IDX + p_idx)
        
        # Step 5: Insert back the exploding kittens:
        self.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = \
            self.total_deck[EKCardTypes.EXPL_KITTEN]

        # Now everyone knows that everyone has at least one defuse:
        self.known_map[:, EKCards.FIRST_PLAYER_IDX:, EKCardTypes.DEFUSE] = 1

        # And everyone knows how many exploding kittens are in the deck:
        self.known_map[:, EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = \
            self.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN]
    

    def random_pick(self, from_idx: int, to_idx: int) -> int:
        """
        Randomly pick a card from the `from_idx` stack, and place it on
        the `to_idx` stack.
         
        Returns:
             (EKCardTypes int): the card that was randomly picked."""

        # Check if we can deterministically draw from deck first:
        if (from_idx == EKCards.DECK_IDX and self.deck_ordered[0] != -1):
            picked_card = self.deck_ordered[0]
        else:
            probs = self.cards[from_idx].copy()
            
            # For deck, take out everything that has a fixed place:
            if (from_idx == EKCards.DECK_IDX):
                for card in self.deck_ordered:
                    if card != -1:
                        probs[card] -= 1

            if np.sum(probs) == 0:
                raise RuntimeError("Trying to randomly pick card from empty!")
            probs = probs / (np.sum(probs))
            picked_card = np.random.choice(len(probs), p = probs)
        self.known_pick(from_idx, to_idx, picked_card)

        return picked_card
    

    def known_pick(self, from_idx: int, to_idx: int, picked_card: int):
        """ Remove `picked_card` from the `from_idx` stack, and place it on the
        `to_idx` stack. """

        if self.cards[from_idx, picked_card] == 0:
            raise RuntimeError("Trying to pick card that is not there!")
        
        # The main operation: moving card from one deck to another
        self.cards[from_idx, picked_card] -= 1
        self.cards[to_idx, picked_card] += 1

        # When a card is placed on the discard pile, everyone sees which card
        # this is. If you knew that player had this card, you now know nothing,
        # but if you know he/she has some other card, you still do:
        if from_idx >= EKCards.FIRST_PLAYER_IDX and to_idx == EKCards.DISCARD_PILE_IDX:
            self.known_map[:, from_idx, picked_card] -= 1
            self.known_map = np.clip(self.known_map, 0, np.Inf)
            return

        # If from_idx corresponds to a player, this player now knows that the
        # to_idx deck has one (more) of this card
        if from_idx >= EKCards.FIRST_PLAYER_IDX:
            player_idx = from_idx - EKCards.FIRST_PLAYER_IDX
            self.known_map[player_idx, to_idx, picked_card] += 1
        
        # Any card could have been taken from the from_idx deck, so if you knew
        # this deck had one card of a certain type, you now dont know anything:
        self.known_map[:, from_idx, :] -= 1

        # Only the player corresponding to to_idx knows a little more, so
        # partially undoing the above step here:
        if to_idx >= EKCards.FIRST_PLAYER_IDX:
            player_idx = to_idx - EKCards.FIRST_PLAYER_IDX
            self.known_map[player_idx, from_idx, :] += 1
            self.known_map[player_idx, from_idx, picked_card] -= 1

        # Clip to zero. You cant know a player has -1 cards ofc lol :-p
        self.known_map = np.clip(self.known_map, 0, np.Inf)

        # If we picked the top card from the *deck*, we have to move the order.
        # Note that I am just assuming we are picking from the top, as the
        # vanilla game does not allow anything else!
        if from_idx == EKCards.DECK_IDX:
            self.deck_ordered[0] = -1
            self.deck_ordered = np.roll(self.deck_ordered, -1)
        
        # Everyone always knows the number of exploding kittens in the deck:
        self.known_map[:, EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = \
            self.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN]
    

    def insert_kitten(self, player_idx: int, placement_idx: int):
        """
        After a player defused a kitten, he/she has to place it back, but can
        decide where in the deck to place it.

        Args:
            player_idx (int): the index (from 0 to 1 - num_players) of the player
            that is placing back the exploding kitten. The corresponding stack
            should contain an exploding kitten!
            placement_idx (int): where in the deck to place the kitten, ranging
            from 0, till the size of the deck, or the size of the max knowns.
            If -1 is given, the kitten is randomly placed.
        """

        self.known_pick(EKCards.FIRST_PLAYER_IDX +  player_idx,
                        EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN)
        
        if placement_idx >= len(self.deck_ordered) or \
            placement_idx >= np.sum(self.cards[EKCards.DECK_IDX]) or \
            placement_idx < 0:
            return  # We interpret this as random placement
        
        self.deck_ordered[placement_idx + 1:] = self.deck_ordered[placement_idx:-1]
        self.deck_ordered[placement_idx] = EKCardTypes.EXPL_KITTEN


    def see_future(self, played_idx) -> np.ndarray:
        """
        With the FUTURE card a player is allowed to see the top 3 cards on
        the deck.
         
        Args:
            played_idx (int): the player that gets to see the future. Its known
            list will be updated accordingly.
        
        Returns:
            A numpy array containing the top 3 cards on the deck, with index 0
            being the next on the be drawn, and index 2 the last.
        """

        tmp_deck = self.cards[EKCards.DECK_IDX].copy()
        top_3 = self.deck_ordered[:3] # not .copy, so changes in place!

        # First take what we know out, because we draw without replacement:
        for card_idx in top_3:
            if card_idx != -1:
                tmp_deck[card_idx] -= 1
        
        # Now we draw what we don't know yet:
        for idx in range(len(top_3)):
            if top_3[idx] != -1:
                continue            # Was already known
            
            if np.sum(tmp_deck) <= 0:
                break               # tmp_deck is empty

            probs = tmp_deck / np.sum(tmp_deck)
            picked_card = np.random.choice(len(probs), p = probs)
            top_3[idx] = picked_card
            tmp_deck[picked_card] -= 1
        
        # The player that is peeking now knows these cards are in the deck:
        self.known_map[played_idx, EKCards.DECK_IDX, top_3] = \
            np.clip(self.known_map[played_idx, EKCards.DECK_IDX, top_3], 1, np.Inf)

        return top_3.copy()
    
    def shuffle(self):
        """ With a SHUFFLE card a player can shuffle the deck """

        self.deck_ordered = -1 * np.ones_like(self.deck_ordered)