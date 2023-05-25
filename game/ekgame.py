import numpy as np
from ekcards import EKCards, EKCardTypes


class EKActionVecDefs:
    """
    Actions are encoded using a vector representation. This class contains
    definitions for such a vector. It tells what indeces into the vector have
    what semantic meaning. (For example, 3 till 11 are one-hot encoded actions
    to take, while 0 till 2 specify arguments that go with said actions).

    Important to note, is that drawing a card from the deck or deciding not to
    play the NOPE card are encoded by all zeros!
    Moreover, when a player is the target of a FAVOR (i.e. he/she has to give
    a card of his/her choosing to someone else), all values should be zero,
    except arr[TARGET_CARD], which should specify the card to give (i.e.
    `EKCardTypes` values).
    """

    PLAYER = 0  # The player making the move

    POINTER = 1  # For cards like GIFT, you have to pick a player to take
    # a gift from. POINTER should then give the index of the
    # player you picked (relative to self). In addition,
    # when placing back a kitten, it should give the index
    # to insert it (-1) for random.

    TARGET_CARD = 2  # With GIFT or FAVOR, you have to select a card to give
    # or take. Thats what you use the val at this index for

    # Now one-hot encoded actions:
    PLAY_ATTACK = 3
    PLAY_FAVOR = 4  # Needs to also set POINTER to a player
    PLAY_NOPE = 5
    PLAY_SHUFFLE = 6
    PLAY_SKIP = 7
    PLAY_SEE_FUTURE = 8
    PLAY_TWO_CATS = 9  # Needs to also set POINTER to a player
    PLAY_THREE_CATS = 10  # NEEDS to set POINTER to a player, and TARGET_CARD
    # to the card he/she wants.
    
    PLACE_BACK_KITTEN = 11

    FUTURE_1 = 12   # For showing future to the player who requested future
    FUTURE_2 = 13
    FUTURE_3 = 14

    VEC_LEN = 15  # The length of an action vector.

class EKGame:
    """
    This class represents an Exploding Kittens (EK) game. It implements most of
    the game logic.
    """

    ACTION_HORIZON = 10  # See at most this many of the most recent actions
    MAX_PLAYERS = 5
    
    REWARD = 0.2        # Everyone gets this when some player is game over
    PENALTY = 1.0       # The game over player gets this

    def __init__(self) -> None:
        """Constructor. Does not do much."""
        self.cards = EKCards()

    def reset(self, num_players: int):
        """
        Reset the game completely.

        Args:
            num_players (int): The number of players to put in the game. Minimum
            is 2, maximum is 5.
        """

        self.num_players = num_players
        self.cards.reset(num_players)

        # Keeps track of who is still in the game:
        self.still_playing = np.ones(num_players, dtype=bool)

        # Reward to still return next time player gets turn:
        self.reward_buffer = np.zeros(num_players, dtype=np.int64)
        self.reward = 0     # Actual reward to return

        # History of actions taken by self and others, with 0 most recent:
        self.action_history = np.zeros([0, EKActionVecDefs.VEC_LEN])

        self.major_player = 0  # The player who's turn it is currently
        self.nope_player = -1  # The player who's turn it is to nope
        self.action_noped = False  # If someone noped (or un-unnoped :-p)
        self.unprocessed_action = None  # An action that will only be processed
        # after everyone has had a chance to nope.

        self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.legal_actions_long = np.zeros([0], dtype=np.int64)

        self.attack_count = 0 # Number of extra cards attacked player must pick
        self.attack_activated = False # Gets true when player could not defend
        # by playing an attack card too

    def update_state(self):
        player = self.get_cur_player()

        # Check if we drew an exploding kitten, but have no defuse:
        if self.cards.cards[
            EKCards.FIRST_PLAYER_IDX + player, EKCardTypes.EXPL_KITTEN] > 0 \
                and self.cards.cards[
                    EKCards.FIRST_PLAYER_IDX + player, EKCardTypes.DEFUSE] == 0:
            
            # Set player to non playing:
            self.still_playing[player] = False
            
            # Give rewards and penalty:
            self.reward_buffer += EKGame.REWARD
            self.reward_buffer[player] -= EKGame.REWARD + EKGame.PENALTY
            self.reward = self.reward_buffer[player]

            # Making list of possible actions empty marks game over:
            self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
            self.legal_actions_long = np.zeros([0], dtype=np.int64)

            # Selecting next player:
            self.major_player = self.get_next_major_idx()

            # But returning old one to tell him/her its game over
            return player
        
        # Check if the whole game is over because there is only 1 player left:
        if np.sum(self.still_playing <= 1):
            # Signal that the game is over by giving no possible acitons:
            self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
            self.legal_actions_long = np.zeros([0], dtype=np.int64)
            return player

        # There might still be an action that must be processed:
        self.process_action(player)

        self.reward = self.reward_buffer[player]
        self.reward_buffer[player] = 0
        self.calc_legal_actions(player)

        return player

    def update_and_get_state(
        self, long_form: bool
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the current visible state of the game for a specific player

        Returns:
            A tuple containing: (1) the index of the current player, (2) the
            reward to return to that player, (3) an numpy array describing the
            players cards and what it knows about all the other cards, (4) the
            last N actions taken by self and other players, (5) a list of
            possible actions. The format depends on the `long_form` boolean
            argument given. See `EKGame.get_legal_actions`.
            ❗IMPORTANT: if list of possible acitons has length 0, it means the
            player is game over!
        """
        player_index = self.update_state()
        return [
            player_index,
            self.reward,
            self.cards.get_state(player_index),
            self.player_centric_action_history(player_index),
            self.get_legal_actions(long_form)
        ]

    def take_action(self, player: int, action: int | np.ndarray):
        """
        Let a player take an action.

        Args:
            player (int): index of player taking the action.
            action: (int | numpy array): Representation of an action. Either an
            index into the long-form array of possible actions, or one of the
            action arrays gotten from calling `get_state` with
            `long_form = False`.
        """
        if isinstance(action, int): # Convert all actions to same format:
            action = self.action_from_long_form(action)
        action = self.player_centric_to_global(player, action)

        # Player is the victim of an attack and does not play an attack too
        # PS: note that playing an attack while attack_activated == True is
        # illegal, so should not end up in self.legal_actions array
        if self.attack_count > 0 and action[EKActionVecDefs.PLAY_ATTACK] == 0:
            self.attack_activated = True

        # Player chose to not take any further actions, so either decided to
        # draw a card from deck or to not nope:
        if np.all(action == 0):
            if player == self.nope_player:
                self.nope_player = self.get_next_nope_idx()
                return
            
            # Passing when you are the major player means drawing a card, as
            # this is what you always have to do at the end of your turn:
            drawn_card = self.cards.random_pick(
                EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX + player)
            if drawn_card == EKCardTypes.EXPL_KITTEN:
                return # 💣🙀: In the update_state function check if game over

            # Move on to next player, unless we are the victim of an attack:
            if (self.attack_count == 0):
                self.major_player = self.get_next_major_idx()
            else:
                self.attack_count -= 1
                self.attack_activated = False if self.attack_count == 0 else True
            
            return
        
        # Every other type of action we register in the action history:
        self.push_action(action)

        # Player played NOPE:
        if action[EKActionVecDefs.PLAY_NOPE] == 1:
            self.cards.known_pick(
                EKCards.FIRST_PLAYER_IDX + player,
                EKCards.DISCARD_PILE_IDX, EKCardTypes.NOPE)
            self.action_noped = not self.action_noped
            self.nope_player = -1
            self.nope_player = self.get_next_nope_idx()

        # Player is the target of a PLAY_FAVOR:
        if player != self.major_player and action[EKActionVecDefs.PLAY_FAVOR]:
            card = action[EKActionVecDefs.TARGET_CARD]
            frm = EKCards.FIRST_PLAYER_IDX + player
            to = EKCards.FIRST_PLAYER_IDX + \
                self.unprocessed_action[EKActionVecDefs.PLAYER]
            self.cards.known_pick(frm, to, card)
            self.unprocessed_action = None
            return

        # For all other actions we must give everyone chance to NOPE:
        self.unprocessed_action = action
        self.action_noped = False
        self.nope_player = -1
        self.nope_player = self.get_next_nope_idx()

        self.play_cards(player, action)

    def get_cur_player(self) -> int:
        """Returns the player who should take an action next"""
        # First give everyone chance to nope, then, if favor card was played,
        # let the target of that card respond, and finally let major player play
        if self.nope_player != -1:
            return self.nope_player
        if self.unprocessed_action != None and \
                self.unprocessed_action[EKActionVecDefs.PLAY_FAVOR] == 1:
            return self.unprocessed_action[EKActionVecDefs.POINTER]
        return self.major_player

    def get_legal_actions(self, long_form: bool) -> np.ndarray:
        """
        Returns a numpy array describing what actions are legal for the current
        player in the current state. If `long_form` is `True`, it returns a
        binary mask, where only legal actions are set to 1. If it is `False`, it
        returns an array of possible actions, as specified by the
        `EKActionVecDefs` class/enum.
        """
        if long_form:
            return self.legal_actions_long
        return self.legal_actions

    def calc_legal_actions(self, player: int):
        """
        Initializes self.legal_actions and self.legal_actions_long with the
        actions that are legal to take for the current player (`player` arg)
        """

        self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.legal_actions_long = np.zeros([0], dtype=np.int64)

        cards = self.cards.cards[EKCards.FIRST_PLAYER_IDX + player]

        # Precondition for most actions is that we are the major player and that
        # there are no exploding kittens in the own deck:
        is_major = player == self.major_player
        major_and_no_ek = is_major and cards[EKCardTypes.EXPL_KITTEN] == 0

        # All relative player indeces, except yourself (at idx 1):
        legal_others = np.arange(EKGame.MAX_PLAYERS)
        legal_others = np.delete(legal_others, 1)

        # Passing:
        self.push_legal_action(
            major_and_no_ek or player == self.nope_player, cards, -1, -1)
        
        # Most actions associated to cards:

        # Attack only allowed if you did not play any other moves yet in your
        # turn, and you are the first or second to play the attack card:
        self.push_legal_action(
            major_and_no_ek and self.attack_activated == False \
                and self.attack_count < 3,
            cards, EKCardTypes.ATTACK, EKActionVecDefs.PLAY_ATTACK
        )
        for p_idx in legal_others:
            self.push_legal_action(
                major_and_no_ek and p_idx < self.num_players,
                cards,
                EKCardTypes.FAVOR,
                EKActionVecDefs.PLAY_FAVOR,
                p_idx,
            )
        self.push_legal_action(
            major_and_no_ek, cards, EKCardTypes.SHUFFLE, EKActionVecDefs.PLAY_SHUFFLE
        )
        self.push_legal_action(
            major_and_no_ek, cards, EKCardTypes.SKIP, EKActionVecDefs.PLAY_SKIP
        )
        self.push_legal_action(
            major_and_no_ek,
            cards,
            EKCardTypes.SEE_FUTURE,
            EKActionVecDefs.PLAY_SEE_FUTURE,
        )

        # Finding the cat card (CAT_A, to CAT_E) we have the most of:
        cats_mask = np.zeros_like(cards)
        cats_mask[EKCardTypes.CAT_A :] = 1
        max_cats_card = np.argmax(cards * cats_mask)

        # Adding all actions related to having two or three cat cards of same:
        for p_idx in legal_others:
            self.push_legal_action(
                major_and_no_ek and p_idx < self.num_players,
                cards,
                max_cats_card,
                EKActionVecDefs.PLAY_TWO_CATS,
                p_idx,
                0,
                2,
            )

            for c_idx in range(1, EKCardTypes.NUM_TYPES):
                self.push_legal_action(
                    major_and_no_ek and p_idx < self.num_players,
                    cards,
                    max_cats_card,
                    EKActionVecDefs.PLAY_THREE_CATS,
                    p_idx,
                    c_idx,
                    3,
                )

        # Placing back an exploding kitten:
        for deck_idx in range(-1, EKCards.INIT_DECK_ORDERED_LEN):
            self.push_legal_action(
                is_major,
                cards,
                EKCardTypes.EXPL_KITTEN,
                EKActionVecDefs.PLACE_BACK_KITTEN,
                deck_idx,
            )

        # Play nope card when its the kind of game where it is not your turn:
        self.push_legal_action(
            player == self.nope_player,
            cards,
            EKCardTypes.NOPE,
            EKActionVecDefs.PLAY_NOPE,
        )

        # If we are not the major player, and we are not a nope player, we must
        # be the target of a FAVOR card someone else played:
        for c_idx in range(1, EKCardTypes.NUM_TYPES):
            self.push_legal_action(
                not is_major and player != self.nope_player,
                cards,
                -1,
                EKActionVecDefs.PLAY_FAVOR,
                0,
                c_idx,
            )

    def push_legal_action(
        self,
        preconditions: bool,
        cards: np.ndarray,
        card: int,
        action: int,
        pointer: int = 0,
        target_card: int = 0,
        card_th: int = 1,
    ):
        """
        Checks if an action would be legal, and pushes to array of legal actions
        if so.

        Args:
            preconditions (bool): only even consider pushing if this is true
            cards (np.ndarray): the stack of cards belonging to the cur player
            card (EKCardTypes): the card associated with the action. -1 for none
            card_th: (int): you need this many cards to be able to play action
            action (EKActionVecDefs): the action itself. -1 for no action!
            pointer (int): the optional pointer. Usually gives relative index of
            another player, but sometimes something else.
            target (int): the optional target. Points to a card. For example, a
            card you want from someone, or want to give away to someone.
        """
        conditions_met = preconditions and (card == -1 or cards[card] >= card_th)
        if conditions_met:
            ac_vec = np.zeros(EKActionVecDefs.VEC_LEN)
            ac_vec[EKActionVecDefs.PLAYER] = 1  # Always 1 because player centric
            ac_vec[EKActionVecDefs.POINTER] = pointer
            ac_vec[EKActionVecDefs.TARGET_CARD] = target_card
            if action != -1:
                ac_vec[action] = 1
            self.legal_actions = np.append(self.legal_actions, [ac_vec], axis=0)
        self.legal_actions_long = np.append(self.legal_actions_long, conditions_met)
    
    def play_cards(self, player: int, action: np.ndarray):
        """ Makes sure the card(s) corresponding to action go to discard pile """
        
        # Check if played anything except cat cards:
        if action[EKActionVecDefs.PLAY_TWO_CATS] != 1 and \
                action[EKActionVecDefs.PLAY_THREE_CATS] != 1:
            
            if action[EKActionVecDefs.PLAY_ATTACK] == 1:
                card = EKCardTypes.ATTACK
            elif action[EKActionVecDefs.PLAY_FAVOR] == 1:
                card = EKCardTypes.FAVOR
            elif action[EKActionVecDefs.PLAY_SHUFFLE] == 1:
                card = EKCardTypes.SHUFFLE
            elif action[EKActionVecDefs.PLAY_SKIP] == 1:
                card = EKCardTypes.SKIP
            else:
                card = EKCardTypes.SEE_FUTURE
            
            self.cards.known_pick(
                EKCards.FIRST_PLAYER_IDX + player, EKCards.DISCARD_PILE_IDX, card)
            return
        
        cat_cards = self.cards.cards[
            EKCards.FIRST_PLAYER_IDX + player, EKCardTypes.CAT_A:].copy()
        
        # Check if played 2 cat cards:
        if action[EKActionVecDefs.PLAY_TWO_CATS] == 1:
            cat_cards[cat_cards < 2] = 0
            nonzero = np.nonzero(cat_cards)[0]
            card = nonzero[cat_cards[nonzero].argmin()]
            card += EKCardTypes.CAT_A
            for _ in range(2):
                self.cards.known_pick(
                    EKCards.FIRST_PLAYER_IDX + player,
                    EKCards.DISCARD_PILE_IDX, card)
            return
        
        # Check if played 3 cat cards:
        if action[EKActionVecDefs.PLAY_THREE_CATS] == 1:
            cat_cards[cat_cards < 3] = 0
            nonzero = np.nonzero(cat_cards)[0]
            card = nonzero[cat_cards[nonzero].argmin()]
            card += EKCardTypes.CAT_A
            for _ in range(3):
                self.cards.known_pick(
                    EKCards.FIRST_PLAYER_IDX + player,
                    EKCards.DISCARD_PILE_IDX, card)
            return
        
        raise RuntimeError(f"Got illegal action: {action}")

    def action_from_long_form(self, idx) -> np.ndarray:
        """ Return action corresponding to index into `legal_actions_long`. """
        indeces = np.cumsum(self.legal_actions_long) - 1
        return self.legal_actions[indeces[idx]]
    
    def player_centric_to_global(self, player: int, action: np.ndarray):
        """ Players view everything player centric: they don't know their own
        index, but instead think of themselves as index 1, the player before as
        index 0, and the players after as index 1 to N. This function converts
        actions players take back into a global point of view.
        """
        action[EKActionVecDefs.PLAYER] = player
        if action[EKActionVecDefs.PLAY_FAVOR] == 1 or \
                action[EKActionVecDefs.PLAY_TWO_CATS] == 1 or \
                action[EKActionVecDefs.PLAY_THREE_CATS] == 1:
            action[EKActionVecDefs.POINTER] = \
                np.mod(action[EKActionVecDefs.POINTER] + player - 1, self.num_players)

        return action
    
    def player_centric_action_history(self, player: int):
        """ Converts the action history into a player centric form """

        # Filter out all observed futures for other players than current one:
        mask = (
            (self.action_history[:, EKActionVecDefs.FUTURE_1] == 0) &
            (self.action_history[:, EKActionVecDefs.FUTURE_2] == 0) &
            (self.action_history[:, EKActionVecDefs.FUTURE_3] == 0)) | \
        (self.action_history[:, EKActionVecDefs.PLAYER] == player)

        history = self.action_history[mask].copy()
        history[:, EKActionVecDefs.PLAYER] = \
            np.mod(history[:, EKActionVecDefs.PLAYER] - player + 1, self.num_players)
        mask = (history[:, EKActionVecDefs.PLAY_FAVOR] == 1) | \
                (history[:, EKActionVecDefs.PLAY_TWO_CATS] == 1) | \
                (history[:, EKActionVecDefs.PLAY_THREE_CATS] == 1)
        history[mask, EKActionVecDefs.POINTER] = \
            np.mod(history[:, EKActionVecDefs.POINTER] - player + 1, self.num_players)
        return history

    def push_action(self, action: np.ndarray):
        """ Push a taken action on the action_history array, and make sure size 
        limit is not exceeded. """
        if len(self.action_history < EKGame.ACTION_HORIZON):
            self.action_history = np.insert(self.action_history, 0, action, axis = 0)
            return

        self.action_history = np.roll(self.action_history, 1, axis = 0)
        self.action_history[0] = action

    def process_action(self, player: int):
        """ Process the `self.unprocessed_action` """
        # Check if we can ignore unprocessed action:
        if player != self.major_player or self.unprocessed_action == None or \
                self.action_noped:
            self.unprocessed_action = None
            self.action_noped = False
            return
        
        ac = self.unprocessed_action

        # Attack: skip your turn and let player after you take two turns. If
        # you are the receiver of an attack, and play an attack too, you skip
        # your turn also, and player after you has to draw 4 cards:
        if ac[EKActionVecDefs.PLAY_ATTACK] == 1:
            self.attack_activated = False
            # Should be 1 the first time, so next player must play 2 rounds
            # and 3 the second time, so next player plays 4 rounds total:
            self.attack_count = self.attack_count * 2 + 1
            self.major_player = self.get_next_major_idx()
        
        # Shuffle the deck:
        elif ac[EKActionVecDefs.PLAY_SHUFFLE] == 1:
            self.cards.shuffle()
        
        # Skip your turn so you don't have to draw a card from the deck:
        elif ac[EKActionVecDefs.PLAY_SKIP] == 1:
            if self.attack_count == 0:
                self.major_player = self.get_next_major_idx()
            else:
                self.attack_count -= 1
                self.attack_activated = False if self.attack_count == 0 else True


        # See the top 3 cards in the deck:
        elif ac[EKActionVecDefs.PLAY_SEE_FUTURE] == 1:
            future = self.cards.see_future()
            
            # Via the action history we communicate future to player:
            acvec = np.zeros(EKActionVecDefs.VEC_LEN)
            acvec[EKActionVecDefs.PLAYER] = player
            acvec[EKActionVecDefs.FUTURE_1:] = future
            self.push_action(acvec)
        
        # Two cats: Take a random card from ohter player:
        elif ac[EKActionVecDefs.PLAY_TWO_CATS] == 1:
            frm = ac[EKActionVecDefs.POINTER] + EKCards.FIRST_PLAYER_IDX
            to = player + EKCards.FIRST_PLAYER_IDX
            self.cards.random_pick(frm, to)
        
        # Three cats: take a card of your choosing from other (if it has it):
        elif ac[EKActionVecDefs.PLAY_THREE_CATS] == 1:
            frm = ac[EKActionVecDefs.POINTER] + EKCards.FIRST_PLAYER_IDX
            to = player + EKCards.FIRST_PLAYER_IDX
            card = ac[EKActionVecDefs.TARGET_CARD]
            if self.cards.cards[frm, card] > 0:
                self.cards.known_pick(frm, to, card)

                # Asking a card is public, so now everyone knows you have it:
                self.cards.known_map[:, to, card] += 1

        self.unprocessed_action = None
        

    def get_next_major_idx(self) -> int:
        """ Get the index of the next player whos turn it is """
        idx = self.major_player
        while True:
            idx = (idx + 1) % self.num_players
            if self.still_playing[idx]:
                return idx
            if idx == self.major_player:
                return idx  # Just to be sure we don't have an infinite loop
    
    def get_next_nope_idx(self) -> int:
        idx = self.nope_player
        while True:
            idx += 1
            if idx == self.num_players:
                return -1
            if self.cards.cards[EKCards.FIRST_PLAYER_IDX + idx, EKCardTypes.NOPE] > 0:
                return idx