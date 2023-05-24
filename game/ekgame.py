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

    VEC_LEN = 12  # The length of an action vector.


class EKGame:
    """
    This class represents an Exploding Kittens (EK) game. It implements most of
    the game logic.
    """

    ACTION_HORIZON = 10  # See at most this many of the most recent actions
    MAX_PLAYERS = 5

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

        # History of actions taken by self and others, with 0 most recent:
        self.action_history = np.zeros([0, EKActionVecDefs.VEC_LEN])

        self.major_player = 0  # The player who's turn it is currently
        self.nope_player = -1  # The player who's turn it is to nope
        self.action_noped = False  # If someone noped (or un-unnoped :-p)
        self.unprocessed_action = None  # An action that will only be processed
        # after everyone has had a chance to nope.

        self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.legal_actions_long = np.zeros([0], dtype=np.int64)

        # TODO logic for the attack card!

    def update_state(self):
        # TODO!

        player = self.get_cur_player()
        self.calc_legal_actions(player)

    def get_state(
        self, long_form: bool
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the current visible state of the game for a specific player

        Returns:
            A tuple containing: (1) the index value of the current player, (2)
            an numpy array describing the players cards and what it knows about
            all the other cards, (3) the last N actions taken by self and other
            players, (4) a list of possible actions. The format depends on the
            `long_form` boolean argument given. See `EKGame.get_legal_actions`.
        """
        player_index = self.get_cur_player()
        return [
            player_index,
            self.cards.get_state(player_index),
            self.player_centric_action_history(player_index),
            self.get_legal_actions(long_form)
        ]

    def take_action(self, player: int, action: int | np.ndarray):

        if isinstance(action, int): # Convert all actions to same format:
            action = self.action_from_long_form(action)
        action = self.player_centric_to_global(player, action)

        # Op het moment dat je de actie neemt, moet je de kaart al spelen en
        # komt ie in de actions array. Maar pas in de update_state doen we het
        # effect van de kaart, zodat er kans is om te nopen!

        # Also responsible for converting back to global player indeces instead of player centric ones
        pass

    def get_cur_player(self) -> int:
        """Returns the player who should take an action next"""
        if self.nope_player != -1:
            return self.nope_player
        if False:  # TODO here add condition for if it is a responder to gift!
            return 1
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

        cards = self.cards.cards[player]

        # Precondition for most actions is that we are the major player and that
        # there are no exploding kittens in the own deck:
        is_major = player == self.major_player
        major_and_no_ek = is_major and cards[EKCardTypes.EXPL_KITTEN] == 0

        # All relative player indeces, except yourself (at idx 1):
        legal_others = np.arange(EKGame.MAX_PLAYERS)
        legal_others = np.delete(legal_others, 1)

        # Most actions associated to cards:
        self.push_legal_action(
            major_and_no_ek, cards, EKCardTypes.ATTACK, EKActionVecDefs.PLAY_ATTACK
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
                    0,
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
                EKCardTypes.FAVOR,
                -1,
                0,
                c_idx,
                0,
            )

    def push_legal_action(
        self,
        preconditions: bool,
        cards: np.ndarray,
        card: int,
        action: int,
        pointer: int = 0,
        target: int = 0,
        card_th: int = 1,
    ):
        """
        Checks if an action would be legal, and pushes to array of legal actions
        if so.

        Args:
            preconditions (bool): only even consider pushing if this is true
            cards (np.ndarray): the stack of cards belonging to the cur player
            card (EKCardTypes): the card associated with the action
            card_th: (int): you need this many cards to be able to play action
            action (EKActionVecDefs): the action itself. -1 for no action!
            pointer (int): the optional pointer. Usually gives relative index of
            another player, but sometimes something else.
            target (int): the optional target. Points to a card. For example, a
            card you want from someone, or want to give away to someone.
        """
        conditions_met = preconditions and cards[card] >= card_th
        if conditions_met:
            ac_vec = np.zeros(EKActionVecDefs.VEC_LEN)
            ac_vec[EKActionVecDefs.PLAYER] = 1  # Always 1 because player centric
            ac_vec[EKActionVecDefs.POINTER] = pointer
            ac_vec[EKActionVecDefs.TARGET_CARD] = target
            if action != -1:
                ac_vec[action] = 1
            self.legal_actions = np.append(self.legal_actions, [ac_vec], axis=0)
        self.legal_actions_long = np.append(self.legal_actions_long, conditions_met)
    
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
        history = self.action_history.copy()
        history[:, EKActionVecDefs.PLAYER] = \
            np.mod(history[:, EKActionVecDefs.PLAYER] - player + 1, self.num_players)
        mask = (history[:, EKActionVecDefs.PLAY_FAVOR] == 1) | \
                (history[:, EKActionVecDefs.PLAY_TWO_CATS] == 1) | \
                (history[:, EKActionVecDefs.PLAY_THREE_CATS] == 1)
        history[mask, EKActionVecDefs.POINTER] = \
            np.mod(history[:, EKActionVecDefs.POINTER] - player + 1, self.num_players)
        return history
