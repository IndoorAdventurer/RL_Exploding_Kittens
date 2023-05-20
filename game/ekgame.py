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


    PLAYER = 0          # The player making the move

    POINTER = 1         # For cards like GIFT, you have to pick a player to take
                        # a gift from. POINTER should then give the index of the
                        # player you picked (relative to self). In addition,
                        # when placing back a kitten, it should give the index
                        # to insert it (-1) for random.
    
    TARGET_CARD = 2     # With GIFT or FAVOR, you have to select a card to give
                        # or take. Thats what you use the val at this index for
    
    # Now one-hot encoded actions:
    PLAY_ATTACK = 3
    PLAY_FAVOR = 4          # Needs to also set POINTER to a player
    PLAY_NOPE = 5
    PLAY_SHUFFLE = 6
    PLAY_SKIP = 7
    PLAY_SEE_FUTURE = 8
    PLAY_TWO_CATS = 9       # Needs to also set POINTER to a player
    PLAY_THREE_CATS = 10    # NEEDS to set POINTER to a player, and TARGET_CARD 
                            # to the card he/she wants.
    PLACE_BACK_KITTEN = 11

    VEC_LEN = 12        # The length of an action vector.


class EKGame:
    """
    This class represents an Exploding Kittens (EK) game. It implements most of
    the game logic.
    """

    ACTION_HORIZON_LEN = 10     # get_state includes this many of last actions

    MAX_PLAYERS = 5

    def __init__(self) -> None:
        """ Constructor. Does not do mutch. """
        self.cards = EKCards()
    
    
    def reset(self, num_players: int):
        """
        Reset the game completely.

        Args:
            num_players (int): The number of players to put in the game. Minimum
            is 2, maximum is 5.
        """
        
        self.cards.reset(num_players)
        
        self.last_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])

        self.major_round_player = 0         # The person whose turn it is.

        # For going over players that can NOPE a move, or need to give a present
        # to major turn player. It is None, or an array of shape N x 2, where
        # arr[n, 0] is player index, and arr[n, 1] the action (nope or present)
        self.minor_round_players = None
        
        # To give the next time get_state lands on them
        self.reward_buffer = np.zeros(num_players)

        # After some plays ATTACK, the next player must draw two cards. If that
        # player also plays ATTACK, the player after that must play 4 (thats max)
        self.attack_cnt = 0
    

    def get_state(self) -> tuple[int, np.ndarray, np.ndarray]:
        """
        For the current player, get the for him/her observable state

        Returns:
            A 3-tuple with the following elements:
            1) A index of the current player, ranging from 0 to 4
            2) Everything the player knows about its own and others' cards. See
            `EKCards.get_state()` for more details.
            3) The last so many actions taken in the game (by self and others).
            Here, actions are sorted from most recent (idx 0) to least recent.
        """

        player = self.major_round_player if self.minor_round_players == None \
            else self.minor_round_players[0, 0]
        cards = self.cards.get_state(player)
        actions = self.last_actions.copy()
        # Actions should always be with yourself at idx 1, player before at idx 0
        actions[:, 0] = np.mod(actions[:, 0] + 1 - player, self.cards.num_players)
        return (player, cards, actions)

    def get_possible_actions(self, long_form: bool) -> np.ndarray:
        """
        Get a list of all actions that are legal in the current game state

        Args:
            long_form (bool): If `False`, returns an `N x VEC_LEN` (See 
            `EKActionVecDefs`) array, containing the N actions that are currently
            legal to take. This is the same format as the `actions` array of
            previously taken actions returned in the `get_state` method.
            If it is `True`, it returns an 82-dimensional array, serving as a
            binary mask over illegal actions. This option can be chosen when
            implementing a model that, for example, returns a Q-value for each
            of the 82 possible actions (of which only a fraction is legal at any
            time).
        
        Returns:
            A representation of the legal actions. The from depends on the
            long_form boolean argument given. See above.
        """
        
        self.actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.long_form = np.zeros(0)

        # All relative player indeces, except yourself (at idx 1):
        legal_others = np.arange(EKGame.MAX_PLAYERS)
        legal_others = np.delete(legal_others, 1)

        # If we are in a minor round, then we should ignore al major actions:
        conditions_good = self.minor_round_players == None

        player_idx = self.major_round_player if conditions_good \
            else self.minor_round_players[0, 0]
        cards = self.cards.cards[EKCards.FIRST_PLAYER_IDX + player_idx]

        # You can always decide to not take action, except when receiving gift:
        is_possible = conditions_good or \
            self.minor_round_players[0, 1] != EKActionVecDefs.PLAY_FAVOR
        if is_possible:
            arr = np.zeros(EKActionVecDefs.VEC_LEN)
            arr[EKActionVecDefs.PLAYER] = 1
            self.actions = np.append(self.actions, [arr], axis=0)
        self.long_form = np.append(self.long_form, is_possible)

        # An auxilary function for appending most actions:
        def action_aux(card: int, action: int, ptr: int = 0, target: int = 0, th: int = 1):
            is_possible = conditions_good and cards[card] >= th and \
                ptr < self.cards.num_players
            if is_possible:
                arr = np.zeros(EKActionVecDefs.VEC_LEN)
                arr[EKActionVecDefs.PLAYER] = 1
                arr[EKActionVecDefs.POINTER] = ptr
                arr[EKActionVecDefs.TARGET_CARD] = target
                arr[action] = 1
                self.actions = np.append(self.actions, [arr], axis=0)
            self.long_form = np.append(self.long_form, is_possible)
    
        action_aux(EKCardTypes.ATTACK, EKActionVecDefs.PLAY_ATTACK)
        for p_idx in legal_others:
            action_aux(EKCardTypes.FAVOR, EKActionVecDefs.PLAY_FAVOR, p_idx)
        action_aux(EKCardTypes.SHUFFLE, EKActionVecDefs.PLAY_SHUFFLE)
        action_aux(EKCardTypes.SKIP, EKActionVecDefs.PLAY_SKIP)
        action_aux(EKCardTypes.SEE_FUTURE, EKActionVecDefs.PLAY_SEE_FUTURE)
        
        cats_mask = np.zeros_like(cards)
        cats_mask[EKCardTypes.CAT_A:] = 1
        max_cats_card = np.argmax(cards * cats_mask)

        for p_idx in legal_others:
            action_aux(max_cats_card, EKActionVecDefs.PLAY_TWO_CATS, p_idx, 0, 2)

            for c_idx in range(1, EKCardTypes.NUM_TYPES):
                action_aux(max_cats_card, EKActionVecDefs.PLAY_THREE_CATS, p_idx, c_idx, 3)
        
        for d_idx in range(-1, EKCards.INIT_DECK_ORDERED_LEN):
            action_aux(EKCardTypes.EXPL_KITTEN, EKActionVecDefs.PLACE_BACK_KITTEN, d_idx)
        
        # For rest of actions we should be in a minor round:
        tmp_bool = self.minor_round_players != None
        minor_action = player_idx = self.major_round_player if conditions_good \
            else self.minor_round_players[0, 1]

        # Action should be specifided as PLAY_NOPE if we want to play nope
        conditions_good = tmp_bool and minor_action == EKActionVecDefs.PLAY_NOPE
        action_aux(EKCardTypes.NOPE, EKActionVecDefs.PLAY_NOPE)

        # Action should be PLAY_FAVOR if we must return a favor to major
        conditions_good = tmp_bool and minor_action == EKActionVecDefs.PLAY_FAVOR
        for c_idx in range(1, EKCardTypes.NUM_TYPES):
            if conditions_good:
                arr = np.zeros(EKActionVecDefs.VEC_LEN)
                arr[EKActionVecDefs.PLAYER] = 1
                arr[EKActionVecDefs.TARGET_CARD] = c_idx
                self.actions = np.append(self.actions, [arr], axis=0)
            self.long_form = np.append(self.long_form, conditions_good)
        
        if long_form:
            return self.long_form.copy()
        
        actions =  self.actions.copy()
        return actions


    def take_action(self, action: int | np.ndarray):
        """
        Accepts an action from a player, and updates the game accordingly.

        Arguments:
            action (int or ndarray): If it is an int, it will interpret it as
            the index into the 82-dimensional action space. If it is a numpy
            array, it will interpret it as an action vector as specified in the
            `EKActionVecDefs` class.
        """
        pass

    def push_action(self, action: np.ndarray):
        """ Push a taken action on the last_actions array, and make sure size 
        limit not exceeded. """
        if len(self.last_actions < EKGame.ACTION_HORIZON_LEN):
            self.last_actions = np.insert(self.last_actions, 0, action, axis = 0)
            return
        
        self.last_actions = np.roll(self.last_actions, 1, axis = 0)
        self.last_actions[0] = action