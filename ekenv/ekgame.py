import numpy as np
from .ekcards import EKCards, EKCardTypes


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
    
    DEFUSE_KITTEN = 11

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
        self.precompute_actions()
    
    def precompute_actions(self):
        """ There are 82 possible actions, and each has a unique representation.
        This function computes all of these beforehand. """
        self.all_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.all_actions_cards = np.zeros([0], dtype=np.int64)
        self.all_actions_pointers = np.zeros([0], dtype=np.int64)
        self.all_actions_target_cards = np.zeros([0], dtype=np.int64)

        # All relative player indeces, except yourself (at idx 1):
        legal_others = np.arange(EKGame.MAX_PLAYERS)
        legal_others = np.delete(legal_others, 1)

        # Passing:
        self.precompute_actions_aux(-1, -1)

        # Most action cards:
        self.precompute_actions_aux(EKCardTypes.ATTACK, EKActionVecDefs.PLAY_ATTACK)
        for p_idx in legal_others:
            self.precompute_actions_aux(EKCardTypes.FAVOR, EKActionVecDefs.PLAY_FAVOR, p_idx)
        self.precompute_actions_aux(EKCardTypes.SHUFFLE, EKActionVecDefs.PLAY_SHUFFLE)
        self.precompute_actions_aux(EKCardTypes.SKIP, EKActionVecDefs.PLAY_SKIP)
        self.precompute_actions_aux(EKCardTypes.SEE_FUTURE, EKActionVecDefs.PLAY_SEE_FUTURE)
        for p_idx in legal_others:
            self.precompute_actions_aux(-1, EKActionVecDefs.PLAY_TWO_CATS, p_idx)
            for t_idx in range(1, EKCardTypes.NUM_TYPES):
                self.precompute_actions_aux(-1, EKActionVecDefs.PLAY_THREE_CATS, p_idx, t_idx)
        
        # Defusing exploding kitten:
        for p_idx in range(-1, EKCards.INIT_DECK_ORDERED_LEN):
            self.precompute_actions_aux(EKCardTypes.DEFUSE, EKActionVecDefs.DEFUSE_KITTEN, p_idx)
        
        # Playing nope:
        self.precompute_actions_aux(EKCardTypes.NOPE, EKActionVecDefs.PLAY_NOPE)

        # Give favor:
        for t_idx in range(1, EKCardTypes.NUM_TYPES):
            self.precompute_actions_aux(-1, EKActionVecDefs.PLAY_FAVOR, 0, t_idx)
        
        # Make sure I did not forget any actions. (Will keep it in just in case)
        assert(len(self.all_actions) == 82)
        assert(len(self.all_actions_cards) == 82)
        assert(len(self.all_actions_pointers) == 82)
        assert(len(self.all_actions_target_cards) == 82)

    def precompute_actions_aux(self,
            card: int, action: int, pointer: int = 0, target_card: int = 0):
        """ auxiliary function for above """
        ac_vec = np.zeros(EKActionVecDefs.VEC_LEN)
        ac_vec[EKActionVecDefs.PLAYER] = 1  # Always 1 because player centric
        ac_vec[EKActionVecDefs.POINTER] = pointer
        ac_vec[EKActionVecDefs.TARGET_CARD] = target_card
        if action != -1:
            ac_vec[action] = 1
        self.all_actions = np.append(self.all_actions, [ac_vec], axis=0)
        self.all_actions_cards = np.append(self.all_actions_cards, card)
        self.all_actions_pointers = np.append(self.all_actions_pointers, pointer)
        self.all_actions_target_cards = \
            np.append(self.all_actions_target_cards, target_card)

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
        self.reward_buffer = np.zeros(num_players)
        self.reward = .0     # Actual reward to return

        # History of actions taken by self and others, with 0 most recent:
        self.action_history = np.zeros([0, EKActionVecDefs.VEC_LEN])

        self.major_player = 0  # The player who's turn it is currently
        self.nope_player = -1  # The player who's turn it is to nope
        self.action_noped = False  # If someone noped (or un-unnoped :-p)
        self.unprocessed_action = np.zeros(0) # An action that will only be
        # processed after everyone has had a chance to nope.

        self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
        self.legal_actions_long = np.zeros([82], dtype=np.int64)

        self.attack_count = 0 # Number of extra cards attacked player must pick
        self.attack_activated = False # Gets true when player could not defend
        # by playing an attack card too

        # Handy boolean for calculating legal actions:
        self.cur_player_is_major = False

    def update_state(self):
        # First check if FAVOR card noped, as this will affect get_cur_player:
        if self.action_noped and self.nope_player == -1 and \
            len(self.unprocessed_action) != 0 and \
                self.unprocessed_action[EKActionVecDefs.PLAY_FAVOR] == 1:
            self.action_noped = False
            self.unprocessed_action = np.zeros(0)

        player = self.get_cur_player()

        # Check if we drew an exploding kitten, but have no defuse:
        if self.cards.cards[
            EKCards.FIRST_PLAYER_IDX + player, EKCardTypes.EXPL_KITTEN] > 0 \
                and self.cards.cards[
                    EKCards.FIRST_PLAYER_IDX + player, EKCardTypes.DEFUSE] == 0:
            
            # Set player to non playing:
            self.still_playing[player] = False

            # Add to action history that player exploded:
            acvec = np.zeros(EKActionVecDefs.VEC_LEN)
            acvec[EKActionVecDefs.PLAYER] = player
            acvec[EKActionVecDefs.TARGET_CARD] = -1 # I use this to signal it
            self.push_action(acvec)
            
            # Give rewards and penalty:
            self.reward_buffer += EKGame.REWARD
            self.reward_buffer[player] -= EKGame.REWARD + EKGame.PENALTY
            self.reward = self.reward_buffer[player]

            # Making list of possible actions empty marks game over:
            self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
            self.legal_actions_long = np.zeros([0], dtype=np.int64)

            # If an attack caused this fatality, we must reset it:
            self.attack_count = 0
            self.attack_activated = False

            # Selecting next player:
            self.major_player = self.get_next_major_idx()

            # But returning old one to tell him/her its game over
            return player
        
        # Check if the whole game is over because there is only 1 player left:
        if np.sum(self.still_playing) <= 1:
            # Signal that the game is over by giving no possible acitons:
            self.legal_actions = np.zeros([0, EKActionVecDefs.VEC_LEN])
            self.legal_actions_long = np.zeros([0], dtype=np.int64)
            self.reward = self.reward_buffer[player]
            return player

        # There might still be an action that must be processed:
        self.process_action(player)

        # Processing action might have changed player in case of skip or attack:
        player = self.get_cur_player()

        self.reward = self.reward_buffer[player]
        self.reward_buffer[player] = 0
        self.calc_legal_actions(player)

        return player

    def update_and_get_state(
        self, long_form: bool
    ) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
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
            action = self.all_actions[action]
        action = self.player_centric_to_global(player, action)

        # Player is the victim of an attack and does not play an attack too
        # PS: note that playing an attack while attack_activated == True is
        # illegal, so should not end up in self.legal_actions array
        if self.attack_count > 0 and action[EKActionVecDefs.PLAY_ATTACK] == 0:
            self.attack_activated = True

        # Player chose to not take any further actions, so either decided to
        # draw a card from deck or to not nope:
        if np.all(action[EKActionVecDefs.PLAYER + 1:] == 0):
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

        # Player played DEFUSE card:
        if action[EKActionVecDefs.DEFUSE_KITTEN] == 1:
            self.cards.known_pick(
                EKCards.FIRST_PLAYER_IDX + player, EKCards.DISCARD_PILE_IDX,
                EKCardTypes.DEFUSE)
            self.cards.insert_kitten(player, int(action[EKActionVecDefs.POINTER]))
            if (self.attack_count == 0):
                self.major_player = self.get_next_major_idx()
            else:
                self.attack_count -= 1
                self.attack_activated = False if self.attack_count == 0 else True
            return
        
        # Player played NOPE:
        if action[EKActionVecDefs.PLAY_NOPE] == 1:
            self.cards.known_pick(
                EKCards.FIRST_PLAYER_IDX + player,
                EKCards.DISCARD_PILE_IDX, EKCardTypes.NOPE)
            self.action_noped = not self.action_noped
            self.nope_player = -1
            self.nope_player = self.get_next_nope_idx()

            # Lets not allow player to nope own nope 😅:
            if self.nope_player == player:
                self.nope_player = self.get_next_nope_idx()

            return

        # Player is the target of a PLAY_FAVOR:
        if not self.cur_player_is_major and action[EKActionVecDefs.PLAY_FAVOR] == 1:
            card = int(action[EKActionVecDefs.TARGET_CARD])
            frm = EKCards.FIRST_PLAYER_IDX + player
            to = int(EKCards.FIRST_PLAYER_IDX +
                self.unprocessed_action[EKActionVecDefs.PLAYER])
            self.cards.known_pick(frm, to, card)
            self.unprocessed_action = np.zeros(0)
            return

        # For all other actions we must give everyone chance to NOPE:
        self.unprocessed_action = action
        self.action_noped = False
        self.nope_player = -1
        self.nope_player = self.get_next_nope_idx()

        self.play_cards(player, action)

    def get_cur_player(self) -> int:
        """Returns the player who should take an action next"""
        self.cur_player_is_major = False
        # First give everyone chance to nope, then, if favor card was played,
        # let the target of that card respond, and finally let major player play
        if self.nope_player != -1:
            return self.nope_player
        if len(self.unprocessed_action) != 0 and \
                self.unprocessed_action[EKActionVecDefs.PLAY_FAVOR] == 1: # FIXME: sometimes you get here while player is not playing...
            return int(self.unprocessed_action[EKActionVecDefs.POINTER])
        self.cur_player_is_major = True
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
        Initializes `self.legal_actions` and `self.legal_actions_long` with the
        actions that are legal to take for `player`
        """

        cards = self.cards.cards[EKCards.FIRST_PLAYER_IDX + player]

        # Precondition for most actions is that we are the major player and that
        # there are no exploding kittens in the own deck:
        is_major = self.cur_player_is_major
        major_and_no_ek = is_major and cards[EKCardTypes.EXPL_KITTEN] == 0

        # Finding the cat card (CAT_A, to CAT_E) we have the most of:
        cats_mask = np.zeros_like(cards)
        cats_mask[EKCardTypes.CAT_A :] = 1
        max_cats = np.max(cards * cats_mask)
        
        # For each action see if we have the required card if any:
        legal_mask = (self.all_actions_cards == -1) | (cards[self.all_actions_cards] > 0)

        # For cat cards we need 2 or 3 instead of 1:
        legal_mask[9:61] = (max_cats >= 3)
        legal_mask[[9, 22, 35, 48]] = (max_cats >= 2)

        # For most actions we must be the major player without expl_kit:
        legal_mask[0:61] = legal_mask[0:61] & major_and_no_ek

        # For passing we can also be nope player:
        legal_mask[0] = legal_mask[0] | (player == self.nope_player)

        # For defusing we must also have exploding kitten:
        legal_mask[61:69] = legal_mask[61:69] & is_major & (not major_and_no_ek)

        # only nope player can play nope...:
        legal_mask[69] = legal_mask[69] & (player == self.nope_player)

        # And for giving favor we cant be major nor nope player:
        legal_mask[70:] = legal_mask[70:] & (not is_major) & (player != self.nope_player)

        # Attack has some extra constraints:
        legal_mask[1] = legal_mask[1] & \
            (self.attack_activated == False) & (self.attack_count < 3)
        
        # for FAVOR, TWO_CATS and THREE_CATS, we must check if the victim is
        # (still) playing, and has at least 1 card:
        ranges = np.r_[2:6, 9:61]
        global_players = np.mod(self.all_actions_pointers[ranges] - 1 + player,
                                self.num_players)
        legal_mask[ranges] = legal_mask[ranges] & \
            (self.all_actions_pointers[ranges] < self.num_players) & \
            self.still_playing[global_players] & \
            (np.sum(self.cards.cards[EKCards.FIRST_PLAYER_IDX + \
                global_players], axis = 1) > 0)

        # Exploding kitten cannot be placed beyond length of deck:
        legal_mask[61:69] = legal_mask[61:69] & \
            (self.all_actions_pointers[61:69] <=
                np.sum(self.cards.cards[EKCards.DECK_IDX]))
        
        # For giving favor, we must ourself posses the card we want to give:
        legal_mask[70:] = legal_mask[70:] & \
            (cards[self.all_actions_target_cards[70:]] > 0)

        self.legal_actions = self.all_actions[legal_mask]
        self.legal_actions_long = legal_mask.astype(np.int64)
    
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
    
    def player_centric_to_global(self, player: int, action: np.ndarray):
        """ Players view everything player centric: they don't know their own
        index, but instead think of themselves as index 1, the player before as
        index 0, and the players after as index 1 to N. This function converts
        actions players take back into a global point of view.
        """
        action = action.copy()

        # Masking everything other players did that you do not know:
        other_mask = action[:, EKActionVecDefs.PLAYER] != player
        defuse_mask = action[:, EKActionVecDefs.DEFUSE_KITTEN] == 1
        favor_mask = action[:, EKActionVecDefs.PLAY_FAVOR] == 1
        action[other_mask & defuse_mask, EKActionVecDefs.POINTER] = 0
        action[other_mask & favor_mask, EKActionVecDefs.TARGET_CARD] = 0

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
        future_mask = (
            (self.action_history[:, EKActionVecDefs.FUTURE_1] == 0) &
            (self.action_history[:, EKActionVecDefs.FUTURE_2] == 0) &
            (self.action_history[:, EKActionVecDefs.FUTURE_3] == 0)) | \
        (self.action_history[:, EKActionVecDefs.PLAYER] == player)

        history = self.action_history[future_mask].copy()
        history[:, EKActionVecDefs.PLAYER] = \
            np.mod(history[:, EKActionVecDefs.PLAYER] - player + 1, self.num_players)
        
        # Also do the same for actions that take a pointer argument:
        haspointer_mask = (history[:, EKActionVecDefs.PLAY_FAVOR] == 1) | \
                (history[:, EKActionVecDefs.PLAY_TWO_CATS] == 1) | \
                (history[:, EKActionVecDefs.PLAY_THREE_CATS] == 1)
        history[haspointer_mask, EKActionVecDefs.POINTER] = \
            np.mod(history[haspointer_mask, EKActionVecDefs.POINTER] -
                   player + 1, self.num_players)
        return history

    def push_action(self, action: np.ndarray):
        """ Push a taken action on the action_history array, and make sure size 
        limit is not exceeded. """
        if len(self.action_history) < EKGame.ACTION_HORIZON:
            self.action_history = np.insert(self.action_history, 0, action, axis = 0)
            return

        self.action_history = np.roll(self.action_history, 1, axis = 0)
        self.action_history[0] = action

    def process_action(self, player: int):
        """ Process the `self.unprocessed_action` """
        # If we are not the major player, we ignore:
        if not self.cur_player_is_major:
            return
        
        # There is no action to process:
        if len(self.unprocessed_action) == 0 or self.action_noped:
            self.unprocessed_action = np.zeros(0)
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
            future = self.cards.see_future(player)
            
            # Via the action history we communicate future to player:
            acvec = np.zeros(EKActionVecDefs.VEC_LEN)
            acvec[EKActionVecDefs.PLAYER] = player
            acvec[EKActionVecDefs.FUTURE_1:] = future
            self.push_action(acvec)
        
        # Two cats: Take a random card from ohter player:
        elif ac[EKActionVecDefs.PLAY_TWO_CATS] == 1:
            frm = int(ac[EKActionVecDefs.POINTER] + EKCards.FIRST_PLAYER_IDX)
            to = player + EKCards.FIRST_PLAYER_IDX

            # Check if frm still has card: maybe played nope inbetween:
            if np.sum(self.cards.cards[frm]) > 0:
                self.cards.random_pick(frm, to)
        
        # Three cats: take a card of your choosing from other (if it has it):
        elif ac[EKActionVecDefs.PLAY_THREE_CATS] == 1:
            frm = int(ac[EKActionVecDefs.POINTER] + EKCards.FIRST_PLAYER_IDX)
            to = player + EKCards.FIRST_PLAYER_IDX
            card = int(ac[EKActionVecDefs.TARGET_CARD])
            if self.cards.cards[int(frm), int(card)] > 0:
                self.cards.known_pick(frm, to, card, True)

        self.unprocessed_action = np.zeros(0)
        

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
            if idx >= self.num_players:
                return -1
            if self.still_playing[idx] and \
                self.cards.cards[EKCards.FIRST_PLAYER_IDX + idx, EKCardTypes.NOPE] > 0:
                return idx