import abc
import numpy as np
from ekgame import EKActionVecDefs


class EKAgent(abc.ABC):
    """
    An Abstract Base Class (ABC) for an agent in the Exploding Kittens
    environment. One can derive a class from this one in order to implement any
    sort of learning agent. Please carefully read all documentation below ;-D 
    """

    # Constants for normalization purposes:
    MAX_CARD_SUM = 52       # max cards you can ever see in 1 place
    MAX_NUM_CARDS_VAL = 6   # max cards of same type you can ever see in 1 place
    MAX_PLAYER_VAL = 4      # the max played idx value
    MAX_POINTER_VAL = 6     # max possible val for pointer in action vec
    MAX_CARD_VAL = 12       # max possible val for target_card

    def __init__(self,
                 call_train_hook: bool,
                 call_record_hook: bool,
                 long_form_actions: bool,
                 include_probs: bool
    ) -> None:
        """
        Constructor

        Args:
            call_train_hook: (bool): if `True`, the `train_hook` method will be
            called during training. This method should be used to update the
            trainable parameters of any learning algorithm used to solve this
            problem.
        
            call_record_hook (bool): if `True`, the `record_hook` method will be
            called during training, containing a single learning experience, to
            be put in something like an experience replay buffer, as input.

            long_form_actions: (bool): The `policy` method takes a
            `legal_actions` array as input. This input can take on two forms.
            If `long_form_actions` is `True`, it will be an 82-dimensional
            binary array, where 0 indicates an action is illegal, and 1 that it
            is legal. In this case, an index into this array (corresponding to a
            legal action, of course) should be returned by the `policy` method.
            If `long_form_actions` is `False`, however, it will be an `N × 15`
            array, where each of the `N` vectors represents an action. In that
            case, one of these should be returned by the `policy` method.

            include_probs (bool): if `True`, the `cards*` arguments in the
            methods below will include for each player `p` and card `c`, the
            probability that `p` has at least one `c`. (Of course, it will also
            include probabilities for the deck).

            PS: note that the non-long form action representations are identical
            to the ones provided in the `action_history` array argument of the
            `policy` method.
        """

        super().__init__()

        self.record = call_record_hook
        self.train = call_train_hook
        self.long_form = long_form_actions
        self.include_probs = include_probs
    
    abc.abstractmethod
    def policy(self,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        """
        Takes the current observable state (`cards` and `action_history`) and a
        description of legal actions (see documentation of constructor) as
        input, and outputs the next action that should be taken (either as int
        or array. Again, see documentation of constructor).
        """
        pass

    def train_hook(self) -> None:
        """
        Gets called when it is an appropriate time to update the trainable
        parameters of the learning algorithm (i.e. after every action). Note 
        that this only happens when `call_train_hook` was `True` in the call to
        `__init__`.
        """
        pass

    def record_hook(self,
            cards_t: np.ndarray,
            action_history_t: np.ndarray,
            action: int | np.ndarray,
            reward: float,
            cards_tp1: np.ndarray,
            action_history_tp1: np.ndarray
    ) -> None:
        """
        Gets called when there is a new state transition that can be put in, for
        example, an action replay buffer. `cards` and `action_history` together
        form the state (`_t` for current state, `_tp1` for next state (t+1)).
        `action` is the action that was taken by the agent (either an int or
        an array, depending on if `long_form_actions` was set to `True` in the
        `__init__` call). Finally, `reward` is the reward that was observed
        during this transition.
        """
        pass

    def normalize_cards(self, cards: np.ndarray) -> np.ndarray:
        """ Can be used within `record_hook` and `policy` to normalize the
        `cards` array to make them more ANN friendly :-) """
        cards[:, 0] /= EKAgent.MAX_CARD_SUM
        cards[:, 1:14] /= EKAgent.MAX_CARD_VAL
        return cards

    def normalize_history(self, history: np.ndarray) -> np.ndarray:
        """ Can be used within `record_hook` and `policy` to normalize the
        `history` array to make them more ANN friendly :-) """
        history[:, EKActionVecDefs.PLAYER] /= EKAgent.MAX_PLAYER_VAL
        history[:, EKActionVecDefs.POINTER] /= EKAgent.MAX_POINTER_VAL
        history[:, EKActionVecDefs.TARGET_CARD] /= EKAgent.MAX_CARD_VAL
        history[:,
            EKActionVecDefs.FUTURE_1,
            EKActionVecDefs.FUTURE_2,
            EKActionVecDefs.FUTURE_3
        ] /= EKAgent.MAX_PLAYER_VAL
        return history

    def normalize_legal_actions(self, legal_actions: np.ndarray) -> np.ndarray:
        """ Can be used within `record_hook` and `policy` to normalize the
        `legal_actions` array to make them more ANN friendly :-) It works both
        for single actions (record hook) as well as multiple (policy).
        
        **IMPORTANT!** Make sure you return an unnormalized action vector within
        the policy method! """
        actions = legal_actions.copy()
        single = len(actions.shape) == 1
        if single:
            actions = np.expand_dims(actions, 0)

        actions[:, EKActionVecDefs.PLAYER] /= EKAgent.MAX_PLAYER_VAL
        actions[:, EKActionVecDefs.POINTER] /= EKAgent.MAX_POINTER_VAL
        actions[:, EKActionVecDefs.TARGET_CARD] /= EKAgent.MAX_CARD_VAL
        actions[:,
            EKActionVecDefs.FUTURE_1,
            EKActionVecDefs.FUTURE_2,
            EKActionVecDefs.FUTURE_3
        ] /= EKAgent.MAX_PLAYER_VAL

        if single:
            actions = np.squeeze(actions, 0)

        return actions