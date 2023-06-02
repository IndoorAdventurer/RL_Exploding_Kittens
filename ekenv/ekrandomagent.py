from .ekagent import EKAgent
import numpy as np


class EKRandomAgent(EKAgent):
    """
    Agent that randomly picks an action from the list of possible ones
    """

    def __init__(self) -> None:
        """ Constructor. Creates agent that plays randomly """
        super().__init__(False, False, False, False)
    
    def policy(self,
            train: bool,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        """ Picks a random action from the `legal_actions` array """
        choice = np.random.randint(0, len(legal_actions))
        return legal_actions[choice]