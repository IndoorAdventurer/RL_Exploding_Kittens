from ekenv import EKAgent, EKTrainer
import numpy as np


class MyLearningAgent(EKAgent):
    """
    Please implement your own learning agent here. Look at the `EKAgent` class
    for instructions!
    """

    def __init__(self,
        call_train_hook: bool,
        call_record_hook: bool,
        long_form_actions: bool,
        include_probs: bool
    ) -> None:
        super().__init__(call_train_hook, call_record_hook,
                         long_form_actions, include_probs)
    
    def policy(self,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        pass

    def train_hook(self) -> None:
        pass

    def record_hook(self,
            cards_t: np.ndarray,
            action_history_t: np.ndarray,
            action: int | np.ndarray,
            reward: float,
            cards_tp1: np.ndarray,
            action_history_tp1: np.ndarray
    ) -> None:
        pass

def get_agents_for_training() -> list[EKAgent]:
    """ Should return a list of agents for training. This list should contain at
    least 2 agents, and at most 5 agents. """
    pass

def get_agents_for_testing() -> list[EKAgent]:
    """ Should return a list of agents for testing. This list should contain at
    least 2 agents, and at most 5 agents. """
    pass

if __name__ == "__main__":
    num_epochs = 100
    num_training_games = 100
    num_testing_games = 10

    trainer = EKTrainer(get_agents_for_training, get_agents_for_testing)

    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        results = trainer.testing_loop(num_testing_games)
        print(results)
        # Do stuff with the results here!