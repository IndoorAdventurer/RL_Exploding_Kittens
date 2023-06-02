import numpy as np
import torch
from ekenv import EKAgent, EKTrainer, EKRandomAgent
from ekmodels import EKNonAtomicTF
from random import shuffle


class NonAtomicTFDQN(EKAgent):
    """
    A Transformer-based DQN which uses a non-atomic description of the action
    space.
    """

    def __init__(self,
        model: torch.nn.Module,
        buffer,
        epsilon: float,
        call_train_hook: bool,
        call_record_hook: bool,
        include_probs: bool
    ) -> None:
        super().__init__(call_train_hook, call_record_hook,
                         False, include_probs)
        self.model = model
        self.buffer = buffer
        self.epsilon = epsilon
    
    def policy(self,
            train: bool,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        
        # Random action with probability epsilon
        if train and np.random.random() < self.epsilon:
            choice = np.random.randint(0, len(legal_actions))
            return legal_actions[choice]
        
        self.model.eval()

        cards_norm = self.normalize_cards(cards)
        hist_norm = self.normalize_history(action_history)
        acs_norm = self.normalize_legal_actions(legal_actions)

        # Creating a batch of all actions:
        cards_norm = np.expand_dims(cards_norm, 0)
        cards_norm = np.repeat(cards_norm, len(acs_norm), axis=0)
        cards_norm = torch.tensor(cards_norm, device="cuda", dtype=torch.float32)

        hist_norm = np.expand_dims(hist_norm, 0)
        hist_norm = np.repeat(hist_norm, len(acs_norm), axis=0)
        acs_norm = np.expand_dims(acs_norm, axis=1)
        hist_norm = np.concatenate([acs_norm, hist_norm], axis=1)
        hist_norm = torch.tensor(hist_norm, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            q_vals = self.model(cards_norm, hist_norm, None, None) \
                .cpu().detach().numpy().squeeze()
        
        choice = np.argmax(q_vals)

        return legal_actions[choice]


    def train_hook(self) -> None:
        pass

    def record_hook(self,
            cards_t: np.ndarray,
            action_history_t: np.ndarray,
            action: int | np.ndarray,
            reward: float,
            cards_tp1: np.ndarray,
            action_history_tp1: np.ndarray,
            legal_actions_tp1: np.ndarray
    ) -> None:
        pass


if __name__ == "__main__":

    model = EKNonAtomicTF().to("cuda")
    agent = NonAtomicTFDQN(model, None, 0.9, True, True, True)
    rando = EKRandomAgent()

    def train_agents() -> list[EKAgent]:
        """ Should return a list of agents for training. This list should contain at
        least 2 agents, and at most 5 agents. """
        al = [agent, rando]
        shuffle(al)
        return al

    def test_agents() -> list[EKAgent]:
        """ Should return a list of agents for testing. This list should contain at
        least 2 agents, and at most 5 agents. """
        return [agent, rando]
    
    def end_of_game(is_training: bool):
        if is_training:
            print("|", end="", flush=True)
            delta = agent.epsilon - 0.05
            agent.epsilon = 0.05 + 0.983 * delta
        else:
            print("x", end="", flush=True)

    num_epochs = 100
    num_training_games = 100
    num_testing_games = 10


    trainer = EKTrainer(train_agents, test_agents, end_of_game)

    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        results = trainer.testing_loop(num_testing_games)
        print(f"Epsilon: {agent.epsilon}")
        print(results)
        # Do stuff with the results here!