import numpy as np
import torch
from ekenv import EKAgent, EKTrainer, EKRandomAgent
from ekmodels import EKAtomicTF
from random import shuffle, choice


class AtomicTFDQN(EKAgent):
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
                         True, include_probs)
        self.model = model
        self.buffer = buffer
        self.epsilon = epsilon
    
    def policy(self,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        
        # Random action with probability epsilon
        if np.random.random() < self.epsilon:
            options = np.where(legal_actions == 1)[0]
            return choice(options).item()
        
        self.model.eval()

        cards_norm = self.normalize_cards(cards)
        cards_norm = torch.tensor(cards_norm, device="cuda", dtype=torch.float32)
        cards_norm = cards_norm.unsqueeze(0)

        hist_norm = self.normalize_history(action_history)
        hist_norm = torch.tensor(hist_norm, device="cuda", dtype=torch.float32)
        hist_norm = hist_norm.unsqueeze(0)

        with torch.no_grad():
            q_vals = self.model(cards_norm, hist_norm, None, None) \
                .detach().cpu().numpy().squeeze()
        
        q_vals[legal_actions == 0] = -float("inf")

        pick = np.argmax(q_vals).item()

        assert(legal_actions[pick] == 1)

        return pick

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


if __name__ == "__main__":

    model = EKAtomicTF().to("cuda")
    agent = AtomicTFDQN(model, None, 0.9, False, False, True)
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