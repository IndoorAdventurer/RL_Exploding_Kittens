from ekenv import EKAgent, EKTrainer
from ekenv import EKRandomAgent
from ekmodels import EKTransformer
from ekutils import EKNonAtomDQVBuf
import numpy as np
import torch
import random


class NonAtomDQVAgent(EKAgent):
    """
    Implements the DQV algorithm for a non-atomic action space
    """

    def __init__(self,
        call_train_hook: bool,
        call_record_hook: bool,
        rpbuf_len: int,
        batch_size: int
    ) -> None:
        super().__init__(call_train_hook, call_record_hook,
                         False, False)
        
        # Hyperparameters:
        cards_dim = 14
        hist_len = 11
        self.init_epsilon = 0.5
        self.end_epsilon = 0.1
        self.epsilon_decay = 0.99
        self.epsilon = self.init_epsilon
        self.batch_size = batch_size
        
        # Models:
        self.q_net = EKTransformer(cards_dim, hist_len, False, 1).to("cuda")
        self.v_net = EKTransformer(cards_dim, hist_len, False, 1).to("cuda")
        self.t_net = EKTransformer(cards_dim, hist_len, False, 1).to("cuda")

        # Other training stuff:
        self.rpbuf = EKNonAtomDQVBuf(rpbuf_len, 10)
    
    def policy(self,
            train: bool,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        
        # Random action with probability epsilon (only during training):
        if train and np.random.random() < self.epsilon:
            choice = np.random.randint(0, len(legal_actions))
            return legal_actions[choice]
        
        self.q_net.eval()

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

        # Predict Q-values and return action corresponding to max:
        with torch.no_grad():
            q_vals = self.q_net(cards_norm, hist_norm, None, None) \
                .cpu().detach().numpy().squeeze()
        
        choice = np.argmax(q_vals)

        return legal_actions[choice]

    def train_hook(self) -> None:
        if len(self.rpbuf.buf) < 500:
            return

        self.q_net.train()
        self.v_net.train()
        self.t_net.eval()

        [   # This makes me want to cry :'-(
            cards_t,
            history_t,
            action,
            reward,
            cards_tp1,
            history_tp1,
            c_t_mask,
            h_t_mask,
            c_tp1_mask,
            h_tp1_mask,
            is_terminal
        ] = self.rpbuf.random_batch(self.batch_size)

        # calculate targets (equal to reward in terminal state):
        with torch.no_grad():
            targets = self.t_net(
                cards_tp1, history_tp1, c_tp1_mask, h_tp1_mask).squeeze(1)
        targets[is_terminal] = 0
        targets += reward

        
        # print(targets.shape)
        if len(reward[is_terminal]) > 0: # and torch.any(reward > 0):
            print(reward[is_terminal])
            exit()

        # TODO update q_net

        # TODO update v_net

    def record_hook(self,
            cards_t: np.ndarray,
            action_history_t: np.ndarray,
            action: int | np.ndarray,
            reward: float,
            cards_tp1: np.ndarray,
            action_history_tp1: np.ndarray,
            legal_actions_tp1: np.ndarray
    ) -> None:
        cards_t = self.normalize_cards(cards_t)
        action_history_t = self.normalize_history(action_history_t)
        cards_tp1 = self.normalize_cards(cards_tp1)
        action_history_tp1 = self.normalize_history(action_history_tp1)
        action = self.normalize_legal_actions(np.expand_dims(action, 0)) \
            .squeeze(0)

        self.rpbuf.append(
            cards_t, action_history_t, action, reward, cards_tp1,
            action_history_tp1, len(legal_actions_tp1) == 0
        )

    def update_target(self):
        """
        Every now and then we update the target model
        """
        self.t_net.load_state_dict(self.v_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = self.end_epsilon + (self.epsilon - self.end_epsilon) * \
            self.epsilon_decay

if __name__ == "__main__":
    num_epochs = 100
    num_training_games = 30
    num_testing_games = 30

    train_agent = NonAtomDQVAgent(True, True, 100_000, 32)
    rando = EKRandomAgent()

    def get_agents_for_training() -> list[EKAgent]:
        """ Should return a list of agents for training. This list should contain at
        least 2 agents, and at most 5 agents. """
        al = [train_agent, rando]
        random.shuffle(al)
        return al

    def get_agents_for_testing() -> list[EKAgent]:
        """ Should return a list of agents for testing. This list should contain at
        least 2 agents, and at most 5 agents. """
        return [train_agent, rando]
    
    def end_of_game_hook(is_training: bool):
        if is_training:
            print("|", end="", flush=True)
        else:
            print("x", end="", flush=True)

    trainer = EKTrainer(get_agents_for_training, get_agents_for_testing, end_of_game_hook)

    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        results = trainer.testing_loop(num_testing_games)
        print(results[0])
        print(results[1])
        # Do stuff with the results here!