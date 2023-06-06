import numpy as np
import torch
from random import shuffle, choice
from ekenv import EKAgent, EKTrainer, EKRandomAgent
from ekmodels import EKTransformer
from ekutils import EKAtomicReplayBuffer


class AtomicTFDQN(EKAgent):
    """
    A Transformer-based DQN which uses an atomic description of the action
    space.
    """

    def __init__(self,
        model: torch.nn.Module,
        optimizer,
        buffer: EKAtomicReplayBuffer,
        batch_size: int,
        epsilon: float,
        call_train_hook: bool,
        call_record_hook: bool,
        include_probs: bool
    ) -> None:
        super().__init__(call_train_hook, call_record_hook,
                         True, include_probs)
        self.model = model
        self.target_model = EKTransformer(14, 10, True, 82).to("cuda")
        self.target_model.load_state_dict(model.state_dict())
        self.loss = torch.nn.MSELoss()
        self.optim = optimizer

        self.buffer = buffer
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.avg_loss = 0
        self.loss_cnt = 0

    
    def policy(self,
            train: bool,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        
        # Random action with probability epsilon
        if train and np.random.random() < self.epsilon:
            options = np.where(legal_actions == 1)[0]
            pick = choice(options).item()

            assert(legal_actions[pick] == 1)

            return pick

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
        
        if len(self.buffer.buf) < 10_000:
            return
        
        self.model.train()
        self.target_model.eval()
        
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
            legal_actions_tp1
        ] = self.buffer.random_batch(self.batch_size)

        # Get predictions of the model (only for the action that we took):
        predictions = self.model(cards_t, history_t, c_t_mask, h_t_mask)
        predictions = predictions.gather(1, action).squeeze()

        # Get targets for next state:
        with torch.no_grad():
            next_preds = self.target_model(
                cards_tp1, history_tp1, c_tp1_mask, h_tp1_mask)
        next_preds[legal_actions_tp1 == 0] = -torch.inf
        next_preds = next_preds.max(1)[0]
        next_preds[next_preds == -torch.inf] = 0

        targets = next_preds + reward

        # For terminal state only reward:
        terminal_states = legal_actions_tp1.sum(1) == 0
        targets[terminal_states] = reward[terminal_states]
        
        loss = self.loss(predictions, targets)
        self.loss_cnt += 1
        self.avg_loss += (loss.item() - self.avg_loss) / self.loss_cnt
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def record_hook(self,
            cards_t: np.ndarray,
            action_history_t: np.ndarray,
            action: int,
            reward: float,
            cards_tp1: np.ndarray,
            action_history_tp1: np.ndarray,
            legal_actions_tp1: np.ndarray
    ) -> None:
        
        cards_t = self.normalize_cards(cards_t)
        action_history_t = self.normalize_history(action_history_t)
        cards_tp1 = self.normalize_cards(cards_tp1)
        action_history_tp1 = self.normalize_history(action_history_tp1)

        self.buffer.append(
            cards_t, action_history_t, action, reward, cards_tp1,
            action_history_tp1, legal_actions_tp1
        )
    
    def update_target(self):
        """
        Every now and then we update the target model
        """
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == "__main__":

    model = EKTransformer(14, 10, True, 82).to("cuda")
    optim = torch.optim.RMSprop(model.parameters(), lr=5e-4)
    rpbuf = EKAtomicReplayBuffer(100_000, 10)
    agent = AtomicTFDQN(model, optim, rpbuf, 64, 0.5, True, True, False)
    rando = EKRandomAgent()
    init_fill = 10_000

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
    
    game_idx = 0
    
    def end_of_game(is_training: bool):
        if is_training:
            print("|", end="", flush=True)
            if len(rpbuf.buf) > init_fill:
                target_eps = 0.05
                delta = agent.epsilon - target_eps
                agent.epsilon = target_eps + 0.98 * delta

            global game_idx
            game_idx = (game_idx + 1) % 50
            if game_idx == 0:
                agent.update_target()
                print(f" Loss: {agent.avg_loss}")
                agent.avg_loss = 0
                agent.loss_cnt = 0

        else:
            print("x", end="", flush=True)

    num_epochs = 100
    num_training_games = 30
    num_testing_games = 30


    trainer = EKTrainer(train_agents, test_agents, end_of_game)

    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        if (len(rpbuf.buf) > init_fill):
            results = trainer.testing_loop(num_testing_games)
            print(f" Epsilon: {agent.epsilon}, buffer size: {len(rpbuf.buf)}")
            print(results[0])
            print(results[1])
            # Do stuff with the results here!