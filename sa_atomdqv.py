from ekenv import EKAgent, EKTrainer
from ekenv import EKRandomAgent
from ekmodels import EKTransformer
from ekutils import EKAtomDQVBuf
import numpy as np
import torch
import random
import argparse


# SINGLE AGENT: I.E. AGANST ONE RANDOM AGENT

class AtomDQVAgent(EKAgent):
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
                         True, False)
        
        # Hyperparameters:
        cards_dim = 14
        hist_len = 10
        self.init_epsilon = 0.5
        self.end_epsilon = 0.1
        self.epsilon_decay = 0.999
        self.epsilon = self.init_epsilon
        self.batch_size = batch_size
        
        # Models:
        self.q_net = EKTransformer(cards_dim, hist_len + 1, True, 82).to("cuda")
        self.v_net = EKTransformer(cards_dim, hist_len, False, 1).to("cuda")
        self.t_net = EKTransformer(cards_dim, hist_len, False, 1).to("cuda")

        # Other training stuff:
        self.rpbuf = EKAtomDQVBuf(rpbuf_len, 10)
        self.q_loss_func = torch.nn.MSELoss()
        self.v_loss_func = torch.nn.MSELoss()
        self.q_optim = torch.optim.RMSprop(self.q_net.parameters(), lr=1e-4)
        self.v_optim = torch.optim.RMSprop(self.v_net.parameters(), lr=1e-4)

        # Recording and printing data:
        self.loss_cnt = 0
        self.q_loss = .0
        self.v_loss = .0
    
    def policy(self,
            train: bool,
            cards: np.ndarray,
            action_history: np.ndarray,
            legal_actions: np.ndarray
    ) -> int | np.ndarray:
        
        # Random action with probability epsilon (only during training):
        if train and np.random.random() < self.epsilon:
            options = np.where(legal_actions == 1)[0]
            pick = random.choice(options).item()

            assert(legal_actions[pick] == 1)

            return pick

        self.q_net.eval()

        cards_norm = self.normalize_cards(cards)
        cards_norm = torch.tensor(cards_norm, device="cuda", dtype=torch.float32)
        cards_norm = cards_norm.unsqueeze(0)

        hist_norm = self.normalize_history(action_history)
        hist_norm = torch.tensor(hist_norm, device="cuda", dtype=torch.float32)
        hist_norm = hist_norm.unsqueeze(0)

        with torch.no_grad():
            q_vals = self.q_net(cards_norm, hist_norm, None, None) \
                .detach().cpu().numpy().squeeze()
        
        q_vals[legal_actions == 0] = -np.inf

        pick = np.argmax(q_vals).item()

        assert(legal_actions[pick] == 1)

        return pick

    def train_hook(self) -> None:
        if len(self.rpbuf.buf) < 10_000:
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

        # update v_net:
        v_preds = self.v_net(cards_t, history_t, c_t_mask, h_t_mask).squeeze(1)
        v_loss = self.v_loss_func(v_preds, targets)
        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        # update q_net:
        q_preds = self.q_net(cards_t, history_t, c_t_mask, h_t_mask)

        q_preds = torch.gather(q_preds, 1, action.unsqueeze(0)).squeeze()

        # action = action.unsqueeze(1)
        # history_t = torch.concatenate([action, history_t], 1)
        # h_t_mask = torch.concatenate([
        #     torch.ones([len(h_t_mask), 1], dtype=torch.bool, device="cuda"),
        #     h_t_mask], 1)
        # q_preds = self.q_net(cards_t, history_t, c_t_mask, h_t_mask).squeeze(1)
        q_loss = self.q_loss_func(q_preds, targets)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        self.loss_cnt += 1
        self.q_loss += (q_loss.item() - self.q_loss) / self.loss_cnt
        self.v_loss += (v_loss.item() - self.v_loss) / self.loss_cnt

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

        self.rpbuf.append(
            cards_t, action_history_t, action, reward, cards_tp1,
            action_history_tp1, np.sum(legal_actions_tp1) == 0
        )

    def update_target(self):
        """
        Every now and then we update the target model
        """
        self.t_net.load_state_dict(self.v_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = self.end_epsilon + (self.epsilon - self.end_epsilon) * \
            self.epsilon_decay
    
    def get_losses(self):
        l_q = self.q_loss
        l_v = self.v_loss
        self.loss_cnt = 0
        self.q_loss = .0
        self.v_loss = .0
        return l_q, l_v

if __name__ == "__main__":
    num_epochs = 100
    num_training_games = 30
    num_testing_games = 30

    parser = argparse.ArgumentParser(description="Train Exploding kittens model with DQV and non-atomic action representations")
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    train_agent = AtomDQVAgent(True, True, 100_000, 64)
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
    
    game_cnt = 0
    def end_of_game_hook(is_training: bool):
        global game_cnt
        if is_training:
            print("|", end="", flush=True)
            if len(train_agent.rpbuf.buf) >= 10_000:
                game_cnt += 1
                train_agent.update_epsilon()
                if game_cnt % 10 == 0:
                    print(f"epsilon: {train_agent.epsilon:.4}, buffer size: {len(train_agent.rpbuf.buf)}")
                if game_cnt % 90 == 0:
                    train_agent.update_target()
                    print("Target updated!")
        else:
            print("x", end="", flush=True)

    trainer = EKTrainer(get_agents_for_training, get_agents_for_testing, end_of_game_hook)

    # Filling up buffer first:
    train_agent.epsilon = 1.0
    while(len(train_agent.rpbuf.buf) < 10_000):
        trainer.training_loop(1)
    game_cnt = 0
    train_agent.epsilon = train_agent.init_epsilon

    out_file = open(args.out_file, "a")
    out_file.write("games, wins, q_loss, v_loss\n")
    
    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        results = trainer.testing_loop(num_testing_games)
        print(results[0])
        print(results[1])
        l_q, l_v = train_agent.get_losses()
        print(f"Q-Loss: {l_q} || V-Loss: {l_v}")
        
        out_file.write(f"{results[1][0, 1] + results[1][1, 0]}, {results[1][0, 1]}, {l_q}, {l_v}\n")
        out_file.flush()
    
    out_file.close()