import numpy as np
from typing import Callable
from .ekagent import EKAgent
from .ekgame import EKGame, EKActionVecDefs
from .appendprobs import append_probabilities


class EKTrainer:
    """
    Class to encapsulate the training testing process for the Exploding Kittens
    environment. Can run a testing and training loop, and makes sure the agents
    callback hooks get called at the appropriate time.
    """

    def __init__(self,
            get_train_agents_func: (Callable[[], list[EKAgent]]),
            get_test_agents_func: Callable[[], list[EKAgent]]
    ) -> None:
        """
        Constructor

        Args:
            get_train_agents_func (() -> list[EKAgent]): This function will get
            called at the start of every training episode (i.e. game), and
            should return a list of agents for training (between 2 and 5). Note
            that they don't need to all train, or be the same. Maybe you want to
            throw random agents in the mix or something :-p

            get_test_agents_func (() -> list[EKAgent]): This function will get
            called at the start of every testing episode, and should return a
            list of agents for testing. Also here minimum is 2 agents, maximum
            is 5.
        """
        self.game = EKGame()
        self.get_train_agents_func = get_train_agents_func
        self.get_test_agents_func = get_test_agents_func
    
    def training_loop(self, num_games: int):
        """
        Runs multiple games for training purposes. For agents that have their
        `call_train_hook` set to `True` it calls `train_hook` after each step.
        For agents that have their `call_record_hook` set to `True` it calls
        `record_hook` when a new state-transition n-tuple becomes available.
        """

        for _ in range(num_games):
            agents = self.get_train_agents_func()
            num_agents = len(agents)
            prev_cards = [None] * num_agents
            prev_history = [None] * num_agents
            prev_action = [None] * num_agents

            self.game.reset(num_agents)

            # We keep playing as long as at least two players are playing:
            while np.sum(self.game.still_playing) > 1:
                [idx, reward, cards, history, actions] = \
                    self.game.update_and_get_state(False)
                
                if agents[idx].include_probs:
                    cards = append_probabilities(
                        cards, self.game.cards.total_deck)

                if agents[idx].record and prev_cards[idx] != None:
                    agents[idx].record_hook(
                        prev_cards[idx],
                        prev_history[idx],
                        prev_action[idx],
                        reward,
                        cards,
                        history
                    )

                if agents[idx].train:
                    agents[idx].train_hook()

                # No legal actions marks game over for this player:
                if len(actions) == 0:
                    continue

                if agents[idx].long_form == True:
                    actions = self.game.get_legal_actions(True)
                
                action = agents[idx].policy(cards, history, actions)
                self.game.take_action(idx, action)

                prev_cards[idx] = cards
                prev_history[idx] = history
                prev_action[idx] = action


    def testing_loop(self, num_games: int) -> np.ndarray:
        """
        Runs multiple games for evaluation purposes. The agents selected for
        this are obtained via the `get_test_agents_func` function given to the
        `__init__` method.

        This function returns an `N × N` array, where `arr[x, y] = z` means that
        player `x` finished ahead of player `y`, `z` times.
        """
        
        # For testing we keep the same agents for all games:
        agents = self.get_test_agents_func()
        num_agents = len(agents)
        win_table = np.zeros([num_agents, num_agents], dtype=np.int64)

        for _ in range(num_games):
            self.game.reset(num_agents)

            # We keep playing as long as at least two players are playing:
            while np.sum(self.game.still_playing) > 1:
                [idx, _, cards, history, actions] = \
                    self.game.update_and_get_state(False)
                
                if agents[idx].include_probs:
                    cards = append_probabilities(
                        cards, self.game.cards.total_deck)

                # No legal actions marks game over for this player:
                if len(actions) == 0:
                    win_table[self.game.still_playing, idx] += 1
                    continue

                if agents[idx].long_form == True:
                    actions = self.game.get_legal_actions(True)
                
                action = agents[idx].policy(cards, history, actions)
                self.game.take_action(idx, action)

        return win_table
