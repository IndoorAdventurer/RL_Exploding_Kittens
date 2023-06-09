import numpy as np
from random import shuffle
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
            get_test_agents_func: Callable[[], list[EKAgent]],
            game_over_hook: Callable[[bool], None] | None = None
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

            game_over_hook ((bool) -> None) | None: Optional hook function that
            gets called at the end of every game. Receives a bool that is `True`
            only in the training loop.
        """
        self.game = EKGame()
        self.get_train_agents_func = get_train_agents_func
        self.get_test_agents_func = get_test_agents_func
        self.game_over_hook = game_over_hook
    
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

            # We keep playing as long as at least two players are playing
            # We cap it at 250, as the theoretical maximum lies around this:
            for _ in range(250):
                if np.sum(self.game.still_playing) <= 1:
                    break
                
                [idx, reward, cards, history, actions] = \
                    self.game.update_and_get_state(False)
                
                player_dead = len(actions) == 0
                
                if agents[idx].include_probs:
                    cards = append_probabilities(
                        cards, self.game.cards.total_deck)

                if agents[idx].long_form == True:
                    actions = self.game.get_legal_actions(True)
                
                if agents[idx].record and prev_cards[idx] is not None:
                    if player_dead and agents[idx].long_form == True:
                        actions = np.zeros(82, dtype=np.int64)
                    
                    agents[idx].record_hook(
                        prev_cards[idx],
                        prev_history[idx],
                        prev_action[idx],
                        reward,
                        cards,
                        history,
                        actions
                    )

                if agents[idx].train:
                    agents[idx].train_hook()

                if player_dead:
                    continue
                
                action = agents[idx].policy(True, cards, history, actions)
                self.game.take_action(idx, action)

                prev_cards[idx] = cards
                prev_history[idx] = history
                prev_action[idx] = action
            
            # Make winner get terminal state still:
            if np.sum(self.game.still_playing) == 1:
                [idx, reward, cards, history, actions] = \
                    self.game.update_and_get_state(False)
                if agents[idx].include_probs:
                    cards = append_probabilities(
                        cards, self.game.cards.total_deck)
                if agents[idx].long_form:
                    actions = self.game.get_legal_actions(True)
                if agents[idx].record and prev_cards[idx] is not None:
                    agents[idx].record_hook(
                        prev_cards[idx],
                        prev_history[idx],
                        prev_action[idx],
                        reward,
                        cards,
                        history,
                        actions
                    )
            
            if self.game_over_hook != None:
                self.game_over_hook(True)


    def testing_loop(self, num_games: int):
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
        reward_list = np.zeros(num_agents)
        win_table = np.zeros([num_agents, num_agents], dtype=np.int64)

        # We will shuffle, so keep track of original indeces to invert later:
        indeces = np.arange(num_agents)

        for _ in range(num_games):
            self.game.reset(num_agents)

            # Shuffle:
            shuf = np.arange(num_agents)
            shuffle(shuf)
            agents = [agents[a_idx] for a_idx in shuf]
            reward_list = reward_list[shuf]
            win_table = win_table[:, shuf]
            win_table = win_table[shuf]
            indeces = indeces[shuf]

            # We keep playing as long as at least two players are playing
            # We cap it at 250, as the theoretical maximum lies around this:
            for _ in range(250):
                if np.sum(self.game.still_playing) <= 1:
                    break

                [idx, reward, cards, history, actions] = \
                    self.game.update_and_get_state(False)
                
                reward_list[idx] += reward
                
                if agents[idx].include_probs:
                    cards = append_probabilities(
                        cards, self.game.cards.total_deck)

                # No legal actions marks game over for this player:
                if len(actions) == 0:
                    win_table[self.game.still_playing, idx] += 1
                    continue

                if agents[idx].long_form == True:
                    actions = self.game.get_legal_actions(True)
                
                action = agents[idx].policy(False, cards, history, actions)
                self.game.take_action(idx, action)

            if self.game_over_hook != None:
                self.game_over_hook(False)
        
        # Invert all the shuffles:
        inverse = np.zeros_like(indeces)
        inverse[indeces] = np.arange(num_agents)
        reward_list = reward_list[inverse]
        win_table = win_table[:, inverse]
        win_table = win_table[inverse]

        return reward_list, win_table
