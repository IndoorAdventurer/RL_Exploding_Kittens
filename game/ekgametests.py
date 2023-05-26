import unittest
from ekgame import EKGame, EKActionVecDefs
from ekcards import EKCards, EKCardTypes
import numpy as np
from pprint import pprint

class EKGameTests(unittest.TestCase):
    
    def test_init(self):
        g = EKGame()
        g.reset(3)
        [player, reward, cards, action_history, mask] = g.update_and_get_state(long_form=True)
        normal = g.get_legal_actions(False)
        self.assertEqual(player, 0,
            "Player zero always is the one that starts. Thats why its player zero.. you know :-p")
        self.assertEqual(len(action_history), 0, "There should not have been any actions taken yet.")
        # I decided against zero padding the actions vector, as the number of players is also not set, so we will have to pad vectors during training anyway

        self.assertEqual(np.sum(mask), len(normal))
        self.assertTrue(np.all(normal[:, 0] == 1),
            "Actions should also give 1 as index for player.")
    
    def test_player_centric(self):
        """ Conversions to and from player space should be each others inverse, and
        the player space should give the correct representation! """
        g = EKGame()

        for num_players in range(2, 6):
            g.num_players = num_players
            for major in range(num_players):
                for minor in range(num_players):
                    nope = np.array([major, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    attack = np.array([major, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    favor = np.array([major, minor, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    give_favor = np.array([major, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    two_cats = np.array([major, minor, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
                    three_cats = np.array([major, minor, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                    vanilla_cards = [nope, attack, give_favor]
                    cards_with_ptr = [favor, two_cats, three_cats]
                    for card in vanilla_cards:
                        g.action_history = np.array([card])
                        player_centric = g.player_centric_action_history(major)[0]
                        self.assertEqual(player_centric[0], 1, "Player should always be at index 1")
                        self.assertEqual(player_centric[1], 0, "For vanilla card pointer to 0")
                        global_ = g.player_centric_to_global(major, player_centric)
                        self.assertTrue(np.all(card == global_),
                            "player_centric_action_history and player_centric_to_global should kind of be each others inverses")
                    for card in cards_with_ptr:
                        g.action_history = np.array([card])
                        player_centric = g.player_centric_action_history(major)[0]
                        self.assertEqual(player_centric[0], 1, "Player should always be at index 1")
                        relative_minor = np.mod(minor - major + 1, g.num_players)
                        self.assertEqual(player_centric[1], relative_minor, "For all these cards, the ptr should be set to relative player index of minor")
                        global_ = g.player_centric_to_global(major, player_centric)
                        self.assertTrue(np.all(card == global_),
                            "player_centric_action_history and player_centric_to_global should kind of be each others inverses")
    
    def test_long_form_indeces(self):
        g = EKGame()
        for _ in range(10):
            g.reset(3)
            g.calc_legal_actions(0)
            short_form = g.get_legal_actions(False)
            long_form = g.get_legal_actions(True)
            short_idx = 0

            for idx in range(len(long_form)):
                if long_form[idx] == 1:
                    short_from_long = g.action_from_long_form(idx)
                    self.assertTrue(np.all(short_form[short_idx] == short_from_long),
                        "We should be able to get the short form action from an index into the long form array, if the value there is 1")
                    short_idx += 1
    
    def reset_game_to_all_on_discard_pile(game: EKGame, num_players: int):
        """ Resets the game to a state in which all cards except the exploding
        kittens are on the discard pile """
        game.reset(num_players)
        game.cards.cards = np.zeros_like(game.cards.cards)
        game.cards.cards[EKCards.DISCARD_PILE_IDX] = game.cards.total_deck
        game.cards.cards[EKCards.DISCARD_PILE_IDX, EKCardTypes.EXPL_KITTEN] = 0
        game.cards.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = game.cards.total_deck[EKCardTypes.EXPL_KITTEN]
        game.cards.known_map = np.zeros_like(game.cards.known_map)
        game.cards.known_map[:, EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = game.cards.total_deck[EKCardTypes.EXPL_KITTEN]
    
    def test_game_with_only_kittens(self):
        """ Testing a really simple game: 5 players and the only cards in the game are exploding kittens, so everyone dies immediately """
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        his_len = 0
        while np.sum(g.still_playing) > 1:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertTrue((len(actions) == 0) == (reward == -1),
                "The moment we receive a reward of -1, is the moment we should be signalled that we are out of the game by receiving no actions")
            self.assertTrue(len(actions) <= 1,
                "We can have at most 1 action we can take, namely, drawing a kitten from the deck")
            self.assertTrue((np.sum(g.still_playing == 0) * 0.2 == reward) or reward == -1,
                "In this one round we play, any reward must correspond to the number of players that died")
            self.assertTrue(len(history) >= his_len, "History can only grow")
            his_len = len(history)
            if len(actions) == 0:
                continue
            g.take_action(player, actions[0])
        
    def test_game_with_only_kittens_and_defuses(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)

        # Give everyone 1 defuse:
        for p_idx in range(5):
            g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + p_idx, EKCardTypes.DEFUSE)

        first_died = False
        goto_next_player = False
        predicted_player = 0
        while np.sum(g.still_playing) > 1:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            if len(actions) == 0:
                first_died = True
            self.assertTrue((not first_died) or (len(actions) <= 1),
                "If the first person has died, the defuses are all used, and we will at most be able to take 1 action (drawing a card)")
            self.assertEqual(predicted_player, player,
                "Every player should always get the round twice before we move to next player!")
            if goto_next_player:
                predicted_player = (predicted_player + 1) % 5
            goto_next_player = not goto_next_player
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
    def test_single_nope_round(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SHUFFLE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.NOPE)

        first3 = np.array([0, 0, 0])
        g.cards.deck_ordered[0:3] = first3.copy()

        should_see = [0, 2, 0, 0]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "We should see player 0 first, then player two because has nope, and then player 0 again to draw card and then to die")
            self.assertTrue(not np.all(g.cards.deck_ordered[0:3] == -1), "Deck should never get shuffled, as action got noped!")
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
    def test_nope_nope(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SHUFFLE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.NOPE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.NOPE)
        first3 = np.array([0, 0, 0])
        g.cards.deck_ordered[0:3] = first3.copy()

        should_see = [0, 2, 3, 0, 0]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "We should see player 0 first, then players 3 and 3, because they have a nope, and then player 0 again")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])

        self.assertTrue(np.all(g.cards.deck_ordered[0:3] == -1), "By the end cards should have been shuffled as the nope was noped!")
    
    def test_shuffle(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SHUFFLE)
        first3 = np.array([1,2,3])
        g.cards.deck_ordered[0:3] = first3.copy()

        # 1:
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 0, "First 3 times must all be player #0")
        self.assertEqual(actions[-1, EKActionVecDefs.PLAY_SHUFFLE], 1, "First time we must get the option to shuffle the deck")
        self.assertTrue(np.all(first3 == g.cards.deck_ordered[0:3]), "Before shuffling, we should know the order")
        g.take_action(player, actions[-1])

        # 2:
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 0, "First 3 times must all be player #0")
        self.assertEqual(len(actions), 1, "Second time we can only pass (i.e. draw card -- which in this case is an exploding kitten)")
        self.assertTrue(np.all(g.cards.deck_ordered[0:3] == -1), "After shuffling not anymore!")
        g.take_action(player, actions[-1])

        # 3:
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 0, "First 3 times must all be player #0")
        self.assertEqual(len(actions), 0, "Last time we are dead and cannot take any actions")


    # print(f"(player #{player} -> {reward:.2} points) || num actions: {len(actions)}, history length: {(len(history))}")


if __name__ == "__main__":
    unittest.main()