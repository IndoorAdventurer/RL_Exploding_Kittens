import unittest
from ekgame import EKGame
from ekcards import EKCards, EKCardTypes
import numpy as np

class EKGameTests(unittest.TestCase):
    
    def test_init(self):
        g = EKGame()
        g.reset(3)
        [player, cards, action_history, mask] = g.get_state(long_form=True)
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
                    nope = np.array([major, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                    attack = np.array([major, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                    favor = np.array([major, minor, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                    give_favor = np.array([major, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    two_cats = np.array([major, minor, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                    three_cats = np.array([major, minor, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
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


if __name__ == "__main__":
    unittest.main()