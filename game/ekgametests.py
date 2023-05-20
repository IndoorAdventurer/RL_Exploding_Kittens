import unittest
from ekgame import EKGame
from ekcards import EKCards, EKCardTypes
import numpy as np

class EKGameTests(unittest.TestCase):
    
    def test_init(self):
        g = EKGame()
        g.reset(3)
        [player, cards, actions] = g.get_state()
        self.assertEqual(player, 0,
            "Player zero always is the one that starts. Thats why its player zero.. you know :-p")
        self.assertEqual(len(actions), 0, "Thre should not have been any actions taken yet.")
        # I decided against zero padding the actions vector, as the number of players is also not set, so we will have to pad vectors during training anyway

        mask = g.get_possible_actions(long_form=True)
        normal = g.get_possible_actions(long_form=False)
        self.assertEqual(np.sum(mask), len(normal))
        self.assertTrue(np.all(normal[:, 0] == 1),
            "Actions should also give 1 as index for player.")
        print(mask.shape)


if __name__ == "__main__":
    unittest.main()