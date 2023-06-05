import unittest
from ekenv import EKGame, EKActionVecDefs, EKCards, EKCardTypes
import numpy as np
import random

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
                    short_from_long = g.all_actions[idx]
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
    
    def test_attack(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.deck_ordered[:4] = EKCardTypes.CAT_A


        player_order = [0, 1, 1, 2]
        num_actions = [2, 1, 1, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
   
            self.assertEqual(p_idx, player,
                "First player 0 is, then player 1 has two play two times because is being attacked. Only after attack complete will player 2 start playing")
            self.assertEqual(num_acts, len(actions),
                "Player 0 has two options: play attack or draw card. All others can only draw card")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.CAT_A], 2,
            "Player 1 should have drawn both the CAT_As")
    
    def test_attack_and_skip(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.SKIP)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.deck_ordered[:4] = EKCardTypes.CAT_A


        player_order = [0, 1, 1, 2]
        num_actions = [2, 2, 1, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
   
            self.assertEqual(p_idx, player,
                "First player 0 is, then player 1 has two play two times because is being attacked. Only after attack complete will player 2 start playing")
            self.assertEqual(num_acts, len(actions),
                "Player 0 has two options: play attack or draw card. Player 1 has two options as well: play skip or draw card. After that all just 1 option")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.CAT_A], 1,
            "Player 1 should have drawn 1 CAT_A because he skipped on first one")
        
    def test_double_attack(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.deck_ordered[:4] = EKCardTypes.CAT_A


        player_order = [0, 1, 2, 2, 2, 2, 3]
        num_actions = [2, 2, 1, 1, 1, 1, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
   
            
            self.assertEqual(p_idx, player,
                "First player 0 and 1 are, then player 2 has to draw 4 cards, and only then player 3 is")
            self.assertEqual(num_acts, len(actions),
                "Player 0 and 1 have two options: play attack or draw card. Player 3 can draw cards")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.CAT_A], 4,
            "Player 2 should have drawn all four CAT_As. He could not play them because there was no one left to take anything from")
    
    def test_double_attack_with_possible_action(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 4, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.deck_ordered[:4] = EKCardTypes.CAT_A


        player_order = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4]
        num_actions = [2, 2, 1, 1, 2, 1, 2, 1, 1, 0, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
   
            
            self.assertEqual(p_idx, player,
                "Player 2 now plays one more, because after two there is someone to take a card from. Player 3 plays 3 because draws exploding kitten")
            self.assertEqual(num_acts, len(actions),
                "Player 2 will now have two options after having gotten two CAT_A cards, and again two options at the final because he has the attack from player 4")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.CAT_A], 1,
            "Player 2 has just 1 CAT_A instead of 4, because he played two and played an attack instead of drawin the last 1")

    def test_tripple_attack_not_possible(self):
        """ Giving player 2 an attack as well should change nothing, as you can only double it, not tripple it """
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.ATTACK)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_B)
        g.cards.deck_ordered[:4] = EKCardTypes.CAT_A
        g.cards.deck_ordered[4] = EKCardTypes.CAT_B


        player_order = [0, 1, 2, 2, 2, 2, 3, 3, 4]
        num_actions = [2, 2, 1, 1, 1, 2, 1, 1, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)

            self.assertEqual(p_idx, player,
                "First player 0 and 1 are, then player 2 has to draw 3 cards, but plays an attack on the thrid turn, meaning that player 4 has to draw two cards")
            self.assertEqual(num_acts, len(actions),
                "Player 0 and 1 have two options: play attack or draw card. Player 3 can draw cards only, untill last, when he can also play attack")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.CAT_A], 3,
            "Player 2 should have drawn 3 CAT_As now, because played attack on last")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 1,
            "Player 3 should have drawn the last CAT_A")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_B], 1,
            "And also the CAT_B")
    
    
    def test_favor(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.FAVOR)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.FAVOR)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.FAVOR)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_B)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_C)

        player_order = [0, 3, 0, 3, 0, 3, 0]
        num_actions = [2, 3, 2, 2, 2, 1, 1]

        for p_idx, num_acts in zip(player_order, num_actions):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)

            self.assertEqual(p_idx, player,
                "We must go back and forward between player 0, that plays a favor, and player 3, who is the target of the favor")
            self.assertEqual(num_acts, len(actions),
                "The number of actions must be mostly 2 for player 0, namely pass or play FAVOR card, while for player 3 it must be the number of cards that can be given away: 3, then 2, then 1")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 1,
            "Player 0 now must have taken all the cat cards from 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_B], 1,
            "Player 0 now must have taken all the cat cards from 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_C], 1,
            "Player 0 now must have taken all the cat cards from 3")
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "Meaning that player 3 does not have them anymore")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_B], 0,
            "Meaning that player 3 does not have them anymore")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_C], 0,
            "Meaning that player 3 does not have them anymore")
    
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
    
    def test_nope_nope_nope(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SHUFFLE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.NOPE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.NOPE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.NOPE)

        first3 = np.array([0, 0, 0])
        g.cards.deck_ordered[0:3] = first3.copy()

        should_see = [0, 2, 3, 2, 0, 0]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "We should see player 0 first, then player two because has nope, and then player 0 again to draw card and then to die")
            self.assertTrue(not np.all(g.cards.deck_ordered[0:3] == -1), "Deck should never get shuffled, as action got noped!")
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
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
    
    def test_skip(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.SKIP)

        should_see = [0, 0, 1, 1, 2, 3, 3, 4, 4, 2, 2]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "Player two should survive an extra round because of the skip")
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
    def test_skip_noped(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.SKIP)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.NOPE)

        should_see = [0, 0, 1, 1, 2, 3, 2, 2, 3, 3, 4, 4]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "Now the skip got noped so player 2 directly dies anyway :-)")
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
    def test_skip_not_noped_because_only_player_with_nope_is_dead(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2, EKCardTypes.SKIP)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.NOPE)

        should_see = [0, 0, 1, 1, 2, 3, 3, 4, 4, 2, 2]
        for idx in should_see:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            self.assertEqual(idx, player,
                "No one should nope, becauese player 0 is the only one with nope and is dead already")
            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])

    def test_see_future_simple(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 3)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SEE_FUTURE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)

        while np.sum(g.still_playing) > 1:
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            
            self.assertTrue((player == 0) == np.any(history[:, EKActionVecDefs.FUTURE_1:] == EKCardTypes.CAT_A) or (len(history) == 0),
                "We should only see a future show up in the history IFF we are player zero")

            if len(actions) == 0:
                continue
            g.take_action(player, actions[-1])
    
    def test_see_future_two_cards(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 3)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.SEE_FUTURE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.SEE_FUTURE)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.DECK_IDX, EKCardTypes.CAT_A)
        g.cards.deck_ordered[0] = EKCardTypes.CAT_A

        # Player one makes a move: first sees future, then draws cat A:
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 0, "Player 0 starts ofc")
        g.take_action(player, actions[-1])
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 0, "After playing future card, player one should get chance to draw card")
        future = history[0, EKActionVecDefs.FUTURE_1:]
        self.assertTrue(np.all(future == g.cards.deck_ordered[:3]),
            "We should perceive a future that is also going to come true")
        g.take_action(player, actions[-1])
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(player, 1, "Player 0 drew CAT_A card, so now its player 1s turn")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 1,
            "Player zero should have drawn a CAT_A card")
        g.take_action(player, actions[-1])
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        future2 = history[0, EKActionVecDefs.FUTURE_1:]
        self.assertTrue(np.all(future[1:] == future2[:-1]),
            "Player 0 and Player 1 their futures should overlap")
        self.assertEqual(player, 1, "Player 1 still has its turn: did future")
        g.take_action(player, actions[-1])
        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 1, int(future2[0])], 1,
            "Whatever player 2 saw in its future, will be what he now drew")

    def test_no_one_to_take_cards_from(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.FAVOR)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)

        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(len(actions), 1,
            "We have many cards to take stuff from others, but since none of the others have cards, we cant do anything")

        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.FAVOR)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.random_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 1)
        g.cards.random_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 2)
        g.cards.random_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3)

        [player, reward, cards, history, actions] = g.update_and_get_state(False)
        self.assertEqual(len(actions), 43,
            "But now that all the others do have cards, we can go wild!")
    
    def test_play_two_cats(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE)

        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 2,
            "At the start player 0 has two cats")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "But player 3 has none")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 0,
            "Moreover, Player 0 has no defuse")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE], 1,
            "But Player 3 has one defuse")

        for idx in range(3):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            
            self.assertEqual(player, 0, "Only player 0 should be seen: first uses two cats, then draws card, than defuses")

            if idx == 2:
                break
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 0,
            "At the end also player zero has no cats")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "As does player 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 1,
            "However, player zero now took the defuse of player 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE], 0,
            "Meaning player 3 does not have it anymore")
    
    def three_cats_card_that_is_there(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_E)

        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 3,
            "At the start player 0 has three cats")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "But player 3 has none")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_E], 0,
            "Moreover, Player 0 has no cat_e")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_E], 1,
            "But Player 3 does")

        for idx in range(3):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            
            self.assertEqual(player, 0, "Only player 0 should be seen: first uses two cats, then draws card, than defuses")

            if idx == 2:
                break
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 0,
            "At the end also player zero has no cat_a")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "As does player 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_E], 1,
            "However, player zero now took the cat_e of player 3")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_E], 0,
            "Meaning player 3 does not have it anymore")
    
    def three_cats_card_that_is_not_there(self):
        g = EKGame()
        EKGameTests.reset_game_to_all_on_discard_pile(g, 5)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A)
        g.cards.known_pick(EKCards.DISCARD_PILE_IDX, EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE)

        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 3,
            "At the start player 0 has three cats")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "But player 3 has none")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 0,
            "Moreover, Player 0 has no defuse")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE], 1,
            "But Player 3 does")

        for idx in range(3):
            [player, reward, cards, history, actions] = g.update_and_get_state(False)
            
            self.assertEqual(player, 0, "Only player 0 should be seen: first uses two cats, then draws card, than defuses")

            if idx == 2:
                break
            g.take_action(player, actions[-1])
        
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 0,
            "At the end also player zero has no cat_a as these were played")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.CAT_A], 0,
            "Player 3 still does not have one either")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 0,
            "Player 0 tried to take cat_e from player 3, which was not there, so player 0 did not get the defuse")
        self.assertEqual(g.cards.cards[EKCards.FIRST_PLAYER_IDX + 3, EKCardTypes.DEFUSE], 1,
            "Meaning player 3 still has it :-)")
    
    def test_the_big_one(self):
        """ Just letting it play a number of games with only random agents """
        g = EKGame()
        for det_seed in range(1996, 2100):
            # print(f"({det_seed})", end="", flush=True)
            # np.random.seed(det_seed)
            # random.seed(det_seed + 1) # +1 so not same if they happen to use same generator :-p

            num_players = np.random.randint(2, 6)
            g.reset(num_players)

            still_playing = np.sum(g.still_playing)

            while np.sum(g.still_playing) > 1:
                [player, reward, cards, history, actions] = g.update_and_get_state(False)
                
                new_still_playing = np.sum(g.still_playing)
                self.assertTrue(new_still_playing <= still_playing,
                    "Number of players can only monotonically decrease")
                still_playing = new_still_playing

                self.assertTrue(np.all(
                    cards[:, 0] >= np.sum(cards[:, 1:], axis=1)
                ), f"You can never think someone has more cards than he/she actually has.")

                self.assertTrue(reward >= -1 and reward <= 1,
                    "Rewards are only ever between -1 and +1")

                if len(actions) == 0:
                    self.assertTrue(g.still_playing[player] == False,
                        "You can only get no actions when you are out of the game.")
                    continue

                g.take_action(player, actions[np.random.randint(len(actions))])
    
    def test_the_big_one_long_form(self):
        """ Same as above but now with long form action selection """
        g = EKGame()
        for det_seed in range(1996, 2100):
            # print(f"({det_seed})", end="", flush=True)
            # np.random.seed(det_seed)
            # random.seed(det_seed + 1) # +1 so not same if they happen to use same generator :-p

            num_players = np.random.randint(2, 6)
            g.reset(num_players)

            still_playing = np.sum(g.still_playing)

            while np.sum(g.still_playing) > 1:
                [player, reward, cards, history, actions] = g.update_and_get_state(False)

                player_dead = len(actions) == 0
                actions = g.get_legal_actions(True)
                
                new_still_playing = np.sum(g.still_playing)
                self.assertTrue(new_still_playing <= still_playing,
                    "Number of players can only monotonically decrease")
                still_playing = new_still_playing

                self.assertTrue(np.all(
                    cards[:, 0] >= np.sum(cards[:, 1:], axis=1)
                ), f"You can never think someone has more cards than he/she actually has.")

                self.assertTrue(reward >= -1 and reward <= 1,
                    "Rewards are only ever between -1 and +1")

                if player_dead:
                    self.assertTrue(g.still_playing[player] == False,
                        "You can only get no actions when you are out of the game.")
                    continue

                options = np.where(actions == 1)[0]
                pick = random.choice(options).item()

                g.take_action(player, pick)


if __name__ == "__main__":
    unittest.main(failfast=True) # argv=["ignore", "EKGameTests.test_the_big_one"]