import unittest
from ekcards import EKCards, EKCardTypes
import numpy as np

class EKCardsTests(unittest.TestCase):

    def init_cards_empty(num_players: int) -> EKCards:
        cards = EKCards()
        if num_players < 2 or num_players > 5:
            raise RuntimeError("Number of players must be between 2 and 5")

        cards.num_players = num_players
        
        # Applying the rules for how many exploding kittens and defuses to put in:
        expl_kits = num_players - 1
        extra_defuses = 2 if num_players <= 3 else 6 - num_players
        total_defuses = num_players + extra_defuses

        # The total number of cards of each type in the game:
        cards.total_deck = np.array([
            expl_kits,      # EXPL_KITTEN
            total_defuses,  # DEFUSE
            4,              # ATTACK
            4,              # FAVOR
            5,              # NOPE
            4,              # SHUFFLE
            4,              # SKIP
            5,              # SEE_FUTURE

            4,              # CAT_A
            4,              # CAT_B
            4,              # CAT_C
            4,              # CAT_D
            4               # CAT_E
        ])
        
        # 2 more than number of players: 1 for deck and  1 for discard pile:
        num_stacks = num_players + 2
        cards.cards = np.zeros([num_stacks, EKCardTypes.NUM_TYPES])

        # "known_map[3, 4, 5] == 2" means that player 3 knows that stack 4 has 2
        # cards of type 5:
        cards.known_map = np.zeros([num_players, num_stacks, EKCardTypes.NUM_TYPES])

        # Sometimes it is known what cards are where in the deck. We must
        # account for this, by keeping a list of known cards. The following
        # array does this. -1 means unknown, otherwise we use values from the
        # Cards enum. Index 0 is the next card to be drawn. We keep track of
        # only the num_players amount of card at the top of the stack.
        cards.deck_ordered = -1 * np.ones(num_players)

        return cards
    
    def assertCardSumCorrect(self, cards: EKCards):
        sums = np.sum(cards.cards, axis = 0)
        self.assertTrue(
            np.all(sums == cards.total_deck),
            "Sum of cards not equal to total_deck: " + str(sums)
        )
    
    def assertExplodingKittensKnown(self, cards: EKCards):
        for idx in range(cards.num_players):
            known_to_player = cards.get_state(idx)

            # Rototate back for comparison:
            known_to_player[2:] = np.roll(known_to_player[2:], idx - 1, axis=0)

            self.assertTrue(np.all(known_to_player[:, EKCardTypes.EXPL_KITTEN + 1] == cards.cards[:, EKCardTypes.EXPL_KITTEN]),
                f"Everyone should always know where all exploding kittens are (\n{known_to_player[:, EKCardTypes.EXPL_KITTEN + 1]} vs \n{cards.cards[:, EKCardTypes.EXPL_KITTEN]})")
    
    def assertDecksEqual(self, a, b):
        self.assertTrue(np.all(a == b),
            f"The given decks are not equal: {a} vs {b}")
    
    def test_basic_card_picking(self):
        """Looking if the behavior of `cards.cards` is correct when using either
        `cards.known_pick` or `cards.random_pick`."""
        cards = EKCardsTests.init_cards_empty(2)
        cards.cards[EKCards.DECK_IDX] = cards.total_deck
        self.assertDecksEqual(cards.cards[EKCards.DECK_IDX], cards.total_deck)
        self.assertCardSumCorrect(cards)

        cards.known_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.NOPE)
        cards.known_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.NOPE)
        cards.known_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.NOPE)
        self.assertExplodingKittensKnown(cards)

        self.assertCardSumCorrect(cards)
        should_be = np.zeros(EKCardTypes.NUM_TYPES)
        should_be[EKCardTypes.NOPE] = 3
        self.assertDecksEqual(cards.cards[EKCards.FIRST_PLAYER_IDX], should_be)

        card_list = []
        card_list += [cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX + 1)]
        card_list += [cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX + 1)]
        card_list += [cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX + 1)]
        self.assertExplodingKittensKnown(cards)

        self.assertCardSumCorrect(cards)
        should_be = np.zeros(EKCardTypes.NUM_TYPES)
        for c in card_list:
            should_be[c] += 1
        self.assertDecksEqual(cards.cards[EKCards.FIRST_PLAYER_IDX + 1], should_be)

        self.assertEqual(np.sum(cards.cards[EKCards.FIRST_PLAYER_IDX + 1]), np.sum(cards.cards[EKCards.FIRST_PLAYER_IDX]),
            "Sums of decks are not equal!")

        threw = False
        for _ in range(10):
            try:
                cards.random_pick(EKCards.FIRST_PLAYER_IDX, EKCards.DECK_IDX)
                cards.random_pick(EKCards.FIRST_PLAYER_IDX + 1, EKCards.DECK_IDX)
                self.assertExplodingKittensKnown(cards)
            except:
                threw = True
                break
        
        self.assertExplodingKittensKnown(cards)
        self.assertTrue(threw, "Did not throw when deck was empty!")
        self.assertCardSumCorrect(cards)
        should_be = np.zeros(EKCardTypes.NUM_TYPES)
        self.assertDecksEqual(cards.cards[EKCards.DECK_IDX], cards.total_deck)
        self.assertDecksEqual(cards.cards[EKCards.FIRST_PLAYER_IDX], should_be)
        self.assertDecksEqual(cards.cards[EKCards.FIRST_PLAYER_IDX + 1], should_be)
    
    def test_reset_func(self):
        """Test if the `cards.reset` function properly initializes the game."""
        cards = EKCards()

        for num_players in range(2, 6):
            # Everything that has to do with cards:
            cards.reset(num_players)
            self.assertEqual(cards.cards.shape[0], num_players + 2,
                "There should be two more stacks than there are players")
            self.assertEqual(cards.cards.shape[1], EKCardTypes.NUM_TYPES,
                "For each deck there should be as many elements as there are card types in the game")

            self.assertCardSumCorrect(cards)
            self.assertEqual(cards.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN], cards.total_deck[EKCardTypes.EXPL_KITTEN],
                "All exploding kittens must be in the deck still after rese")
            self.assertTrue(np.all(cards.cards[EKCards.DISCARD_PILE_IDX:, EKCardTypes.EXPL_KITTEN] == 0),
                "Exploding kittens can't be anywhere other than in the discard pile")
            self.assertTrue(np.all(cards.cards[EKCards.FIRST_PLAYER_IDX:, EKCardTypes.DEFUSE] >= 1),
                "All players must have at least one defuse!")
            self.assertTrue(np.all(np.sum(cards.cards[EKCards.FIRST_PLAYER_IDX:, :], axis = 1) == 8),
                "All players must have exactly 8 cards")
            
            # Everything that has to do with the known map:
            self.assertExplodingKittensKnown(cards)

            self.assertTrue(np.all(cards.known_map[:, EKCards.FIRST_PLAYER_IDX:, EKCardTypes.DEFUSE] == 1),
                "At the start everyone should know that everyone has 1 defuse")

    def test_known_functioning(self):
        cards = EKCards()
        cards.reset(3)

        # Things that have to do with deck:
        card = EKCardTypes.EXPL_KITTEN
        while card == EKCardTypes.EXPL_KITTEN:
            card = cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX)
            self.assertExplodingKittensKnown(cards)

        # Things that have to do with other players:
        tmp_map = cards.known_map.copy()
        while True:
            card = cards.random_pick(EKCards.FIRST_PLAYER_IDX, EKCards.FIRST_PLAYER_IDX + 1)
            if card != EKCardTypes.DEFUSE:
                break

            cards.known_pick(EKCards.FIRST_PLAYER_IDX + 1, EKCards.FIRST_PLAYER_IDX, card)
            cards.known_map = tmp_map.copy()
        
        self.assertEqual(cards.known_map[0, EKCards.FIRST_PLAYER_IDX + 1, card], 1,
            "The from player should know that the to players has the card that was just taken")
        self.assertEqual(np.sum(cards.known_map[1:, EKCards.FIRST_PLAYER_IDX:, :]), 5,
            "The rest should know only that the other two have one defuse. In addition, the one who took the non defuse knows that from still has his/her")

        self.assertCardSumCorrect(cards)
        
        cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE] = 2
        cards.known_map[:, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE] = 2
        cards.known_map[:, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A] = 1
        cards.known_pick(EKCards.FIRST_PLAYER_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.DEFUSE)
        self.assertEqual(cards.known_map[2, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 1,
            "Player 2 should still know that player 0 has 1 defuse, because he/she knew he had 2 first")
        self.assertEqual(cards.known_map[1, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 1,
            "The same goes here for player 1, who took the defuse")
        self.assertEqual(cards.known_map[2, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 0,
            "Player 2 does no longer know if player 1 has a CAT_A, because 1 might have taken it.")
        self.assertEqual(cards.known_map[1, EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A], 1,
            "Player 1 does know, because he/she knows the defuse was taken and not CAT_A")

        self.assertExplodingKittensKnown(cards)
    
    def test_ordered_deck_and_see_future(self):
        """See if we deterministically draw from the deck when an order is known. Also test the see_future method for this in one go :-)"""
        
        cards = EKCards()
        cards.reset(5)

        future = cards.see_future(0)
        self.assertTrue(np.all(cards.known_map[0, EKCards.DECK_IDX, future] >= 1),
            "The player looking at the future should now know that these cards are in the deck")
        real_future = np.zeros_like(future)
        for idx in range(len(real_future)):
            real_future[idx] = cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX)

        self.assertTrue(np.all(future == real_future),
            "The future must come true!")
        
        self.assertTrue(np.all(cards.deck_ordered == -1),
            "The future should be cleared again")


        future = cards.see_future(0)
        cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX)
        cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX)
        future2 = cards.see_future(0)
        self.assertEqual(future[2], future2[0],
            "Calling see future while part of the future is known, should give same known values")
        
        self.assertEqual(future2[0], cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX))
        self.assertEqual(future2[1], cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX))
        self.assertEqual(future2[2], cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX))

        self.assertTrue(np.all(cards.deck_ordered == -1),
            "The future should be cleared again")

        cards.cards[EKCards.DECK_IDX, EKCardTypes.EXPL_KITTEN] = 2
        cards.deck_ordered[1] = EKCardTypes.EXPL_KITTEN
        cards.deck_ordered[2] = EKCardTypes.EXPL_KITTEN
        self.assertNotEqual(cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX), EKCardTypes.EXPL_KITTEN,
            "There are only 2 exploding kittens in the game, and they both have a fixed place, so this cant ever be another one!")
        self.assertEqual(cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX), EKCardTypes.EXPL_KITTEN,
            "We set the future, so we should get the future!")
        self.assertEqual(cards.random_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX), EKCardTypes.EXPL_KITTEN,
            "We set the future, so we should get the future!")
    
    def test_place_back_kitten(self):
        """ Test if the insert_kitten function works the way it should. """

        cards = EKCards()
        cards.reset(5)

        for idx in range(6):
            cards.deck_ordered = -1 * np.ones_like(cards.deck_ordered)
            cards.known_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.EXPL_KITTEN)
            self.assertExplodingKittensKnown(cards)
            self.assertCardSumCorrect(cards)
            cards.insert_kitten(0, idx)
            self.assertExplodingKittensKnown(cards)
            self.assertCardSumCorrect(cards)
            self.assertEqual(cards.deck_ordered[idx], EKCardTypes.EXPL_KITTEN,
                "We should find kitten where we place it")
            self.assertEqual(np.sum(cards.deck_ordered == -1), len(cards.deck_ordered) - 1,
                "And nothing else elsewhere")
        
        cards.deck_ordered = -1 * np.ones_like(cards.deck_ordered)
        cards.known_pick(EKCards.DECK_IDX, EKCards.FIRST_PLAYER_IDX, EKCardTypes.EXPL_KITTEN)
        cards.insert_kitten(0, -1)
        self.assertTrue(np.all(cards.deck_ordered == -1),
            "-1 as index should mean insert exploding kitten in random place in deck")
    
    def test_get_state(self):
        """ Tests if the get_state function its returned array is formatted like it should be. """

        cards = EKCards()

        for num_players in range(2, 6):
            # Everything that has to do with cards:
            cards.reset(num_players)
            
            # Randomly move two cards from deck to discard pile:
            cards.random_pick(EKCards.DECK_IDX, EKCards.DISCARD_PILE_IDX)
            cards.random_pick(EKCards.DECK_IDX, EKCards.DISCARD_PILE_IDX)

            for player_idx in range(0, num_players):
                state = cards.get_state(player_idx)
                self.assertEqual(state[EKCards.DISCARD_PILE_IDX, 0], 2,
                    "Must show that discard pile has 2 cards")
                self.assertTrue(np.all(state[EKCards.FIRST_PLAYER_IDX:, 0] == 8),
                    "Must show that all players have 8 cards")
                self.assertEqual(np.sum(state[EKCards.FIRST_PLAYER_IDX + 1]), 16,
                    "Players own cards should be visible, and put as second player (because first is player before him/her)")
                self.assertTrue(np.all(state[EKCards.FIRST_PLAYER_IDX:, 1 + EKCardTypes.DEFUSE] >= 1),
                    "It should know that all players still have their defuse.")
    
    def test_to_discard_pile_positive(self):
        cards = EKCards()
        cards.reset(5)

        cards.cards[EKCards.FIRST_PLAYER_IDX, EKCardTypes.CAT_A] = 2
        cards.known_pick(EKCards.FIRST_PLAYER_IDX, EKCards.DISCARD_PILE_IDX, EKCardTypes.CAT_A)
        self.assertTrue(np.all(cards.known_map[:, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE] == 1),
            "When placing a card on the discard pile, everyone should still know which cards you have that were already known")
        cards.known_pick(EKCards.FIRST_PLAYER_IDX, EKCards.FIRST_PLAYER_IDX + 1, EKCardTypes.CAT_A)
        self.assertTrue(np.all(cards.known_map[2:, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE] == 0),
            "But if someone else takes it, this is not the case.")
        self.assertEqual(cards.known_map[1, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE], 1,
            "Then only the player who took it knows")
    
    def test_to_discard_pile_negative(self):
        cards = EKCards()
        cards.reset(5)

        self.assertTrue(np.all(cards.known_map[:, EKCards.FIRST_PLAYER_IDX:, EKCardTypes.DEFUSE] == 1),
            "At the start, again, everyone know that everyone has a defuse")
        cards.known_pick(EKCards.FIRST_PLAYER_IDX, EKCards.DISCARD_PILE_IDX, EKCardTypes.DEFUSE)
        self.assertTrue(np.all(cards.known_map[:, EKCards.FIRST_PLAYER_IDX, EKCardTypes.DEFUSE] == 0),
            "But when someone uses their defuse, this is no longer the case")

if __name__ == "__main__":
    unittest.main()