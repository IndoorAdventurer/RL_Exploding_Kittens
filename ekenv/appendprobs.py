import numpy as np

# Turning the comb function into one that works on vectors:
comb_vec = np.vectorize(np.math.comb)

def append_probabilities(stacks: np.ndarray, total_deck: np.ndarray) -> np.ndarray:
    """
    Uses the hypergeometric distribution to calculate probabilites of cards
    being in certain places.
    """

    num_stacks = len(stacks)
    num_cards = 13
    
    # For each type of card we know of how many we don't know where they are:
    known_total = total_deck - np.sum(stacks[:, 1:], axis=0)

    # We know how many cards are in what stack:
    stack_counts = stacks[:, 0]

    # Arguments for hypergeometric distribution:
    N = np.sum(known_total)
    K = known_total
    n = stack_counts - np.sum(stacks[:, 1:], axis=1)

    # Calculate hypergeometric distribution for each stack-card-combination:
    NminK = np.reshape((N - K), [1, -1])
    nmink = np.reshape(n, [-1, 1])
    
    if not np.all(nmink >= 0):
        print(n)
        print(stack_counts)
        print(stacks)
        print("", flush=True)

    NminKchoosenmink = comb_vec(NminK, nmink)
    Nchoosen = np.reshape(comb_vec(N, n), [-1, 1])
    answ = 1 - NminKchoosenmink / Nchoosen

    # Setting everything we do know to 1:
    answ = np.clip(answ  + (stacks[:, 1:] > 0), 0, 1)

    # Append probabilities to stack (containing factual info) and return:
    return np.append(stacks, answ, axis=1)