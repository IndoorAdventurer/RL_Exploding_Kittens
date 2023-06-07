from collections import deque
import numpy as np
import random
import torch


class EKNonAtomDQVBuf:

    def __init__(self, max_len: int, history_length: int) -> None:
        """
        Constructor

        Args:
            max_len (int): Will keep at most this many of the most recent
            experiences stored.
            history_length (int): The history must always be this long. If it is
            shorter, it must pad it with zeros.
        """
        
        self.buf = deque([], max_len)
        self.cards_len = 7      # This is fixed: 5 players + deck + discard pile
        self.hist_len = history_length
    
    def append(self,
            cards_t: np.ndarray,
            history_t: np.ndarray,
            action: np.ndarray,
            reward: float,
            cards_tp1: np.ndarray,
            history_tp1: np.ndarray,
            is_terminal: bool
    ):
        cards_t, c_t_mask = self.zero_pad(cards_t, self.cards_len)
        history_t, h_t_mask = self.zero_pad(history_t, self.hist_len)

        cards_tp1, c_tp1_mask = self.zero_pad(cards_tp1, self.cards_len)
        history_tp1, h_tp1_mask = self.zero_pad(history_tp1, self.hist_len)

        self.buf.append((
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
        ))
    
    def random_batch(self, batch_size: int):
        sample = random.sample(self.buf, batch_size)
        return (
            torch.tensor(np.array([x[0] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[1] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[2] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[3] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[4] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[5] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[6] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[7] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[8] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[9] for x in sample]), dtype=torch.float32, device="cuda"),
            torch.tensor(np.array([x[10] for x in sample]), dtype=torch.bool, device="cuda"),
        )   # This just makes me want to cry... 😭
    
    def zero_pad(self, to_pad: np.ndarray, pad_len: int):
        """
        Zero-pads the to_pad array, and returns the array and a padding mask
        """

        cur_len = len(to_pad)
        mask = np.zeros(pad_len)
        mask[:cur_len] = 1

        padded = np.pad(to_pad, ((0, pad_len - cur_len), (0, 0)))

        return padded, mask