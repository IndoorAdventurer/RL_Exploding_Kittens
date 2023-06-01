import torch
from torch import nn
from .ekpreprocessnw import EKPreprocessNW


class EKAtomicTF(nn.Module):

    def __init__(self,
        cards_dim: int = 27,
        history_len: int = 11
    ) -> None:
        super().__init__()

        d_model = 256

        self.cards_preproc = EKPreprocessNW(7, cards_dim, d_model, True)
        self.history_preproc = EKPreprocessNW(history_len, 15, d_model, True)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, 82)

    def forward(self, cards, history, card_mask, history_mask):

        cards, card_mask = self.cards_preproc(cards, card_mask)
        history, history_mask = self.history_preproc(history, history_mask)

        out = self.transformer(
            cards,
            history,
            src_key_padding_mask=card_mask,
            tgt_key_padding_mask=history_mask
        )[:, 0]
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    
    nw = EKAtomicTF()

    cards = torch.ones([3, 7, 27])
    cards_mask = torch.ones([3, 7])
    cards_mask[:, 6:] = 0

    history = torch.ones([3, 10, 15])
    history_mask = torch.ones([3, 10])
    history_mask[:, 4:] = 0

    out = nw(cards, history, cards_mask, history_mask)

    print(out.shape)