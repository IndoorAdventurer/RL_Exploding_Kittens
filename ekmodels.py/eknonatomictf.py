import torch
from torch import nn
from ekpreprocessnw import EKPreprocessNW


class EKNonAtomicTF(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        d_model = 256

        self.cards_preproc = EKPreprocessNW(7, 27, d_model, False)
        self.history_preproc = EKPreprocessNW(10, 15, d_model, False)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, cards, history, card_mask, history_mask):

        cards = self.cards_preproc(cards)
        history = self.history_preproc(history)

        out = self.transformer(
            cards,
            history,
            src_key_padding_mask=card_mask,
            tgt_key_padding_mask=history_mask
        )[:, 0]
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    
    nw = EKNonAtomicTF()

    cards = torch.ones([3, 7, 27])
    cards_mask = torch.ones([3, 7])
    cards_mask[:, 6:] = 0

    history = torch.ones([3, 10, 15])
    history_mask = torch.ones([3, 10])
    history_mask[:, 4:] = 0

    out = nw(cards, history, cards_mask, history_mask)

    print(out.shape)