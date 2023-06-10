import torch
from torch import nn
from .ekpreprocessnw import EKPreprocessNW


class EKTransformer(nn.Module):
    """
    A transformer-based neural network architecture that hopefully is well
    suited for solving the exploding kittens card game. The encoder takes the
    knowledge of what cards are where as input, while the decoder takes the
    action history as input.
    """

    def __init__(self,
        cards_dim: int = 27,
        history_len: int = 11,
        class_token: bool = False,
        out_dim: int = 1,
        nhead: int = 4,
        d_model: int = 128,
        dim_feedforward: int = 256
    ) -> None:
        """
        Constructor

        Args:
            cards_dim (int): size of input card embeddings. Is 14 for the
            vanllia representation, but 27 if you include probability
            calculations.
            history_len (int): the maximum number of embeddings to include in
            the history, which will be the input of the decoder. (Note that this
            excludes optional class tokens, which get prepended to this).
            class_token (bool): if `True`, a learned class token will be
            prepended to the input of the decoder.
            out_dim (int): the dimensionality of the final output of the model.
            nhead (int): number of attention heads for both the encoder and
            decoder blocks of the transformer.
            d_model (int): size of the embeddings for both encoder and decoder
            dim_feedforward (int): number of hidden layers for the feedforward
            layers inside the encoder and decoder blocks.
        """
        super().__init__()

        self.cards_preproc = EKPreprocessNW(7, cards_dim, d_model, False)
        self.history_preproc = EKPreprocessNW(history_len, 15, d_model, class_token)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, cards, history, card_mask, history_mask):

        cards = self.cards_preproc(cards)

        if self.history_preproc.cls_token:
            history, history_mask = self.history_preproc(history, history_mask)
        else:
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
    
    nw = EKTransformer()

    cards = torch.ones([3, 7, 27])
    cards_mask = torch.ones([3, 7])
    cards_mask[:, 6:] = 0

    history = torch.ones([3, 10, 15])
    history_mask = torch.ones([3, 10])
    history_mask[:, 4:] = 0

    out = nw(cards, history, cards_mask, history_mask)

    print(out.shape)