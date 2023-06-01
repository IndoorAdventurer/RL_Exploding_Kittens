import torch
from torch import nn
import math

class EKPreprocessNW(nn.Module):
    """
    Scales up input tokens using a project matrix, and adds position encodings.
    Optionally prepends a class token.
    """

    def __init__(self, max_len: int, in_dim: int, out_dim: int, cls_token: bool) -> None:
        """
        Constructor

        Args:
            max_len (int): maximum length of input sequence (`S` in shape
            specifications below).
            in_dim (int): dimension of a single input embedding, such that the
            input to this module has form: `B × S × in_dim`.
            out_dim (int): dimension of single output embedding, such taht the
            output of this module has form: `B × S × out_dim`.
            cls_token (bool): if `True`, a class token will be prepended to the
            sequence.
        """
        super().__init__()
        self.cls_token = cls_token

        self.affine_projection = nn.Linear(in_dim, out_dim)

        if cls_token:
            self.learned_token = nn.Linear(1, out_dim, False)

        # Position encodings as in Vaswani et al, roughly
        # (https://pytorch.org/tutorials/beginner/transformer_tutorial.html):
        pos_encs = torch.zeros([1, max_len, out_dim])
        pos = torch.arange(max_len).view(1, max_len, 1)
        div_term = torch.exp(torch.arange(0, out_dim, 2) *
            (-math.log(10000.0) / out_dim))
        pos_encs[0, :, 0::2] = torch.sin(pos * div_term)
        pos_encs[0, :, 1::2] = torch.cos(pos * div_term)
        pos_encs *= 0.5
        self.register_buffer("pos_encs", pos_encs)
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        
        # Project onto higher dimension:
        inp_shape = embeddings.shape
        embeddings = embeddings.view(-1, inp_shape[-1])
        embeddings = self.affine_projection(embeddings)
        embeddings = embeddings.view(inp_shape[0], inp_shape[1], embeddings.shape[-1])

        # Add position encodings:
        embeddings += self.pos_encs[:, :inp_shape[1]]

        # Prepend class token:
        if self.cls_token:
            tok = self.learned_token(torch.tensor([1.],
                device=embeddings.device).unsqueeze(0))
            tok = tok.unsqueeze(0)
            tok = tok.repeat(inp_shape[0], 1, 1)
            embeddings = torch.concatenate([tok, embeddings], 1)

            if mask != None:
                prepend = torch.ones([inp_shape[0], 1],
                    dtype=torch.float32, device=mask.device)
                mask = torch.concatenate([prepend, mask], 1)

            return embeddings, mask

        return embeddings
