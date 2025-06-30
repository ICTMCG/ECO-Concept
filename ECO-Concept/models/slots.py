from torch import nn
import torch
from entmax import normmax_bisect


class ConceptQuerySlotAttention(nn.Module):
    """
    We follow the implementation from
    "Concept-centric transformers: Enhancing model interpretability through object-centric concept learning within a shared global workspace."
    [https://arxiv.org/abs/2305.15775]
    We replace softmax with sparsity normalization to associate each input token with only a few concepts.
    """

    def __init__(
            self,
            slot_size,
            num_iterations=1,
            truncate='bi-level',
            epsilon=1e-8,
            drop_path=0.2,
    ):
        super().__init__()
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_iterations = num_iterations

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

    def forward(self, features, slots_init, mask):
        features = self.norm_feature(features)
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        _, C, _ = slots_init.shape
        slots = slots_init

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat_interleave(C, dim=1).cuda()

        for i in range(self.num_iterations):
            if i == self.num_iterations - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * scale

            # Mask padding
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask == 0, 0)

            attn = normmax_bisect(attn_logits, dim=1)

            if mask is not None:
                attn = attn.masked_fill(mask == 0, 0)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )

            slots = slots.reshape(B, -1, D)

        return slots, attn


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m
