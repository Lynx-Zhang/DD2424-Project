import torch
import torch.nn as nn

class LoReFTIntervention(nn.Module):
    """
    Low-rank Representation Fine-tuning (LoReFT) Intervention.
    Formula: h' = h + R^T (W(Rh) + b - Rh)
    where R is a low-rank orthogonal matrix (typically fixed or learned),
    and W, b are learned parameters.
    """
    def __init__(self, embed_dim, low_rank_dimension=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.r = low_rank_dimension
        
        # Low-rank projection R (typically orthogonal)
        # We'll initialize it as a semi-orthogonal matrix
        self.R = nn.Parameter(torch.zeros(self.r, embed_dim))
        nn.init.orthogonal_(self.R)
        self.R.requires_grad = False # Often fixed in some variants, but can be learned
        
        # Learned transformation in the low-rank subspace
        self.learned_W = nn.Linear(self.r, self.r)
        nn.init.eye_(self.learned_W.weight)
        nn.init.zeros_(self.learned_W.bias)

    def forward(self, h):
        # h: [batch, seq_len, embed_dim]
        
        # Project to low-rank subspace
        h_low = torch.matmul(h, self.R.T) # [batch, seq_len, r]
        
        # Apply learned transformation
        h_transformed = self.learned_W(h_low) # [batch, seq_len, r]
        
        # Compute intervention (h_transformed - h_low) and project back
        intervention = torch.matmul(h_transformed - h_low, self.R) # [batch, seq_len, embed_dim]
        
        return h + intervention

class ReFTWrapper(nn.Module):
    """
    A wrapper to apply ReFT interventions at specific positions.
    """
    def __init__(self, intervention_module, positions='last'):
        super().__init__()
        self.intervention = intervention_module
        self.positions = positions # 'all', 'last', 'first'

    def forward(self, h, attention_mask=None):
        if self.positions == 'all':
            return self.intervention(h)
        
        elif self.positions == 'last':
            # Apply only to the last non-padding token
            if attention_mask is not None:
                last_idx = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(h.size(0), device=h.device)
                
                # We only want to modify the last token, so we create a copy or use scatter
                h_new = h.clone()
                h_last = h[batch_idx, last_idx].unsqueeze(1) # [batch, 1, d]
                h_new[batch_idx, last_idx] = self.intervention(h_last).squeeze(1)
                return h_new
            else:
                h_new = h.clone()
                h_new[:, -1:] = self.intervention(h[:, -1:])
                return h_new
                
        elif self.positions == 'first':
            h_new = h.clone()
            h_new[:, 0:1] = self.intervention(h[:, 0:1])
            return h_new
            
        return h
