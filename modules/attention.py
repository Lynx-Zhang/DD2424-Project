import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn

try:
  from torch.nn.attention.flex_attention import flex_attention as _flex_attention_eager, create_block_mask
  # Compile for true block-sparse O(N*w). Without compile FlexAttention
  # materializes the full N*N score matrix (~no speedup vs baseline).
  flex_attention = torch.compile(_flex_attention_eager, dynamic=False)
  _HAS_FLEX_ATTENTION = True
except ImportError:
  _HAS_FLEX_ATTENTION = False


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # Attention implementation selector. Default = 'baseline' keeps the original
    # behaviour for every existing caller. Switch via config.attn_impl in
    # {'baseline', 'flash', 'swa'} to enable the acceleration experiments.
    self.attn_impl = getattr(config, 'attn_impl', 'baseline')
    self.swa_window_size = getattr(config, 'swa_window_size', 128)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    # calculate attention scores
    a = query @ key.transpose(-1, -2) / (self.attention_head_size ** 0.5) # B, H, S, S

    # causal mask
    seq_len = query.size(-2)
    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
      diagonal=1
    )

    a = a.masked_fill(
      causal_mask,
      torch.finfo(a.dtype).min
    )

    a = a + attention_mask

    # softmax + dropout
    p = self.dropout(torch.softmax(a, dim=-1))

    context =  p @ value # B H S D
    context = rearrange(context, 'b h t d -> b t (h d)')

    return context

  def _flash_attention(self, query, key, value, attention_mask):
    # FlashAttention via PyTorch SDPA. On L4 (sm_89) with bf16/fp16 inputs and
    # is_causal=True without an attn_mask, this dispatches to the FA2 kernel.
    # When a padding mask is present we combine it with the causal mask and
    # accept the mem-efficient backend, which still beats the manual baseline.
    dropout_p = self.dropout.p if self.training else 0.0

    has_padding = (attention_mask is not None) and bool((attention_mask < 0).any())

    if not has_padding:
      context = F.scaled_dot_product_attention(
        query, key, value,
        dropout_p=dropout_p,
        is_causal=True,
      )
    else:
      seq_len = query.size(-2)
      causal = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool).triu(1)
      pad = (attention_mask < 0)  # [B, 1, 1, N]
      attn_mask = causal[None, None] | pad  # broadcasts to [B, 1, N, N]
      context = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
      )

    return rearrange(context, 'b h t d -> b t (h d)')

  def _swa_attention(self, query, key, value, attention_mask):
    # Sliding Window Attention (Longformer-style). Three tiers, fastest first:
    #   1. PyTorch FlexAttention with a BlockMask  (torch >= 2.5, no extra deps)
    #   2. flash-attn library's native window_size (requires flash-attn)
    #   3. Banded mask via SDPA                    (always works; no speedup)
    dropout_p = self.dropout.p if self.training else 0.0
    w = self.swa_window_size
    seq_len = query.size(-2)

    if _HAS_FLEX_ATTENTION:
      has_padding = (attention_mask is not None) and bool((attention_mask < 0).any())
      if not has_padding:
        cache_key = (seq_len, w, query.device)
        if getattr(self, '_swa_cache_key', None) != cache_key:
          window = w
          def swa_mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            in_window = (q_idx - kv_idx) <= window
            return causal & in_window
          self._swa_block_mask = create_block_mask(
            swa_mask_mod, B=None, H=None,
            Q_LEN=seq_len, KV_LEN=seq_len,
            device=query.device,
          )
          self._swa_cache_key = cache_key
        context = flex_attention(query, key, value, block_mask=self._swa_block_mask)
        if dropout_p > 0:
          context = F.dropout(context, p=dropout_p, training=self.training)
        return rearrange(context, 'b h t d -> b t (h d)')

    try:
      from flash_attn import flash_attn_func
      q_ = query.transpose(1, 2).contiguous()
      k_ = key.transpose(1, 2).contiguous()
      v_ = value.transpose(1, 2).contiguous()
      out = flash_attn_func(
        q_, k_, v_,
        dropout_p=dropout_p,
        causal=True,
        window_size=(w, 0),
      )
      return rearrange(out, 'b n h d -> b n (h d)')
    except ImportError:
      pass

    i = torch.arange(seq_len, device=query.device)
    diff = i[:, None] - i[None, :]
    band_keep = (diff >= 0) & (diff <= w)
    band_mask = ~band_keep  # True = mask out

    has_padding = (attention_mask is not None) and bool((attention_mask < 0).any())
    if has_padding:
      pad = (attention_mask < 0)
      attn_mask = band_mask[None, None] | pad
    else:
      attn_mask = band_mask[None, None]

    context = F.scaled_dot_product_attention(
      query, key, value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=False,
    )
    return rearrange(context, 'b h t d -> b t (h d)')

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    if self.attn_impl == 'baseline':
      return self.attention(key_layer, query_layer, value_layer, attention_mask)
    elif self.attn_impl == 'flash':
      return self._flash_attention(query_layer, key_layer, value_layer, attention_mask)
    elif self.attn_impl == 'swa':
      return self._swa_attention(query_layer, key_layer, value_layer, attention_mask)
    else:
      raise ValueError(
        f"Unknown attn_impl: {self.attn_impl!r}. Expected 'baseline', 'flash', or 'swa'."
      )
