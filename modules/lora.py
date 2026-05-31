import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Freezing the base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros((r, base_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r)))
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.reset_parameters()
        self.merge_weights = merge_weights
        self.merged = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.base_layer.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights
                self.base_layer.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.merged:
            return self.base_layer(x)
        else:
            result = self.base_layer(x)
            lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result + lora_output
