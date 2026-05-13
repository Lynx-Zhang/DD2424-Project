import torch
import torch.nn as nn
from modules.lora import LoRALinear
from modules.reft import LoReFTIntervention, ReFTWrapper

def get_lora_model(model, r=8, lora_alpha=16, target_modules=["query", "key", "value"]):
    """
    Replaces target Linear layers in the model with LoRALinear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if any of the target module names are in the full name
            if any(target in name for target in target_modules):
                # print(f"Applying LoRA to {name}")
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                
                # Replace the layer
                lora_layer = LoRALinear(module, r=r, lora_alpha=lora_alpha)
                setattr(parent, child_name, lora_layer)
    return model

def get_reft_model(model, r=4, layers=[10, 11], positions='last'):
    """
    Applies ReFT interventions to specific layers.
    Since GPT2Model has a list of layers, we can wrap the layers or modify the forward pass.
    A simple way is to wrap the layers themselves or add a module that gets called in the loop.
    """
    # In our GPT2Model, layers are in self.gpt_layers (nn.ModuleList)
    if hasattr(model, 'gpt_layers'):
        for i in layers:
            if i < len(model.gpt_layers):
                orig_layer = model.gpt_layers[i]
                
                # We need a wrapper that calls the original layer and then the intervention
                class ReFTLayerWrapper(nn.Module):
                    def __init__(self, layer, intervention):
                        super().__init__()
                        self.layer = layer
                        self.intervention = intervention
                    def forward(self, hidden_states, attention_mask):
                        h = self.layer(hidden_states, attention_mask)
                        return self.intervention(h, attention_mask)
                
                intervention = ReFTWrapper(
                    LoReFTIntervention(model.config.hidden_size, low_rank_dimension=r),
                    positions=positions
                )
                model.gpt_layers[i] = ReFTLayerWrapper(orig_layer, intervention)
    return model

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
