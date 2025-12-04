# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
import timm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_last_n_layers(model, n):
    layers = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    else:
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            layers = base.encoder.layer

    if layers is None:
        return

    total = len(layers)
    for layer in layers[max(0, total - n):]:
        for param in layer.parameters():
            param.requires_grad = True

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT)
        )
        fused_dim = config.HIDDEN_DIM * 3 + 1
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

        # === PARTIAL FREEZE TEXT =============================
        freeze_all(self.text_model)
        unfreeze_last_n_layers(self.text_model, n=6)

        # === PARTIAL FREEZE IMAGE ===========================
        freeze_all(self.image_model)
        if hasattr(self.image_model, "blocks"):
            for block in getattr(self.image_model, "blocks")[-5:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            for param in self.image_model.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, image, mass=None,
                mask_text=False, mask_image=False):
        # TEXT
        if not mask_text:
            t = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            t_emb = mean_pooling(t, attention_mask)
            t_emb = self.text_proj(t_emb)
        else:
            t_emb = torch.zeros((image.size(0), self.text_proj[0].out_features), device=image.device)

        # IMAGE
        if not mask_image:
            img_feat = self.image_model(image)
            i_emb = self.image_proj(img_feat)
        else:
            i_emb = torch.zeros((image.size(0), self.image_proj[0].out_features), device=image.device)

        interaction = t_emb * i_emb
        if mass is None:
            mass = torch.zeros((image.size(0), 1), device=image.device)
        else:
            mass = mass.view(-1, 1)

        fused = torch.cat([t_emb, i_emb, interaction, mass], dim=1)
        return self.regressor(fused).squeeze(1)
