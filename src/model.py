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
    encoder_layers = model.encoder.layer  # <-- правильный путь
    total = len(encoder_layers)
    for layer in encoder_layers[total - n:]:
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
            nn.BatchNorm1d(config.HIDDEN_DIM),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(config.HIDDEN_DIM),
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 3, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

        # === PARTIAL FREEZE TEXT =============================
        freeze_all(self.text_model)
        unfreeze_last_n_layers(self.text_model, n=8)

        # === PARTIAL FREEZE IMAGE ===========================
        freeze_all(self.image_model)
        if hasattr(self.image_model, "blocks"):
            for block in self.image_model.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, image,
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

        fused = torch.cat([t_emb, i_emb, t_emb * i_emb], dim=1)
        return self.regressor(fused).squeeze(1)
