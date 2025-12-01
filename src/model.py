# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
import timm


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Text model
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)

        # Image model
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Projections
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, input_ids, attention_mask, image):
        t = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = t.last_hidden_state[:, 0, :]  # CLS token

        img_feat = self.image_model(image)

        t_emb = self.text_proj(text_feat)
        i_emb = self.image_proj(img_feat)

        fused = torch.cat([t_emb, i_emb], dim=1)
        pred = self.regressor(fused).squeeze(1)
        return pred
