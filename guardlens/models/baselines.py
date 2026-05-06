import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig
from guardlens.models.guardlens import GuardLens


class ConversationDeBERTa(nn.Module):
    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.backbone_loaded = False
        self.classifier = nn.Sequential(
            nn.Linear(config.backbone_dim, config.cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim, 1),
        )

    def setup_backbone(self):
        if self.backbone_loaded:
            return
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(self.config.backbone_name)
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.backbone_loaded = True

    def forward(self, input_ids, attention_mask, **kwargs):
        ctx = torch.no_grad() if self.config.freeze_backbone else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :].float()
        logits = self.classifier(cls_embed).squeeze(-1)
        return {
            "cls_logits": logits,
            "attr_logits": None,
            "attr_probs": None,
            "pooled": cls_embed,
        }


class TurnLevelClassifier(nn.Module):
    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.backbone_loaded = False
        self.turn_classifier = nn.Sequential(
            nn.Linear(config.backbone_dim, config.cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim, 1),
        )

    def setup_backbone(self):
        if self.backbone_loaded:
            return
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(self.config.backbone_name)
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.backbone_loaded = True

    def forward(self, input_ids, attention_mask, turn_mask, **kwargs):
        B, T, S = input_ids.shape
        flat_ids = input_ids.reshape(B * T, S)
        flat_mask = attention_mask.reshape(B * T, S)

        ctx = torch.no_grad() if self.config.freeze_backbone else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :].float().reshape(B, T, -1)

        turn_logits = self.turn_classifier(cls_embeds).squeeze(-1)
        turn_logits = turn_logits.masked_fill(turn_mask == 0, -1e9)
        conv_logits = turn_logits.max(dim=1).values

        return {
            "cls_logits": conv_logits,
            "attr_logits": None,
            "attr_probs": None,
            "pooled": cls_embeds.mean(dim=1),
        }


class GuardLensNoFusion(GuardLens):
    def __init__(self, config: GuardLensConfig):
        ablation_config = GuardLensConfig(
            **{k: v for k, v in vars(config).items() if not k.startswith("_")}
        )
        ablation_config.use_gated_fusion = False
        super().__init__(ablation_config)


class GuardLensNoCF(GuardLens):
    pass
