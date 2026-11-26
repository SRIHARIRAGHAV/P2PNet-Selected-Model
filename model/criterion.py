import torch
import torch.nn.functional as F
from torch import nn


class SetCriterionCrowd(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.5,
                 point_loss_coef=0.0002):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.point_loss_coef = point_loss_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        src_logits = outputs["pred_logits"]
        target_classes = torch.zeros(
            src_logits.shape[:2],
            dtype=torch.long,
            device=src_logits.device,
        )

        for batch_index, (_, target_index) in enumerate(indices):
            if len(target_index) > 0:
                target_classes[batch_index, indices[batch_index][0]] = 1

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )

        point_losses = []
        for batch_index, (src_index, target_index) in enumerate(indices):
            if len(src_index) == 0:
                continue
            src_points = outputs["pred_points"][batch_index, src_index]
            target_points = targets[batch_index]["point"][target_index]
            point_losses.append(F.mse_loss(src_points, target_points, reduction="none").sum())

        num_points = max(1, sum(len(t["point"]) for t in targets))
        loss_point = (
            torch.stack(point_losses).sum() / num_points
            if point_losses
            else outputs["pred_points"].sum() * 0.0
        )
        return {
            "loss_ce": loss_ce,
            "loss_point": loss_point * self.point_loss_coef,
        }
