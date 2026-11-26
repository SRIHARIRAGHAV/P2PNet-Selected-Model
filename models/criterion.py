import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcherCrowd(nn.Module):
    def __init__(self, cost_class=1.0, cost_point=0.05):
        super().__init__()
        if cost_class < 0 or cost_point < 0:
            raise ValueError("Matching costs must be non-negative")
        if cost_class == 0 and cost_point == 0:
            raise ValueError("At least one matching cost must be positive")
        self.cost_class = cost_class
        self.cost_point = cost_point

    @torch.no_grad()
    def forward(self, outputs, targets):
        probabilities = outputs["pred_logits"].float().softmax(-1)
        predicted_points = outputs["pred_points"].float()
        matches = []

        for batch_index, target in enumerate(targets):
            target_points = target["point"].to(
                device=predicted_points.device,
                dtype=predicted_points.dtype,
            )
            if target_points.numel() == 0:
                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=predicted_points.device,
                )
                matches.append((empty, empty))
                continue

            class_cost = -probabilities[batch_index, :, 1, None].expand(
                -1,
                target_points.shape[0],
            )
            point_cost = torch.cdist(
                predicted_points[batch_index],
                target_points,
                p=2,
            )
            total_cost = self.cost_class * class_cost + self.cost_point * point_cost
            source_indices, target_indices = linear_sum_assignment(
                total_cost.detach().cpu().numpy()
            )
            matches.append((
                torch.as_tensor(
                    source_indices,
                    dtype=torch.long,
                    device=predicted_points.device,
                ),
                torch.as_tensor(
                    target_indices,
                    dtype=torch.long,
                    device=predicted_points.device,
                ),
            ))
        return matches


class SetCriterionCrowd(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        eos_coef=0.5,
        point_loss_coef=0.0002,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if eos_coef <= 0 or point_loss_coef < 0:
            raise ValueError("Loss coefficients must be non-negative")
        self.num_classes = num_classes
        self.matcher = matcher
        self.point_loss_coef = point_loss_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        predicted_logits = outputs["pred_logits"]
        predicted_points = outputs["pred_points"]
        if predicted_logits.shape[-1] != self.num_classes + 1:
            raise ValueError("Logit class dimension does not match num_classes")
        if predicted_points.shape[:2] != predicted_logits.shape[:2]:
            raise ValueError("Predicted points and logits must align")
        if len(targets) != predicted_logits.shape[0]:
            raise ValueError("One target dictionary is required per image")

        matches = self.matcher(outputs, targets)
        target_classes = torch.zeros(
            predicted_logits.shape[:2],
            dtype=torch.long,
            device=predicted_logits.device,
        )
        point_losses = []
        point_count = 0

        for batch_index, (source_indices, target_indices) in enumerate(matches):
            target_points = targets[batch_index]["point"].to(predicted_points.device)
            if source_indices.numel() == 0:
                continue
            target_classes[batch_index, source_indices] = 1
            selected_points = predicted_points[batch_index, source_indices]
            matched_points = target_points[target_indices]
            point_losses.append(
                F.mse_loss(
                    selected_points.float(),
                    matched_points.float(),
                    reduction="sum",
                )
            )
            point_count += target_points.shape[0]

        classification_loss = F.cross_entropy(
            predicted_logits.float().transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        if point_losses:
            point_loss = torch.stack(point_losses).sum() / max(1, point_count)
        else:
            point_loss = predicted_points.float().sum() * 0.0
        return {
            "loss_ce": classification_loss,
            "loss_point": point_loss * self.point_loss_coef,
        }
