import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcherCrowd(nn.Module):
    def __init__(self, cost_class=1.0, cost_point=0.05):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point

    @torch.no_grad()
    def forward(self, outputs, targets):
        out_prob = outputs["pred_logits"].softmax(-1)
        out_points = outputs["pred_points"]
        indices = []
        for batch_index, target in enumerate(targets):
            points = target["point"]
            if len(points) == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue
            cost_class = -out_prob[batch_index][:, 1][:, None].expand(-1, len(points))
            cost_point = torch.cdist(out_points[batch_index], points, p=2)
            cost = self.cost_point * cost_point + self.cost_class * cost_class
            src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.long, device=out_points.device),
                torch.as_tensor(tgt_idx, dtype=torch.long, device=points.device),
            ))
        return indices
