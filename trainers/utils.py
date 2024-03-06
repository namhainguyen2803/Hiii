import torch
import torch.nn as nn

class SinkhornAlgorithm(nn.Module):

    def __init__(self, epsilon=0.1, iterations=100, threshold=1e-9):
        super(SinkhornAlgorithm, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.threshold = threshold

    def _compute_matrix_H(self, u, v, cost_matrix):
        kernel = -cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel

    def forward(self, p, q, cost_matrix):

        u = torch.zeros_like(p)
        v = torch.zeros_like(q)

        for i in range(self.iterations):
            old_u = u
            old_v = v

            H = self._compute_matrix_H(u, v, cost_matrix)
            u = self.epsilon * (torch.log(p + 1e-8) - torch.logsumexp(H, dim=-1)) + u

            if H.ndim == 3:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(0, 2, 1)
            else:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(1, 0)

            v = self.epsilon * (torch.log(q + 1e-8) - torch.logsumexp(H, dim=-1)) + v

            diff = torch.sum(torch.abs(u - old_u), dim=-1) + torch.sum(torch.abs(v - old_v), dim=-1)
            mean_diff = torch.mean(diff)

            if mean_diff.item() < self.threshold:
                break

        K = self._compute_matrix_H(u, v, cost_matrix)
        pi = torch.exp(K)

        return pi