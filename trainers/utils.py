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

def DBOT(a, b_d, b_u, cost_matrix, reg, num_iterations, device='cpu'):


    num_target = cost_matrix.shape[1]
    P = torch.exp(-cost_matrix / reg)
    for i in range(num_iterations):

        source_marginal_P = torch.sum(P, dim=1)
        P = torch.matmul(torch.diag(a / source_marginal_P), P)

        target_marginal_P = torch.sum(P, dim=0)
        P = torch.matmul(P, torch.diag(torch.max(b_d / target_marginal_P, torch.ones(num_target).to(device))))

        target_marginal_P = torch.sum(P, dim=0)
        P = torch.matmul(P, torch.diag(torch.min(b_u / target_marginal_P, torch.ones(num_target).to(device))))

    return P