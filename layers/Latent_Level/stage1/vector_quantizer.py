import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, z: torch.tensor, point=False) -> torch.tensor: # latents (8, 128, 8, 8, 2)
        z = z.permute(0, 2, 3, 4, 1).contiguous()  # [B x D x H x W x Z] -> [B x H x W x Z x D]
        latents_shape = z.shape # ( 8, 8, 8, 2, 128 )
        flat_latents = z.view(-1, self.D)  # [BHWZ x D] = [1024, 128]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHWZ x K]

        # Get the encoding that has the min distance
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHWZ, 1]

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        if point :
            vq_loss = commitment_loss * self.beta
        else :
            vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        z_q = z + (z_q - z).detach()

        return z_q.permute(0, 4, 1, 2, 3).contiguous(), vq_loss, min_encoding_indices, latents_shape

    def codebook_to_embedding(self, encoding_inds, latents_shape): # latents (16, 512, 8, 8, 2)
        # Convert to one-hot encodings
        z_q = self.embedding(encoding_inds).view(latents_shape)
        return z_q.permute(0, 4, 1, 2, 3).contiguous()
