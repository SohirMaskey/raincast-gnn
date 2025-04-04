import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class EmbedStations(nn.Module):
    """A module to embed station IDs into a feature vector.

    Args:
        num_stations_max (int): The maximum number of stations.
        embedding_dim (int): The dimension of the embedding vector.

    Attributes:
        embed (nn.Embedding): The embedding layer.

    """

    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass of the EmbedStations module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after embedding the station IDs.

        """
        station_ids = x[..., 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat(
            (emb_station, x[..., 1:]), dim=-1
        )  # Concatenate embedded station_id to rest of the feature vector
        return x


class MakePositive(nn.Module):
    """
    A module that ensures the output tensor has positive values for sigma.

    Args:
        None

    """

    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MakePositive module.

        Args:
            x (torch.Tensor): The input tensor containting mu and sigma of shape (S,2).

        Returns:
            torch.Tensor: A tensor with positive values for sigma.

        """
        mu, sigma = torch.split(x, 1, dim=-1)
        sigma = F.softplus(sigma) + EPS  # ensure that sigma is positive
        mu_sigma = torch.cat([mu, sigma], dim=-1)
        return mu_sigma
    
class PostProcess(nn.Module):
    """
    A module that apllies some post processing to the output tensor dependent on the loss function.
    """

    def __init__(self ,loss, grad_u):
        super(PostProcess, self).__init__()
        self.loss = loss
        self.grad_u = grad_u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PostProcess module.

        Args:
            x (torch.Tensor): The input tensor containting mu and sigma of shape (S,2).

        Returns:
            torch.Tensor: A tensor with corresponding transformations.

        """
        if self.loss == "NormalCRPS":
            output = MakePositive()(x)
        elif self.loss == "MixedNormalCRPS":
            mu, sigma, p = torch.split(x, 1, dim = -1)
            sigma = F.softplus(sigma) + EPS
            p = F.sigmoid(p)
            output = torch.cat([mu, sigma, p], dim = -1)
        elif self.loss == "MixedLoss":
            if self.grad_u == "True":
                mu, sigma, p, sigma_u, u = torch.split(x, 1, dim=-1)
                sigma = F.softplus(sigma) + EPS
                sigma_u = F.softplus(sigma_u) + EPS
                p = F.sigmoid(p)
                u = F.sigmoid(u)*2.12
                output = torch.cat([mu, sigma, p, sigma_u, u], dim = -1)
            else:
                mu, sigma, p, sigma_u = torch.split(x, 1, dim=-1)
                sigma = F.softplus(sigma) + EPS
                sigma_u = F.softplus(sigma_u) + EPS
                p = F.sigmoid(p)
                output = torch.cat([mu, sigma, p, sigma_u], dim = -1)

        return output
