import torch
import torch.nn as nn
from torch.nn import Linear, ModuleList, ReLU

from torch_geometric.nn import GINEConv

from models.model_utils import PostProcess
from models.loss import NormalCRPS, MixedNormalCRPS, MixedLoss

class ResGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, hidden_channels: int):
        super(ResGnn, self).__init__()
        assert num_layers > 0, "num_layers must be > 0."

        # Create Layers
        self.convolutions = ModuleList()
        
        out = hidden_channels
        for layer in range(num_layers):
            # Define an MLP with BatchNorm1d inserted after each linear transformation.
            mlp = nn.Sequential(
                Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, out)
            )
            self.convolutions.append(
                GINEConv(nn=mlp, train_eps=True, edge_dim=1)
            )
            # Update in_channels for next layer to match the output dimension.
            if layer == num_layers - 1:
                out = out_channels
        self.relu = ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                # First Layer
                x = conv(x, edge_index, edge_attr)
                x = self.relu(x)
            else:
                x = x + self.relu(conv(x, edge_index, edge_attr))  # Residual Layers
        return x

# Simple DeepSet Encoder for ensemble predictions.
class DeepSetEncoder(nn.Module):
    def __init__(self, ensemble_in_dim, hidden_channels, out_channels):
        super(DeepSetEncoder, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(ensemble_in_dim, hidden_channels),
            #nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            #nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, ensemble_feats):
        # ensemble_feats: [N, E, ensemble_in_dim]
        phi_out = self.phi(ensemble_feats)   # [N, E, hidden_channels]
        aggregated = phi_out.sum(dim=1)       # [N, hidden_channels] (order invariant)
        return self.rho(aggregated)            # [N, out_channels]

class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,               # dimension of extra station features (x_extra) 
        hidden_channels_gnn,       # hidden size for GIN layers
        out_channels_gnn,          # output dimension from the GIN network (pre aggregation)
        num_layers_gnn,
        optimizer_class,
        optimizer_params,
        loss,
        grad_u=False,
        u=0.5,
        xi=0.5
    ):
        super(GNN, self).__init__()

        # Set loss and output dim
        self.loss = loss
        self.grad_u = grad_u
        self.u = u
        self.xi = xi
        if self.loss == "NormalCRPS":
            self.loss_fn = NormalCRPS()
            self.out_channels = 2
        elif self.loss == "MixedNormalCRPS":
            self.loss_fn = MixedNormalCRPS()
            self.out_channels = 3
        elif self.loss == "MixedLoss":
            if self.grad_u == "True":
                self.loss_fn = MixedLoss(grad_u = True, xi = self.xi)
                self.out_channels = 5
            else:
                self.loss_fn = MixedLoss(grad_u = False, u = self.u, xi = self.xi)
                self.out_channels = 4

        # DeepSet encoder to embed the ensemble forecasts.
        self.deepset = DeepSetEncoder(ensemble_in_dim=in_channels,
                                      hidden_channels=hidden_channels_gnn,
                                      out_channels=hidden_channels_gnn)
        # The node feature becomes the concatenation of:
        #   - Extra station features (data.x) of dimension in_channels.
        #   - DeepSet embedding from ensemble predictions of dimension deepset_out_dim.
        total_node_in_dim = in_channels + hidden_channels_gnn #in_channels + hidden_channels_deepset
        self.dim_red = Linear(total_node_in_dim, hidden_channels_gnn)

        # Residual GINE network.
        self.conv = ResGnn(
            in_channels=hidden_channels_gnn,
            hidden_channels=hidden_channels_gnn,
            out_channels=hidden_channels_gnn,
            num_layers=num_layers_gnn
        )
        # Simple aggregation: a linear layer to map to final out_channels.
        self.aggr = nn.Linear(out_channels_gnn, self.out_channels)
        # Postprocessing module (assumed defined elsewhere).
        self.postprocess = PostProcess(self.loss, self.grad_u)
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

    def forward(self, data):
        # data must have: x_extra, ensemble, edge_index, edge_attr, batch.
        # Embed ensemble forecasts with DeepSet.
        ensemble_emb = self.deepset(data.ensemble)  # shape [N, deepset_out_dim]
        # Concatenate extra station features.
        node_features = torch.cat([data.x, ensemble_emb], dim=1)  # shape [N, total_node_in_dim]
        node_features = self.dim_red(node_features)
        
        # Process graph using residual GATv2 network.
        x = self.conv(node_features, data.edge_index, data.edge_attr)
        x = self.aggr(x)
        x = self.postprocess(x)
        return x

    def training_step(self, batch):
        y_hat = self.forward(batch)
        loss = self.loss_fn.crps(prediction=y_hat, y=batch.y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1
        )  # The batch size is not actually 1 but the loss is already averaged over the batch
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)

    def validation_step(self, batch):
        y_hat = self.forward(batch)
        loss = self.loss_fn.crps(prediction=y_hat, y=batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def test_step(self, batch):
        y_hat = self.forward(batch)
        loss = self.loss_fn.crps(prediction=y_hat, y=batch.y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def initialize(self, dataloader):
        batch = next(iter(dataloader))
        self.validation_step(batch, 0)
