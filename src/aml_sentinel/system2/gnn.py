"""GNN experiment: GraphSAGE edge classifier over account-pair graphs (ADR-016).

**This is an experiment, not part of the production pipeline.** It answers one
question: does message passing over the account graph rank laundering pairs better
than LightGBM's per-transaction scores aggregated to pairs?

Model: two ``SAGEConv`` layers (mean aggregation) over the undirected period graph
produce node embeddings. An MLP scores each directed pair from
``[h_payer, h_payee, edge_features]``.

Training: full batch on the train-period graph, with BCE loss weighted for class
imbalance. It early-stops on validation-pair average precision. Outputs are ranking
scores, **not calibrated probabilities**.

Runs on CPU, or on GPU when available. GPU scatter operations are not bit-exact, so
results are reproducible only to small numerical differences.

``torch`` and ``torch_geometric`` are optional (``requirements/gnn.txt``) and are
imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score

from aml_sentinel.exceptions import ConfigurationError
from aml_sentinel.system2.gnn_data import PeriodGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GnnConfig:
    """Training hyperparameters (fixed, not tuned on test).

    Attributes:
        hidden: Embedding width.
        lr: Adam learning rate.
        weight_decay: L2 penalty.
        max_epochs: Upper bound on full-batch epochs.
        patience: Epochs without validation-AP improvement before stopping.
        max_pos_weight: Cap on the positive-class loss weight.
        seed: Random seed.
    """

    hidden: int = 64
    lr: float = 5e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    max_pos_weight: float = 50.0
    seed: int = 42


def _require_torch() -> Any:
    try:
        import torch
        import torch_geometric  # noqa: F401
    except ImportError:
        raise ConfigurationError(
            "GNN experiment needs torch and torch_geometric: pip install -r requirements/gnn.txt"
        ) from None
    return torch


def build_model(d_node: int, d_edge: int, hidden: int) -> Any:
    """Create the GraphSAGE edge classifier.

    Args:
        d_node: Node feature width.
        d_edge: Edge feature width.
        hidden: Embedding width.

    Returns:
        A ``torch.nn.Module`` taking ``(x, mp_edge_index, pairs, edge_attr)``.
    """
    torch = _require_torch()
    from torch_geometric.nn import SAGEConv

    module_base: Any = torch.nn.Module

    class EdgeSAGE(module_base):  # type: ignore[misc]  # torch is an optional, lazy import
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(d_node, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden + d_edge, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, x: Any, mp_edges: Any, pairs: Any, edge_attr: Any) -> Any:
            h = torch.relu(self.conv1(x, mp_edges))
            h = self.conv2(h, mp_edges)
            z = torch.cat([h[pairs[0]], h[pairs[1]], edge_attr], dim=1)
            return self.head(z).squeeze(-1)

    return EdgeSAGE()


def _tensors(
    graph: PeriodGraph, mean: npt.NDArray[np.float32], std: npt.NDArray[np.float32], device: Any
) -> tuple[Any, Any, Any, Any]:
    torch = _require_torch()
    x = torch.tensor((graph.x - mean) / std, dtype=torch.float32, device=device)
    pairs = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    mp = torch.cat([pairs, pairs.flip(0)], dim=1)  # undirected message passing
    attr = torch.tensor(graph.edge_attr, dtype=torch.float32, device=device)
    return x, mp, pairs, attr


@dataclass
class TrainedGnn:
    """A trained model with the normalisation it expects.

    Attributes:
        model: The torch module.
        mean: Node-feature means (from the train graph).
        std: Node-feature standard deviations.
        device: Torch device.
        best_epoch: Epoch with the best validation AP.
        best_val_ap: That validation AP.
    """

    model: Any
    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    device: Any
    best_epoch: int
    best_val_ap: float

    def predict(self, graph: PeriodGraph) -> npt.NDArray[np.float64]:
        """Ranking scores (sigmoid of logits) for every pair of ``graph``.

        Args:
            graph: Period graph.

        Returns:
            Scores in [0, 1] (not calibrated probabilities).
        """
        torch = _require_torch()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(*_tensors(graph, self.mean, self.std, self.device))
        return np.asarray(torch.sigmoid(logits).cpu().numpy(), dtype=np.float64)


def train_gnn(train: PeriodGraph, val: PeriodGraph, cfg: GnnConfig | None = None) -> TrainedGnn:
    """Train on the train-period graph, early-stopping on validation-pair AP.

    Args:
        train: Train-period graph (its labels are the targets).
        val: Validation-period graph (labels used only for early stopping).
        cfg: Hyperparameters.

    Returns:
        The best model by validation AP.
    """
    cfg = cfg or GnnConfig()
    torch = _require_torch()
    torch.manual_seed(cfg.seed)
    np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train.x.mean(axis=0)
    std = train.x.std(axis=0)
    std[std == 0] = 1.0

    model = build_model(train.x.shape[1], train.edge_attr.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = _tensors(train, mean, std, device)
    y = torch.tensor(train.y, dtype=torch.float32, device=device)
    pos = max(float(train.y.sum()), 1.0)
    pos_weight = torch.tensor(min((len(train.y) - pos) / pos, cfg.max_pos_weight), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trained = TrainedGnn(model, mean, std, device, best_epoch=0, best_val_ap=-1.0)
    best_state = None
    stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(*tr), y)
        loss.backward()
        opt.step()
        val_ap = float(average_precision_score(val.y, trained.predict(val)))
        if val_ap > trained.best_val_ap:
            trained.best_val_ap, trained.best_epoch, stale = val_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("gnn trained", extra={"best_epoch": trained.best_epoch})
    return trained
