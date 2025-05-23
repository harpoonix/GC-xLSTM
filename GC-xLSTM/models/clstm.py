import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np
from copy import deepcopy
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlock,
    mLSTMBlock,
)
from experiments.lr_scheduler import (
    LinearWarmupCosineAnnealing,
    CosineIncrementConstant,
)
from typing import Iterable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def calculate_balanced_accuracy(GC_pred, GC_true):
    true_positives = (GC_pred * GC_true).sum()
    false_positives = (GC_pred * (1 - GC_true)).sum()
    true_negatives = ((1 - GC_pred) * (1 - GC_true)).sum()
    false_negatives = ((1 - GC_pred) * GC_true).sum()
    
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    
    bal_acc = (tpr + (1 - fpr)) / 2
    return bal_acc

class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        """
        LSTM model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        """
        super(LSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.lstm.flatten_parameters()
        self.linear = nn.Conv1d(hidden, 1, 1)

    def init_hidden(self, batch):
        """Initialize hidden states for LSTM cell."""
        device = self.lstm.weight_ih_l0.device
        return (
            torch.zeros(1, batch, self.hidden, device=device),
            torch.zeros(1, batch, self.hidden, device=device),
        )

    def forward(self, X, hidden=None):
        # Set up hidden state.
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        # Apply LSTM.
        X, hidden = self.lstm(X, hidden)

        # Calculate predictions using output layer.
        # X has shape (batch, T, num_series).
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class LassoWeights(nn.Module):
    def __init__(self, num_series, context_length = None):
        """
        Linear projection layer and associated group lasso weights
        """
        super(LassoWeights, self).__init__()
        self.context_length = context_length
        if context_length is None:
            self.param = nn.Parameter((torch.randn(num_series) * 1e-4))
        else:
            self.param = nn.Parameter((torch.randn(context_length, num_series) * 1e-4))
        self.softmax = nn.Softmax(dim=0)

    def get_weights(self) -> torch.Tensor:
        # return torch.ones_like(self.param)
        return self.softmax(self.param.flatten()).reshape(self.context_length, -1) if self.context_length is not None else self.softmax(self.param)

class Projection(nn.Module):
    def __init__(self, num_series, hidden, context_length):
        """
        Linear projection layer along with bias for each time step in the context.
        """
        super(Projection, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(context_length, num_series, hidden))
        self.bias = nn.Parameter(torch.Tensor(context_length, hidden))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, X):
        """
        Apply linear projection to input tensor X.
        Args:
          X: input tensor of shape (batch, context_length, num_series).
        
        Returns:
            X: projected tensor of shape (batch, context_length, hidden).
        """
        
        # X = torch.einsum('bcd,cdh->bch', X, self.weight)
        # X = X + self.bias.unsqueeze(0)
        return torch.einsum('bcd,cdh->bch', X, self.weight) + self.bias.unsqueeze(0)
    
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias

class xLSTM(nn.Module):
    def __init__(self, num_series, hidden, config: xLSTMBlockStackConfig):
        """
        xLSTM model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        """
        super(xLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden
        if not config.use_lags:
            self.projection = nn.Linear(num_series, hidden)
            self.lasso_weights = LassoWeights(num_series)
        else:
            self.projection = Projection(num_series, hidden, config.context_length)
            self.lasso_weights = LassoWeights(num_series, config.context_length)

        # Set up network
        self.config = config
        self.xlstm_stack: xLSTMBlockStack = xLSTMBlockStack(self.config)
        self.linear = nn.Conv1d(hidden, 1, 1)

    def forward(self, X):
        # Set up hidden state.

        # Apply LSTM.
        # X = torch.einsum('bcd,hd->bch', X, self.projection)
        X = self.projection(X)
        X = self.xlstm_stack(X)

        # Calculate predictions using output layer.
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1)


class cLSTM(nn.Module):
    def __init__(self, num_series, hidden):
        """
        cLSTM model with one LSTM per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        """
        super(cLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks = nn.ModuleList(
            [LSTM(num_series, hidden) for _ in range(num_series)]
        )

    def forward(self, X, hidden=None):
        """
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        """
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [self.networks[i](X, hidden[i]) for i in range(self.p)]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden

    def GC(self, threshold=True):
        """
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        """
        # check the input-hidden connection weight, tells us which input is important for change in hidden state
        # aka which time series affects this particular time series
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class componentXLSTM(nn.Module):
    def __init__(self, num_series, hidden, config: xLSTMBlockStackConfig):
        """
        componentXLSTM model with one xLSTM per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        """
        super(componentXLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks: Iterable[xLSTM] = nn.ModuleList(
            [xLSTM(num_series, hidden, config) for _ in range(num_series)]
        )

    def forward(self, X):
        """
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        """
        pred = [self.networks[i](X) for i in range(self.p)]
        # print(f'size of pred: {pred[0].shape}')
        pred = torch.cat(pred, dim=2)
        # print(f'size of pred after cat: {pred.shape}')
        return pred

    def GC(self, threshold=True, use_lags=False):
        """
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        """
        # check the input-hidden connection weight, tells us which input is important for change in hidden state
        # aka which time series affects this particular time series
        if not use_lags:
            if isinstance(self.networks[0].projection, nn.Linear):
                GC = [
                    torch.norm(net.projection.weight, dim=0) for net in self.networks
                ]  # take the norm along output dimension,
                # none of the outputs should depend on the input for non granger causality,
                # which means that norm along output is correct
            else:
                GC = [
                    torch.norm(net.projection.weight, dim=(0, 2)) for net in self.networks
                ]
        else:
            # individual lags
            GC = [
                torch.norm(net.projection.weight, dim=2).transpose(0,1) for net in self.networks
            ] # each element shape is (context, num_series)
        
        GC = torch.stack(GC)
            
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class cLSTMSparse(nn.Module):
    def __init__(self, num_series, sparsity, hidden):
        """
        cLSTM model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          hidden: number of units in LSTM cell.
        """
        super(cLSTMSparse, self).__init__()
        self.p = num_series
        self.hidden = hidden
        self.sparsity = sparsity

        # Set up networks.
        self.networks = nn.ModuleList(
            [LSTM(int(torch.sum(sparsity[i].int())), hidden) for i in range(num_series)]
        )

    def forward(self, X, hidden=None):
        """
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        """
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [
            self.networks[i](X[:, :, self.sparsity[i]], hidden[i])
            for i in range(self.p)
        ]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden


def alpha_loss(net: xLSTM, lam_alpha, use_log=False):
    """Calculate alpha loss for xLSTM model."""
    if not net.config.use_lags:
        alpha_loss = (
            net.lasso_weights.get_weights()
            @ (torch.norm(net.projection.weight, dim=0).detach())
        )
    else:
        alpha_loss = torch.sum(
            net.lasso_weights.get_weights()
            * (torch.norm(net.projection.weight, dim=(2)).detach())
        )
    if use_log:
        alpha_loss = torch.log(alpha_loss + 1e-7)
    return lam_alpha * alpha_loss


def prox_update_xlstm(network: xLSTM, lam, lr, threshold=None):
    """Perform in place proximal update on first layer weight matrix."""
    W = network.projection.weight
    lasso_weights = network.lasso_weights.get_weights()
    if not network.config.use_lags:
        norm = torch.norm(W, dim=0)
        context_length_factor = 1
    else:
        norm = torch.norm(W, dim=2, keepdim=True)
        context_length_factor = network.config.context_length
        lasso_weights = lasso_weights.unsqueeze(-1)
    if threshold is not None:
        W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.where(
            norm - (lr * lam) >= threshold, norm - (lr * lam), torch.zeros_like(norm)
        )
    else:
        W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(
            norm - (lasso_weights * network.p * context_length_factor * lr * lam), min=0.0
        )


def regularize(network, lam, xlstm=False):
    """Calculate regularization term for first layer weight matrix."""
    if xlstm:
        W = network.projection.weight
        return 10 * lam * torch.sum(torch.norm(W, dim=0))
    else:
        W = network.lstm.weight_ih_l0
        return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam, xlstm=False):
    """Apply ridge penalty at linear layer and hidden-hidden weights."""
    if xlstm:
        return lam * (
            torch.sum(network.linear.weight**2)
            + sum(
                [
                    (
                        (
                            torch.sum(block.xlstm.igate.weight**2)
                            + torch.sum(block.xlstm.fgate.weight**2)
                            + torch.sum(block.xlstm.ogate.weight**2)
                            + torch.sum(block.xlstm.zgate.weight**2)
                        )
                        if isinstance(block, sLSTMBlock)
                        else (
                            (
                                torch.sum(block.xlstm.q_proj.weight**2)
                                + torch.sum(block.xlstm.k_proj.weight**2)
                                + torch.sum(block.xlstm.v_proj.weight**2)
                            )
                            if isinstance(block, mLSTMBlock)
                            else (
                                TypeError(
                                    f"Block of type {type(block)} is not supported."
                                )
                            )
                        )
                    )
                    for block in network.xlstm_stack.blocks
                ]
            )
        )

    return lam * (
        torch.sum(network.linear.weight**2) + torch.sum(network.lstm.weight_hh_l0**2)
    )


def restore_parameters(model, best_model):
    """Move parameter values from best_model to model."""
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    """
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    """
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    target = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start + 1 : end + 1]
    return input.detach(), target.detach()

def arrange_input_with_stride(data, context, stride):
    """
    Arrange a single time series into overlapping short sequences with a given stride.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
      stride: step size between consecutive context windows.
    """
    assert context >= 1 and isinstance(context, int)
    assert stride >= 1 and isinstance(stride, int)

    if data.ndim == 3:
        T = data.shape[1]
    else:
        T = data.shape[0]
    max_start = T - context - 1
    if max_start < 0:
        num_samples = 0
    else:
        num_samples = (max_start // stride) + 1

    if num_samples == 0:
        return (
            torch.empty(0, context, data.shape[1], dtype=torch.float32, device=data.device),
            torch.empty(0, context, data.shape[1], dtype=torch.float32, device=data.device),
        )

    # Generate start indices for each sample
    start_indices = torch.arange(num_samples, device=data.device) * stride
    # Generate context indices (0 to context-1)
    context_indices = torch.arange(context, device=data.device)

    # Compute input and target indices using broadcasting
    input_indices = start_indices.unsqueeze(1) + context_indices.unsqueeze(0)
    if data.ndim == 3:
        input = data[:, input_indices].to(torch.float32)
    else:
        input = data[input_indices].to(torch.float32)

    target_indices = input_indices + 1
    if data.ndim == 3:
        target = data[:, target_indices].to(torch.float32)
    else:
        target = data[target_indices].to(torch.float32)

    return input.detach(), target.detach()


def train_model_ista(
    clstm: componentXLSTM,
    X,
    context,
    lr,
    max_iter,
    lam=0,
    lam_alpha=0.1,
    lam_ridge=0,
    lookback=5,
    lam_warmup_steps=5000,
    check_every=50,
    lr_warmup_steps=2000,
    lr_decay_factor=1e-3,
    true_GC=None,
    threshold=None,
    use_lam_scheduler=False,
    use_lam_alpha_scheduler=False,
    alpha_loss_scale=1,
    lam_alpha_start_step=0,
    use_log = False,
    sequence_stride: int = 1,
    verbose=1,
    batch_size: int = 256,
    **kwargs,
):
    """Train model with Adam."""
    logger.info(f"Arguments passed to train_model_ista")
    for key, value in locals().items():
        logger.info(f"{key}: {value}")

    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction="mean")
    train_loss_list = []
    pred_loss_list = []
    alpha_loss_list = []
    var_usage_list = []
    accuracy_list = []
    balanced_accuracy_list = []
    lam_list = []
    best_accuracy_gc = None
    best_accuracy_model = None

    print(f"in training, size of X: {X.shape}")
    # in training, size of X: torch.Size([1, 1000, 10])

    # Set up data.
    X, Y = zip(*[arrange_input_with_stride(x, context, sequence_stride) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    print(f"after arranging input, size of X: {X.shape}, size of Y: {Y.shape}")
    
    # Create DataLoader for batching
    dataset = TensorDataset(X, Y)
    sampler = RandomSampler(dataset, replacement=True, num_samples=256*max_iter)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler)
    
    # after arranging input, size of X: torch.Size([990, 10, 10]), size of Y: torch.Size([990, 10, 10])
    # For early stopping.
    best_it = [None] * p
    best_loss = [np.inf] * p
    best_model = [None] * p

    optimizer = optim.AdamW(clstm.parameters(), lr=lr)
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        lr_warmup_steps,
        max_iter,
        lr,
        lr * lr_decay_factor,
    )

    lam_scheduler = CosineIncrementConstant(
        max_lr=lam,
        min_lr=0,
        start_step=0,
        end_step=lam_warmup_steps,
    )

    lam_alpha_scheduler = CosineIncrementConstant(
        max_lr=lam_alpha,
        min_lr=0,
        start_step=lam_alpha_start_step,
        end_step=lam_warmup_steps
    )
    
    X, Y = next(iter(dataloader))
    
    # Calculate smooth error.
    pred = [clstm.networks[i](X) for i in range(p)]
    logger.info(f"in training, size of pred: {pred[0].shape}")
    individual_pred_loss = [loss_fn(pred[i][:, :, 0], Y[:, :, i]) for i in range(p)]

    pred_loss = sum(individual_pred_loss)
    ridge = sum(
        [ridge_regularize(net, lam_ridge, xlstm=True) for net in clstm.networks]
    ) # = 0

    # smooth = pred_loss + ridge
    smooth = pred_loss + ridge + sum([alpha_loss(net, lam_alpha, use_log) for net in clstm.networks])
    
    # for it in range(max_iter):
    for it, (X, Y) in tqdm(enumerate(dataloader), total=max_iter):
    # for it in tqdm(range(max_iter)):
        # Take gradient step.
        smooth.backward()
        optimizer.step()
        lr_scheduler.step()
        if use_lam_scheduler:
            lam = lam_scheduler.compute_lr(it)
        if use_lam_alpha_scheduler:
            lam_alpha = lam_alpha_scheduler.compute_lr(it)

        # Take prox step.
        if lam > 0:
            for net in clstm.networks:
                prox_update_xlstm(
                    net,
                    lam,
                    (
                        threshold
                        if threshold is not None
                        else optimizer.param_groups[0]["lr"]
                    ),
                )

        clstm.zero_grad()

        # Calculate loss for next iteration.
        pred = [clstm.networks[i](X) for i in range(p)]
        individual_pred_loss = [loss_fn(pred[i][:, :, 0], Y[:, :, i]) for i in range(p)]
        pred_loss = sum(individual_pred_loss)
        ridge = sum(
            [ridge_regularize(net, lam_ridge, xlstm=True) for net in clstm.networks]
        )
        individual_alpha_loss = [alpha_loss(net, lam_alpha, use_log) for net in clstm.networks]
        # smooth = pred_loss + ridge
        smooth = pred_loss + ridge + sum(individual_alpha_loss)

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            # individual_nonsmooth = [regularize(net, lam, xlstm=True) for net in clstm.networks]
            nonsmooth = alpha_loss_scale * sum(individual_alpha_loss)
            mean_loss = (pred_loss + nonsmooth) / p #changed
            individual_mean_loss = [
                (individual_pred_loss[i] + alpha_loss_scale * individual_alpha_loss[i]) for i in range(p)
            ]
            train_loss_list.append(mean_loss.detach().cpu().numpy())
            pred_loss_list.append(pred_loss.detach().cpu().numpy() / p)
            alpha_loss_list.append(nonsmooth.detach().cpu().numpy() / p)
            lam_list.append(lam)
            # lr_list.append(optimizer.param_groups[0]['lr'])

            if verbose > 0:
                logger.info(("-" * 10 + "Iter = %d" + "-" * 10) % (it + 1))
                logger.info("Loss = %f" % mean_loss)
                logger.info("Pred Loss = %f" % (pred_loss / p))
                logger.info("Alpha Loss = %f" % (nonsmooth / p))
                predicted_gc = clstm.GC()
                var_usage = 100 * torch.mean(predicted_gc.float()).detach()
                logger.info("Variable usage = %.2f%%" % (var_usage))
                var_usage_list.append(var_usage.cpu().numpy())
                accuracy = 100 * np.mean(predicted_gc.cpu().data.numpy() == true_GC)
                bal_acc = 100 * calculate_balanced_accuracy(predicted_gc.cpu().data.numpy(), true_GC)
                logger.info("Accuracy = %.2f%%" % accuracy)
                logger.info("Balanced accuracy = %.2f%%" % bal_acc)
                balanced_accuracy_list.append(bal_acc)
                accuracy_list.append(accuracy)
                if (best_accuracy_gc is None) or (accuracy == np.max(accuracy_list)):
                    best_accuracy_gc = clstm.GC(threshold=False)
                    best_accuracy_model = deepcopy(clstm)

            # Check for early stopping.
            for i in range(p):
                if individual_mean_loss[i] < best_loss[i]:
                    best_loss[i] = individual_mean_loss[i]
                    best_it[i] = it
                    best_model[i] = deepcopy(clstm.networks[i])

            # if mean_loss < best_loss:
            #     best_loss = mean_loss
            #     best_it = it
            #     best_model = deepcopy(clstm)
            # elif (it - best_it) == lookback * check_every and it > max_iter // 2:
            #     if verbose:
            #         logger.info('Stopping early')
            #     break

    # Restore best model.
    for i in range(p):
        if best_model[i]:
            restore_parameters(clstm.networks[i], best_model[i])

    return (
        train_loss_list,
        pred_loss_list,
        alpha_loss_list,
        var_usage_list,
        accuracy_list,
        balanced_accuracy_list,
        best_accuracy_gc,
        best_accuracy_model,
        lam_list,
    )
