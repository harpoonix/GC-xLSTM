import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, num_series):
        """
        Linear projection layer and associated group lasso weights
        """
        super(LassoWeights, self).__init__()
        self.param = nn.Parameter((torch.randn(num_series) * 1e-4))
        self.softmax = nn.Softmax(dim=0)

    def get_weights(self) -> torch.Tensor:
        # return torch.ones_like(self.param)
        return self.softmax(self.param)


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
        self.projection = nn.Linear(num_series, hidden)
        self.lasso_weights = LassoWeights(num_series)
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

        GC = [
            torch.norm(net.projection.weight, dim=0) for net in self.networks
        ]  # take the norm along output dimension,
        # none of the outputs should depend on the input for non granger causality,
        # which means that norm along output is correct

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


def prox_update(network: LSTM, lam, lr):
    """Perform in place proximal update on first layer weight matrix."""
    W = network.lstm.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(
        norm - (lr * lam), min=0.0
    )
    network.lstm.flatten_parameters()


def alpha_loss(net: xLSTM, lam_alpha, use_log=False):
    """Calculate alpha loss for xLSTM model."""
    alpha_loss = (
        net.lasso_weights.get_weights()
        @ (torch.norm(net.projection.weight, dim=0).detach())
    )
    if use_log:
        alpha_loss = torch.log(alpha_loss + 1e-7)
    return lam_alpha * alpha_loss


def prox_update_xlstm(network: xLSTM, lam, lr, threshold=None):
    """Perform in place proximal update on first layer weight matrix."""
    W = network.projection.weight
    norm = torch.norm(W, dim=0)
    if threshold is not None:
        W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.where(
            norm - (lr * lam) >= threshold, norm - (lr * lam), torch.zeros_like(norm)
        )
    else:
        W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(
            norm - (network.lasso_weights.get_weights() * network.p * lr * lam), min=0.0
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
    input = data[input_indices].to(torch.float32)

    target_indices = input_indices + 1
    target = data[target_indices].to(torch.float32)

    return input.detach(), target.detach()


def train_model_gista(
    clstm,
    X,
    context,
    lam,
    lam_ridge,
    lr,
    max_iter,
    check_every=50,
    r=0.8,
    lr_min=1e-8,
    sigma=0.5,
    monotone=False,
    m=10,
    lr_decay=0.5,
    begin_line_search=True,
    switch_tol=1e-3,
    verbose=1,
):
    """
    Train cLSTM model with GISTA.

    Args:
      clstm: clstm model.
      X: tensor of data, shape (batch, T, p).
      context: length for short overlapping subsequences.
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    """
    p = clstm.p
    clstm_copy = deepcopy(clstm)
    loss_fn = nn.MSELoss(reduction="mean")
    lr_list = [lr for _ in range(p)]

    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p):
        net = clstm.networks[i]
        pred, _ = net(X)
        mse = loss_fn(pred[:, :, 0], Y[:, :, i])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = clstm.networks[i]
            net_copy = clstm_copy.networks[i]

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(), net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it)

                # Check line search criterion.
                pred, _ = net_copy(X)
                mse = loss_fn(pred[:, :, 0], Y[:, :, i])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam)
                    loss = smooth + nonsmooth
                    tol = (0.5 * sigma / lr_it) * sum(
                        [
                            torch.sum((param - temp_param) ** 2)
                            for param, temp_param in zip(
                                net.parameters(), net_copy.parameters()
                            )
                        ]
                    )

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print("Taking step, network i = %d, lr = %f" % (i, lr_it))
                        print("Gap = %f, tol = %f" % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (lr_list[i] ** (1 - lr_decay)) * (lr_it**lr_decay)

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print("Network %d converged" % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                clstm.networks[i], clstm_copy.networks[i] = net_copy, net

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print("Done at iteration = %d" % (it + 1))
            break

        # Check progress
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it + 1))
                print("Total loss = %f" % loss_mean)
                print(
                    "MSE = %f, Ridge = %f, Nonsmooth = %f"
                    % (mse_mean, ridge_mean, nonsmooth_mean)
                )
                print(
                    "Variable usage = %.2f%%" % (100 * torch.mean(clstm.GC().float()))
                )

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print("Switching to line search")

    return train_loss_list, train_mse_list


def train_model_adam(
    clstm,
    X,
    context,
    lr,
    max_iter,
    lam=0,
    lam_ridge=0,
    lookback=5,
    check_every=50,
    verbose=1,
):
    """Train model with Adam."""
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(clstm.parameters(), lr=lr)
    train_loss_list = []

    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred = [clstm.networks[i](X)[0] for i in range(p)]
        loss = sum([loss_fn(pred[i][:, :, 0], Y[:, :, i]) for i in range(p)])

        # Add penalty term.
        if lam > 0:
            loss = loss + sum([regularize(net, lam) for net in clstm.networks])

        if lam_ridge > 0:
            loss = loss + sum(
                [ridge_regularize(net, lam_ridge) for net in clstm.networks]
            )

        # Take gradient step.
        loss.backward()
        optimizer.step()
        clstm.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it + 1))
                print("Loss = %f" % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(clstm)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print("Stopping early")
                break

    # Restore best model.
    restore_parameters(clstm, best_model)

    return train_loss_list


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
    for it in tqdm(range(max_iter)):
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
                logger.info("Accuracy = %.2f%%" % accuracy)
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
        best_accuracy_gc,
        best_accuracy_model,
        lam_list,
    )


def train_unregularized(
    clstm, X, context, lr, max_iter, lookback=5, check_every=50, verbose=1
):
    """Train model with Adam."""
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(clstm.parameters(), lr=lr)
    train_loss_list = []

    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred, hidden = clstm(X)
        loss = sum([loss_fn(pred[:, :, i], Y[:, :, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        clstm.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it + 1))
                print("Loss = %f" % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(clstm)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print("Stopping early")
                break

    # Restore best model.
    restore_parameters(clstm, best_model)

    return train_loss_list
