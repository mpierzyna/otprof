import torch


@torch.compile
def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient.

    Args:
    y_true (torch.Tensor): Ground truth values
    y_pred (torch.Tensor): Predicted values

    Returns:
    torch.Tensor: Pearson correlation coefficient
    """
    # Roll dim to last
    y_true = y_true.transpose(dim, -1)
    y_pred = y_pred.transpose(dim, -1)

    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Remove mean
    vx = y_true - torch.mean(y_true, dim=-1).unsqueeze(-1)
    vy = y_pred - torch.mean(y_pred, dim=-1).unsqueeze(-1)

    # Compute correlation
    num = torch.sum(vx * vy, dim=-1)
    den = torch.sqrt(torch.sum(vx**2, dim=-1)) * torch.sqrt(torch.sum(vy**2, dim=-1))
    corr = num / den

    return corr


@torch.compile
def pearson_r_loss(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int, reduce: str = "mean") -> torch.Tensor:
    """
    Compute the negative Pearson correlation coefficient as a loss function.

    Args:
    y_true (torch.Tensor): Ground truth values
    y_pred (torch.Tensor): Predicted values

    Returns:
    torch.Tensor: Negative Pearson correlation coefficient
    """
    # Return negative correlation
    corr = pearson_r(y_true, y_pred, dim=dim)
    loss = 1 - corr.abs()
    if reduce == "mean":
        return loss.mean()
    else:
        return loss


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Normalized RMSE according to (Koh et al.)"""
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    serr = (y_true - y_pred) ** 2
    if w is not None:
        w = w.to(y_true.device)  # Ensure same device
        serr = serr * w  # weighted squared error
    rmse = torch.sqrt(torch.mean(serr))

    return rmse / torch.sqrt(var_true + var_pred)


@torch.compile
def edist(x: torch.Tensor, y: torch.Tensor, *, normalize: bool = True, reduce: str | None = "mean") -> torch.Tensor:
    """Energy distance between two sets of samples.

    Parameters
    ----------
    x : torch.Tensor
        First set of samples, shape (batch, levels, features) or (batch, levels).
    y : torch.Tensor
        Second set of samples, shape (batch, levels, features) or (batch, levels).
    normalize : bool, optional
        If True, normalize the energy distance to be in the range [0, 1].
        Default is True.

    Returns
    -------
    torch.Tensor
        The energy distance between the two sets of samples.
    """

    def _edist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances using cdist
        dist_xy = torch.cdist(x, y, p=2)  # (n, m)
        dist_xx = torch.cdist(x, x, p=2)  # (n, n)
        dist_yy = torch.cdist(y, y, p=2)  # (m, m)

        a = dist_xy.mean()
        b = dist_xx.mean()
        c = dist_yy.mean()

        E = 2 * a - b - c
        if normalize:
            return E / (2 * a)  # bound the energy distance to [0, 1]

        return E

    assert x.ndim == y.ndim, "Tensors must have the same number of dimensions"

    if x.ndim == 2:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

    _, _, f = x.shape
    E = torch.tensor([_edist(x[:, :, i], y[:, :, i]) for i in range(f)], device=x.device, dtype=x.dtype)
    if reduce == "mean":
        E = E.mean()

    return E


def non_uniform_central_diff(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute central differences of `y` along the vertical dimension (dim=1) for non-uniform spacing `z`.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor of shape (batch, vert, features)
    z : torch.Tensor
        Vertical coordinate tensor of shape (batch, vert)

    Returns
    -------
    torch.Tensor
        Central differences of `y` along the vertical dimension.
    """
    # y: (batch, vert, features)
    # z: (batch, vert)

    # z slices (Batch, Vert)
    z_minus = z[:, :-2]
    z_curr = z[:, 1:-1]
    z_plus = z[:, 2:]

    f1 = z_curr - z_minus
    f2 = z_plus - z_curr
    a = f2 / f1  # shape: (batch, vert-2)

    # FIX: Unsqueeze 'a' and 'f1' to (batch, vert-2, 1) to broadcast with y's feature dim
    a = a.unsqueeze(-1)
    f1 = f1.unsqueeze(-1)

    # y slices (Batch, Vert, Features)
    y_minus = y[:, :-2]
    y_curr = y[:, 1:-1]
    y_plus = y[:, 2:]

    # Implementation of the provided formula
    num = y_plus + (a**2 - 1) * y_curr - a**2 * y_minus
    den = a * (a + 1) * f1
    return num / den


def non_uniform_fwd_diff(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute forward differences of `y` along the vertical dimension (dim=1) for non-uniform spacing `z`.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor of shape (batch, vert, features)
    z : torch.Tensor
        Vertical coordinate tensor of shape (batch, vert)

    Returns
    -------
    torch.Tensor
        Forward differences of `y` along the vertical dimension.
    """
    # Compute simple forward differences
    # z.diff(dim=1) is (batch, vert-1). Unsqueeze to (batch, vert-1, 1)
    dz = z.diff(dim=1).unsqueeze(-1)
    return y.diff(dim=1) / dz


@torch.compile
def ddz_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    z: torch.Tensor,
    norm: int = 2,
    reduce: str = "mean",
) -> torch.Tensor:
    """Loss based on the vertical derivative of the prediction and target.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values of shape (batch, levels, features).
    y_pred : torch.Tensor
        Predicted values of shape (batch, levels, features).
    z : torch.Tensor
        Vertical coordinate tensor of shape (batch, levels)
    dim : int
        Dimension along which to compute the vertical derivative.

    Returns
    -------
    torch.Tensor
        The mean squared error of the vertical derivatives.
    """

    def _loss(a, b):
        ab_err = torch.linalg.vector_norm(a - b, ord=norm, dim=1)
        if reduce == "mean":
            return ab_err.mean()
        else:
            return ab_err

    assert z.shape[0] == y_true.shape[0], "z and y must have the same batch size"

    # Compute forward differences
    dy_true = non_uniform_fwd_diff(y_true, z)
    dy_pred = non_uniform_fwd_diff(y_pred, z)
    loss = _loss(dy_true, dy_pred)

    # Compute central differences
    dy_true_cd = non_uniform_central_diff(y_true, z)
    dy_pred_cd = non_uniform_central_diff(y_pred, z)
    loss += _loss(dy_true_cd, dy_pred_cd)

    return loss
