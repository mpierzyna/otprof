from __future__ import annotations
import torch
from typing import List, Literal, Tuple
import xarray as xr
import numpy as np


def _validate_ds(ds: xr.Dataset):
    # Check stacked sample dimension
    if "sample" not in ds.dims:
        raise ValueError("Dataset must have a 'sample' dimension.")

    # Check dimension
    if len(ds.sizes) > 2:
        raise ValueError("Dataset must be at most 2D (sample + vertical dimension).")

    # Check NaNs
    is_nan = ds.isnull().any()
    vars_nan = [v for v in is_nan if is_nan[v]]
    if vars_nan:
        raise ValueError(f"Dataset contains NaN values in variables: {vars_nan}")

    # Check Infs
    is_inf = ds.isin([np.inf, -np.inf]).any()
    vars_inf = [v for v in is_inf if is_inf[v]]
    if vars_inf:
        raise ValueError(f"Dataset contains Inf values in variables: {vars_inf}")


def _flatten_era5(ds: xr.Dataset) -> xr.DataArray:
    """Flatten ERA5 variables to 2D array with sample as first dimension.
    Dataset[(sample, pl), ..., (sample, pl_h), ..., (sample, ), ...] -> DataArray[(sample, feature)]
    with feature = (pl vars) x n_pl + (pl_h vars) x n_pl_h + (other vars)
    """
    vars_era5 = list(ds.data_vars)

    ## ERA5
    # Split variables into pl, pl_h, and others
    vars_pl = [v for v in vars_era5 if "pl" in ds[v].dims]
    vars_pl_h = [v for v in vars_era5 if "pl_h" in ds[v].dims]
    vars_other = [v for v in vars_era5 if v not in vars_pl and v not in vars_pl_h]

    # Stack variables separately because of different dimensions
    # We have to drop feature MultiIndex because concat will otherwise fail
    _da_pl = ds[vars_pl].to_array().stack(feature=["variable", "pl"])
    _da_pl = _da_pl.reset_index("feature").drop_vars("pl")
    da = _da_pl
    if vars_pl_h:
        _da_pl_h = ds[vars_pl_h].to_array().stack(feature=["variable", "pl_h"])
        _da_pl_h = _da_pl_h.reset_index("feature").drop_vars("pl_h")
        da = xr.concat([da, _da_pl_h], dim="feature")
    if vars_other:
        _da_other = ds[vars_other].to_array().stack(feature=["variable"])
        _da_other = _da_other.reset_index("feature")
        da = xr.concat([da, _da_other], dim="feature")

    return da.transpose("sample", ...)


class FlatProfileDataset(torch.utils.data.Dataset):
    """Dataset with flattended/stacked ERA5 and WRF profiles. Multi-output dataset."""

    def __init__(
        self,
        ds_era5: xr.Dataset,
        vars_era5: List[str] | None,
        ds_wrf: xr.Dataset,
        vars_wrf: List[str] | None,
        vars_wrf_forcing: List[str] | None,
        device=None,
    ):
        if vars_era5 is None:
            vars_era5 = list(ds_era5.data_vars)
        if vars_wrf is None:
            vars_wrf = list(ds_wrf.data_vars)

        # Make sure, sample dimension is present
        if "sample" not in ds_era5.dims:
            raise ValueError("Dataset must have a 'sample' dimension.")
        _validate_ds(ds_era5[vars_era5])

        if "sample" not in ds_wrf.dims:
            raise ValueError("WRF dataset must have a 'sample' dimension.")
        _validate_ds(ds_wrf[vars_wrf + (vars_wrf_forcing if vars_wrf_forcing else [])])

        self.da_in = _flatten_era5(ds_era5[vars_era5])

        # Add WRF forcing variables if provided as input
        if vars_wrf_forcing:
            # Stack...
            _da_wrf_forcing = ds_wrf[vars_wrf_forcing].to_array().stack(feature=["variable", "bottom_top"])
            _da_wrf_forcing = _da_wrf_forcing.reset_index("feature").drop_vars("bottom_top")

            # ...prefix variables to indicate it is forcing var.
            _da_wrf_forcing["variable"] = _da_wrf_forcing.coords["variable"].str + "_frc"

            # ...and concatenate to input features
            self.da_in = xr.concat([self.da_in, _da_wrf_forcing], dim="feature")

        ## WRF / Target
        da_wrf = ds_wrf[vars_wrf].to_array().stack(feature=["variable", "bottom_top"])
        self.wrf_coord_features = da_wrf.coords["feature"]  # Save for unstacking later
        da_wrf = da_wrf.reset_index("feature")
        self.da_wrf = da_wrf.transpose("sample", ...)

        # Convert to tensors
        self.X = torch.tensor(self.da_in.values, dtype=torch.float32, device=device)
        self.y = torch.tensor(self.da_wrf.values, dtype=torch.float32, device=device)

        _, self.n_in = self.X.shape
        _, self.n_out = self.y.shape

        assert self.X.shape[0] == self.y.shape[0], "Mismatch in number of samples between ERA5 and WRF datasets."

    def get_X_var_mask(self, var_name: str) -> np.ndarray:
        """Get mask for a specific input variable."""
        if var_name not in self.da_in.coords["variable"].values:
            raise ValueError(f"Variable '{var_name}' not found in input features.")
        return self.da_in.coords["variable"].values == var_name  # noqa: returns boolean np.ndarray

    def get_y_var_mask(self, var_name: str) -> np.ndarray:
        """Get mask for a specific target variable."""
        if var_name not in self.da_wrf.coords["variable"].values:
            raise ValueError(f"Variable '{var_name}' not found in target features.")
        return self.da_wrf.coords["variable"].values == var_name  # noqa: returns boolean np.ndarray

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """One stacked (X, y) pair of shapes (n_in, ) and (n_out, ).

        with
        - n_in = (pl vars) x n_pl + (pl_h vars) x n_pl_h + (other vars) + (wrf forcing vars) x n_bt
        - n_out = (wrf target vars) x n_bt

        where
        - pl: pressure levels (ERA5)
        - pl_h: half pressure levels (ERA5)
        - bt: bottom_top levels (WRF)

        Parameters
        ----------
        idx: int

        Returns
        -------
        x: torch.Tensor
        y: torch.Tensor
        """
        x = self.X[idx]
        y = self.y[idx]

        return x, y

    def unstack_y(self, y: torch.Tensor) -> xr.Dataset:
        """Unstack y tensor to xarray Dataset here each variable has shape (sample, bottom_top)."""
        da = xr.DataArray(
            y.cpu().numpy(),
            dims=["sample", "feature"],
            coords={"feature": self.wrf_coord_features},  # We can't add sample coord here because y may be prediction.
        )
        da = da.unstack("feature")
        ds = da.to_dataset(dim="variable")
        return ds


class ProfileDataset(torch.utils.data.Dataset):
    """Dataset with profiles outputs as matrix for conv-based models.
    Surface features get repeated along vertical dimension.
    """

    def __init__(
        self,
        ds_X: xr.Dataset,
        vars_X: List[str] | None,
        ds_y: xr.Dataset,
        vars_y: List[str] | None,
        vars_y_frc: List[str] | None,
        device=None,
    ):
        # Select vars
        if vars_X is None:
            vars_X = list(ds_X.data_vars)
        if vars_y is None:
            vars_y = list(ds_y.data_vars)

        # Validate datasets
        _validate_ds(ds_X[vars_X])
        # Allow empty target lists: only validate if there are variables to validate
        vars_to_validate = []
        if vars_y:
            vars_to_validate += vars_y
        if vars_y_frc:
            vars_to_validate += vars_y_frc
        if vars_to_validate:
            _validate_ds(ds_y[vars_to_validate])

        dim_s, dim_z_X = list(ds_X.dims)
        if dim_s != "sample":
            dim_s, dim_z_X = dim_z_X, dim_s  # Swap if necessary
            assert dim_s == "sample"  # validation should take care of this

        dim_s, dim_z_y = list(ds_y.dims)
        if dim_s != "sample":
            dim_s, dim_z_y = dim_z_y, dim_s  # Swap if necessary
            assert dim_s == "sample"  # validation should take care of this

        # Get input surface variables and expand them along vertical dimension
        vars_X_sfc = [v for v in vars_X if dim_z_X not in ds_X[v].dims]
        ds_X_sfc = ds_X[vars_X_sfc].expand_dims({dim_z_X: ds_X.sizes[dim_z_X]}, axis=1)

        vars_X_vert = [v for v in vars_X if v not in vars_X_sfc]
        ds_X_vert = ds_X[vars_X_vert]

        # Concat inputs into DataArray and convert to tensor
        if len(vars_X_sfc) > 0:
            da_X = xr.concat([ds_X_sfc.to_array(), ds_X_vert.to_array()], dim="variable")
        else:
            da_X = ds_X_vert.to_array()
        da_X = da_X.transpose("sample", dim_z_X, ...)
        self.X = torch.tensor(da_X.values, dtype=torch.float32, device=device)
        self.vars_X: List[str] = da_X.coords["variable"].values.tolist()
        self.n_vars_X: int = len(self.vars_X)
        self.n_levels_X: int = self.X.shape[1]

        # Forcing variables from y dataset
        self.y_frc = None
        self.vars_y_frc: List[str] = []
        self.n_vars_y_frc: int = 0
        if vars_y_frc:
            da_y_forcing = ds_y[vars_y_frc].to_array().transpose("sample", dim_z_y, ...)
            self.y_frc = torch.tensor(da_y_forcing.values, dtype=torch.float32, device=device)
            self.vars_y_frc = da_y_forcing.coords["variable"].values.tolist()
            self.n_vars_y_frc = len(self.vars_y_frc)

        # Create target tensor if variables were provided, else set to None and keep vars empty
        if vars_y:
            da_y = ds_y[vars_y].to_array().transpose("sample", dim_z_y, ...)
            self.y = torch.tensor(da_y.values, dtype=torch.float32, device=device)
            self.vars_y = da_y.coords["variable"].values.tolist()
            self.n_vars_y: int = len(self.vars_y)
            self.n_levels_y: int = self.y.shape[1]
            self._sample_coord = da_y.coords["sample"]  # Save for unstacking later
        else:
            self.y = None
            self.vars_y = []
            self.n_vars_y = 0
            self.n_levels_y = 0
            self._sample_coord = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y_frc is not None:
            X_frc = self.y_frc[idx]
        else:
            X_frc = [torch.nan]

        if self.y is None:
            y_out = [torch.nan]
        else:
            y_out = self.y[idx]

        return self.X[idx], X_frc, y_out

    def unstack_y(self, y: torch.Tensor, sample_coord: None | Literal["self"] | xr.DataArray) -> xr.Dataset:
        """Unstack y tensor to xarray Dataset here each variable has shape (sample, )."""
        # Create dataarray and convert to dataset
        da = xr.DataArray(
            y.cpu().numpy(),
            dims=["sample", "level", "variable"],
            coords={
                "variable": self.vars_y,
            },
        )
        ds = da.to_dataset(dim="variable")

        # Assign coord if passed, but don't unstack. Discarded NaNs can yield unexpected results.
        if sample_coord is not None:
            if isinstance(sample_coord, xr.DataArray):
                ds = ds.assign_coords(sample=sample_coord)
            elif sample_coord == "self":
                ds = ds.assign_coords(sample=self._sample_coord)

        return ds


class IntegratedVarDataset(torch.utils.data.Dataset):
    """Dataset with integrated variables as output, e.g., column integrals.
    Surface features in input get repeated along vertical dimension.
    """

    def __init__(
        self,
        ds_X: xr.Dataset,
        vars_X: List[str] | None,
        ds_y: xr.Dataset,
        vars_y: List[str] | None,
        vars_y_frc: List[str] | None,
        expand_X_sfc: bool,
        device=None,
    ):
        """Dataset with integrated variable outputs as matrix for conv-based models.
        Surface features get repeated along vertical dimension if expand_X_sfc is True.
        """
        # Select vars
        if vars_X is None:
            vars_X = list(ds_X.data_vars)
        if vars_y is None:
            vars_y = list(ds_y.data_vars)

        # Validate datasets
        _validate_ds(ds_X[vars_X])
        # Allow empty target lists: only validate if there are variables to validate
        vars_to_validate = []
        if vars_y:
            vars_to_validate += vars_y
        if vars_y_frc:
            vars_to_validate += vars_y_frc
        if vars_to_validate:
            _validate_ds(ds_y[vars_to_validate])

        dim_s, dim_z = list(ds_X.dims)
        if dim_s != "sample":
            dim_s, dim_z = dim_z, dim_s  # Swap if necessary
            assert dim_s == "sample"  # validation should take care of this

        # If target vars are provided, ensure they are column-integrated (only sample dim)
        if vars_y:
            assert (
                len(ds_y[vars_y].dims) == 1
            ), "Target variables must be column-integrated, i.e., only sample dim can remain."

        # Get input surface variables
        vars_X_sfc = [v for v in vars_X if dim_z not in ds_X[v].dims]
        ds_X_sfc = ds_X[vars_X_sfc]
        if expand_X_sfc:
            # Optionally, expand them along vertical dimension
            ds_X_sfc = ds_X_sfc.expand_dims({dim_z: ds_X.sizes[dim_z]}, axis=1)

        # Get vertical input variables
        vars_X_vert = [v for v in vars_X if v not in vars_X_sfc]
        ds_X_vert = ds_X[vars_X_vert]

        # Concat surface inputs to vert inputs or store separately.
        # When expanding, sfc inputs will be part of X, so for consistency, pretend there are no sfc vars
        self.X_sfc = None
        self.vars_X_sfc = []
        self.n_vars_X_sfc = 0

        if len(vars_X_sfc) > 0:
            if expand_X_sfc:
                # When expanding, concat expanded sfc and vert vars
                da_X = xr.concat([ds_X_sfc.to_array(), ds_X_vert.to_array()], dim="variable")
            else:
                # Else, keep sfc and vert vars separate
                da_X = ds_X_vert.to_array()
                da_X_sfc = ds_X_sfc.to_array().transpose("sample", ...)
                self.X_sfc = torch.tensor(da_X_sfc.values, dtype=torch.float32, device=device)
                self.vars_X_sfc = vars_X_sfc
                self.n_vars_X_sfc = len(self.vars_X_sfc)
        else:
            # Nothing to do with sfc vars here because there aren't any
            da_X = ds_X_vert.to_array()

        da_X = da_X.transpose("sample", dim_z, ...)
        self.X = torch.tensor(da_X.values, dtype=torch.float32, device=device)
        self.vars_X: List[str] = da_X.coords["variable"].values.tolist()
        self.n_vars_X: int = len(self.vars_X)
        self.n_levels_X: int = self.X.shape[1]

        # Forcing variables from y dataset
        self.y_frc = None
        self.vars_y_frc = []
        self.n_vars_y_frc = 0
        if vars_y_frc:
            da_y_forcing = ds_y[vars_y_frc].to_array().transpose("sample", dim_z, ...)
            self.y_frc = torch.tensor(da_y_forcing.values, dtype=torch.float32, device=device)
            self.vars_y_frc = da_y_forcing.coords["variable"].values.tolist()
            self.n_vars_y_frc = len(self.vars_y_frc)

        # Create target tensor if variables were provided, else set to None and keep vars empty
        if vars_y:
            da_y = ds_y[vars_y].to_array().transpose("sample", ...)
            self.y = torch.tensor(da_y.values, dtype=torch.float32, device=device)
            self.vars_y = da_y.coords["variable"].values.tolist()
            self.n_vars_y: int = len(self.vars_y)
            self._sample_coord = da_y.coords["sample"]  # Save for unstacking later
        else:
            self.y = None
            self.vars_y = []
            self.n_vars_y = 0
            self._sample_coord = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shapes:
        - X: (n_vars_X, n_levels_X)
        - X_sfc: (n_vars_X_sfc, ) or NaN
        - y_frc: (n_vars_y_frc, n_levels_y) or NaN
        - y: (n_vars_y, )
        """
        X_sfc = [torch.nan] if self.X_sfc is None else self.X_sfc[idx]
        y_frc = [torch.nan] if self.y_frc is None else self.y_frc[idx]
        y_out = [torch.nan] if self.y is None else self.y[idx]
        return self.X[idx], X_sfc, y_frc, y_out

    def unstack_y(self, y: torch.Tensor, sample_coord: None | Literal["self"] | xr.DataArray) -> xr.Dataset:
        """Unstack y tensor to xarray Dataset here each variable has shape (sample, )."""
        # Create dataarray and convert to dataset
        da = xr.DataArray(
            y.cpu().numpy(),
            dims=["sample", "variable"],
            coords={
                "variable": self.vars_y,
            },
        )
        ds = da.to_dataset(dim="variable")

        # Assign coord if passed, but don't unstack. Discarded NaNs can yield unexpected results.
        if sample_coord is not None:
            if isinstance(sample_coord, xr.DataArray):
                ds = ds.assign_coords(sample=sample_coord)
            elif sample_coord == "self":
                ds = ds.assign_coords(sample=self._sample_coord)

        return ds
