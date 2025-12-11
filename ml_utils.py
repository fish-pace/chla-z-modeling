# --- required packages
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def _require_keras():
    try:
        import importlib
        tf = importlib.import_module("tensorflow")
        # you can use tf.keras.* everywhere; no need to import keras separately
        return tf
    except Exception as e:
        raise ImportError(
            "TensorFlow/Keras required for this function. "
            "Install with `pip install tensorflow` or `yourpkg[cnn]`."
        ) from e

def time_series_split(
    data: xr.Dataset,
    num_var,
    cat_var=None,
    mask="ocean_mask",
    split_ratio=(0.7, 0.2, 0.1),
    seed=42,
    X_mean=None,
    X_std=None,
    y_var="y",
    years=None,               # select one year, a list of years, or a slice
    cast_float32=True,
    contiguous_splits=False,
    return_full=False,
    nan_max_frac_y=0.5, # what fraction of missing days allowed in y
    nan_max_frac_v=0.05, # what fraction of missing days allowed in numerical vars
    add_missingness=False,
    verbose=False
):
    """
    Pure-NumPy splitter/normalizer for xarray Dataset (NumPy-backed). 
    Splits time indices randomly into train/val/test.
    Normalizes numerical variables only, using either provided or training-set mean/std. 
    Replaces NaNs with 0s.
    Removes days with too many NaNs (> 
    
    Parameters: 
      data: xarray dataset with 'time' dimension 
      years: year(s) to use for training
      num_var: list of numerical variable names (to normalize) 
      cat_var: list of categorical variable names (no normalization) 
      y_var: name of response variable in data. 
      mask: name of the mask in the data. 0 = ignore; 1 = use; can be static or one for each time step (y)
      split_ratio: tuple (train, val, test), must sum to 1.0 
      seed: random seed 
      nan_max_frac_y: maximum percent missing values for response
      nan_max_frac_v: maximum percent missing values for explanatory variables
      X_mean, X_std: optional mean/std arrays for num_var only (shape = [n_num_vars]) 
      cast_float32 : If True, cast outputs to float32 (good for TF)
      verbose: print out info
      return_full: return X and y
      contiguous_splits: versus random splits
      
    Returns: 
      X, y: full input and response arrays (NumPy arrays) 
      X_train, y_train, X_val, y_val, X_test, y_test: split data X_mean, X_std: mean and std used for normalization
      If return_full=False, X and y are None.
    """
    if cat_var is None:
        cat_var = []
    input_var = list(num_var) + list(cat_var)

    # --- checks
    if "time" not in data.dims:
        raise ValueError("Dataset must contain a 'time' dimension.")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")
    if "ocean_mask" not in data:
        raise KeyError("Dataset must contain 'ocean_mask' (1=ocean, 0=land).")

    # ---------- subset by year(s) ----------
    if years is not None:
        if isinstance(years, (str, int)):
            data = data.sel(time=str(years))
        elif isinstance(years, slice):
            data = data.sel(time=years)
        else:
            # assume iterable of years (ints/strs)
            ti = pd.DatetimeIndex(np.asarray(data["time"].values))
            yrs = set(int(y) for y in years)
            sel = xr.DataArray(np.isin(ti.year, list(yrs)), coords={"time": data["time"]}, dims=["time"])
            data = data.sel(time=sel)
    if data.sizes.get("time", 0) == 0:
        raise ValueError("No timesteps left after year filtering.")

    # create a template for broadcasting 2D -> 3D
    template = data[y_var]

    # NaN-based time filtering where mask = 1
    ocean = data["ocean_mask"].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims({"time": data["time"]}).broadcast_like(template)
    else:
        ocean = ocean.broadcast_like(template)

    spatial_dims = [d for d in ocean.dims if d != "time"]
    ocean_pix_per_t = ocean.sum(dim=spatial_dims)

    check_vars = input_var + [y_var]
    valid_times = xr.DataArray(np.ones(data.sizes["time"], dtype=bool), coords={"time": data["time"]}, dims=["time"])

    for v in check_vars:
        if v not in data:
            raise KeyError(f"Variable '{v}' not found in dataset.")
        arr = data[v]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(template)
        else:
            arr = arr.broadcast_like(template)
        frac = nan_max_frac_y if v == y_var else nan_max_frac_v
        nan_thresh = frac * ocean_pix_per_t                     # (time,)
        v_nan = xr.apply_ufunc(np.isnan, arr) & ocean
        v_nan_count = v_nan.sum(dim=spatial_dims)
        # Remove days with too many NaNs
        valid_times = valid_times & (v_nan_count < nan_thresh)

    before = int(data.sizes["time"])
    data = data.sel(time=valid_times)
    ocean = ocean.sel(time=valid_times)
    after = int(data.sizes["time"])
    if after == 0:
        raise ValueError("No timesteps left after NaN filtering.")
    if verbose:
        yrs_msg = f" (years={years})" if years is not None else ""
        print(f"[NaN filter]{yrs_msg} kept {after}/{before} days "
              f"(≤ {nan_max_frac*100:.1f}% NaNs over ocean per variable).")
    
        # --- days-per-month report ---
        t = pd.to_datetime(data["time"].values)
    
        # group by year-month (works for one or many years)
        per_month = (
            pd.Series(1, index=pd.Index(t, name="time"))
            .groupby([t.year, t.month])
            .sum()
            .astype(int)
        )
        # prettify as "YYYY-MM"
        per_month.index = [f"{y:04d}-{m:02d}" for y, m in per_month.index]
        print("Days kept per month:")
        for ym, cnt in per_month.items():
            print(f"    {ym}: {cnt}")
    
        # compact 12-month line when only a single year is present
        if len(pd.unique(t.year)) == 1:
            counts = (
                pd.Series(1, index=t)
                .groupby(t.month)
                .sum()
                .reindex(range(1, 13), fill_value=0)
                .astype(int)
            )
            print("By month (Jan..Dec):", " ".join(f"{c:2d}" for c in counts.values))
    
    # ---------- split indices ----------
    time_len = data.sizes["time"]
    rng = np.random.default_rng(seed)
    all_indices = rng.choice(time_len, size=time_len, replace=False)

    # Compute indices for splitting data into train, validate, and test
    train_end = int(split_ratio[0] * time_len)
    val_end = int((split_ratio[0] + split_ratio[1]) * time_len)
    train_idx = np.sort(all_indices[:train_end])
    val_idx = np.sort(all_indices[train_end:val_end])
    test_idx = np.sort(all_indices[val_end:])


    # ---------- helpers ----------
    def fetch(var):
        tmpl = data[y_var]                    # current (post-filter) template
        arr  = data[var]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(tmpl)
        else:
            # ensure identical order & coords; avoid resurrecting dropped times
            arr = arr.transpose("time", ...).reindex_like(tmpl)
        out = arr.values.astype("float32", copy=False) if cast_float32 else arr.values
        return out

    # stats from training; compute before imputation (with median)
    if num_var:
        if X_mean is None or X_std is None:
            means, stds = [], []
            for v in num_var:
                a = fetch(v)
                a_tr = a[train_idx]
                means.append(np.nanmean(a_tr, axis=(0, 1, 2)))
                stds.append( np.nanstd( a_tr, axis=(0, 1, 2)))
            X_mean = np.asarray(means, dtype="float32" if cast_float32 else a.dtype)
            X_std  = np.asarray(stds,  dtype="float32" if cast_float32 else a.dtype)
        X_std_safe = np.where(X_std == 0, 1.0, X_std)
    else:
        X_mean = np.array([], dtype="float32" if cast_float32 else float)
        X_std  = np.array([], dtype="float32" if cast_float32 else float)
        X_std_safe = X_std

    # ---- precompute per-pixel medians for num_var using ONLY training data
    ocean_np = ocean.transpose("time","lat","lon").values
        
    medians = []
    for v in num_var:
        a      = fetch(v)                 # (T, H, W)
        a_tr   = a[train_idx]             # (t, H, W)
        oce_tr = ocean_np[train_idx]      # (t, H, W) boolean
    
        # Mask: land OR invalid values
        masked = np.ma.array(a_tr, mask=(~oce_tr) | (~np.isfinite(a_tr)))
    
        # Per-pixel median across time (returns masked result if all masked)
        med_ma = np.ma.median(masked, axis=0)        # (H, W) masked array
        med    = med_ma.filled(np.nan)               # fill all-masked pixels with NaN
    
        # Fallback for pixels with no finite ocean values
        cnt = np.isfinite(masked.filled(np.nan)).sum(axis=0)
        if masked.count() > 0:
            global_med = float(np.ma.median(masked))
        else:
            global_med = 0.0
        med = np.where(cnt > 0, med, global_med).astype('float32', copy=False)
    
        medians.append(med)     
        
    def build_split(idx):
        chans = []
    
        # numeric (normalize, impute NaNs with per-pixel medians; optional missingness)
        for k, v in enumerate(num_var):
            a = fetch(v)             # (T, H, W)
            a = a[idx]               # (t, H, W)
            oce_t = ocean_np[idx]  # (t, H, W)
            miss = (~np.isfinite(a)) & oce_t
            # impute with per-pixel median
            a = np.where(miss, medians[k], a)
            # normalize
            a = (a - X_mean[k]) / X_std_safe[k]
            a = np.where(oce_t, a, 0.0)  # set values over land to 0
            chans.append(a.astype("float32", copy=False))
            if add_missingness:
                # add a 0/1 channel indicating originally-missing inputs
                chans.append(miss.astype("float32", copy=False))
    
        # categorical (just fill NaNs with 0 or a benign default)
        for v in cat_var:
            a = fetch(v)             # (T, H, W)
            a = a[idx]
            a = np.nan_to_num(a)     # okay for categorical/auxiliary
            chans.append(a.astype("float32", copy=False))
    
        if not chans:
            raise ValueError("No input variables provided.")
        # stack channels last → (t, H, W, C)
        return np.stack(chans, axis=-1)
    
    # IMPORTANT: keep NaNs in y so your masked loss can ignore cloudy/land pixels!
    y_full = data[y_var].transpose("time", ...).values
    if cast_float32:
        y_full = y_full.astype("float32", copy=False)
    
    def take_y(idx):
        y_s = y_full[idx]        # DO NOT nan to 0 here
        return y_s

    # ---------- build splits ----------
    X_train = build_split(train_idx); y_train = take_y(train_idx)
    X_val   = build_split(val_idx);   y_val   = take_y(val_idx)
    X_test  = build_split(test_idx);  y_test  = take_y(test_idx)

    if return_full:
      X = build_split(slice(0, time_len))
      y = take_y(slice(0, time_len))
    else:
      X = None
      y = None

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std


# --- Saving a model bundle ---

from dataclasses import dataclass

@dataclass
class MLBundle:
    model: object
    meta: dict
    data: dict | None = None       # <- holds dataset, train_idx, test_idx, etc.
    predict_fn: callable | None = None
    plot_fn: callable | None = None

    def predict(self, *args, **kwargs):
        if self.predict_fn is None:
            raise AttributeError("No predict_fn stored in this bundle.")
        return self.predict_fn(*args, **kwargs)

    def plot(self, *args, **kwargs):
        if self.plot_fn is None:
            raise AttributeError("No plot_fn stored in this bundle.")
        return self.plot_fn(*args, **kwargs)

def save_ml_bundle(
    zip_path,
    model,
    dataset=None,
    train_idx=None,
    test_idx=None,
    meta=None,
    predict_helper=None,
    plot_helper=None,
    extra_helpers=None,
):
    """
    Save a model bundle (BRT, CNN, etc.) to a single .zip.

    Contents:
      - model.pkl   or model.keras
      - data.pkl    (optional: dataset, train_idx, test_idx)
      - meta.json   (includes helper function source if provided)

    meta["model_kind"] controls how the model is saved:
      - "keras"   -> saved as model.keras
      - anything else -> pickled as model.pkl

    If `model` is a collection (dict/list/tuple), it is always pickled,
    and metadata fields are added:

      - meta["model_is_collection"] = True
      - meta["model_collection_type"] = "dict" | "list" | "tuple"
      - meta["n_submodels"]
      - meta["model_keys"] (for dict)
    """
    import inspect
    import json
    import pickle
    import tempfile
    import zipfile
    from pathlib import Path

    meta = dict(meta or {})

    # --- Detect if model is a collection ---
    is_collection = isinstance(model, (dict, list, tuple))
    meta["model_is_collection"] = bool(is_collection)

    if is_collection:
        if isinstance(model, dict):
            meta["model_collection_type"] = "dict"
            meta["model_keys"] = [str(k) for k in model.keys()]
        elif isinstance(model, list):
            meta["model_collection_type"] = "list"
        else:
            meta["model_collection_type"] = "tuple"

        meta["n_submodels"] = len(model)

    # --- Infer model kind if not given ---
    if "model_kind" not in meta:
        if is_collection:
            meta["model_kind"] = "pickle"
        else:
            cls_name = type(model).__name__.lower()
            if "sequential" in cls_name or "functional" in cls_name:
                meta["model_kind"] = "keras"
            else:
                meta["model_kind"] = "pickle"

    # --- Gather helper source code ---
    helpers_src = {}

    # Predict helper
    if predict_helper is not None:
        name = predict_helper.__name__
        meta["predict_helper_name"] = name
        try:
            src = inspect.getsource(predict_helper)
            helpers_src[name] = src
        except OSError:
            pass

    # Plot helper
    if plot_helper is not None:
        name = plot_helper.__name__
        meta["plot_helper_name"] = name
        try:
            src = inspect.getsource(plot_helper)
            helpers_src[name] = src
        except OSError:
            pass

    # Extra helpers (e.g. feature adders, single-depth predictor, etc.)
    extra_helpers = extra_helpers or {}
    for name, fn in extra_helpers.items():
        try:
            helpers_src[name] = inspect.getsource(fn)
        except OSError:
            pass

    if helpers_src:
        meta["helpers"] = helpers_src

    # --- If we are given indices, stash them in meta too (JSON-safe) ---
    if train_idx is not None:
        meta["train_idx"] = np.asarray(train_idx).tolist()
    if test_idx is not None:
        meta["test_idx"] = np.asarray(test_idx).tolist()

    # --- Write everything into a temp dir then zip it ---
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1) Save model
        if meta["model_kind"] == "keras" and not is_collection:
            model_path = tmp / "model.keras"
            model.save(model_path)
        else:
            model_path = tmp / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # 2) Save data (dataset + indices)
        data = {
            "dataset": dataset,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        data_path = tmp / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

        # 3) Save meta
        meta_path = tmp / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # 4) Zip everything
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, arcname=model_path.name)
            z.write(data_path, arcname=data_path.name)
            z.write(meta_path, arcname=meta_path.name)

    return str(zip_path)

def _print_bundle_usage(bundle, bundle_path):
    meta = bundle.meta
    data = bundle.data or {}
    model_kind = meta.get("model_kind", "pickle")
    predict_name = meta.get("predict_helper_name")
    plot_name = meta.get("plot_helper_name")

    is_collection = meta.get("model_is_collection", False)
    collection_type = meta.get("model_collection_type")
    model_keys = meta.get("model_keys", [])
    n_submodels = meta.get("n_submodels")

    print(f"\nLoaded ML bundle from: {bundle_path}")
    print(f"  model_kind : {model_kind}")
    if is_collection:
        print(f"  model_type : collection ({collection_type}), n_submodels={n_submodels}")
        if collection_type == "dict" and model_keys:
            example_key = model_keys[0]
            print(f"  example key: {example_key}")
    else:
        print("  model_type : single model")

    if "target_name" in meta:
        print(f"  target     : {meta['target_name']}")

    if "feature_cols" in meta:
        print(f"  features   : {len(meta['feature_cols'])} columns")

    train_idx = meta.get("train_idx")
    test_idx = meta.get("test_idx")
    if train_idx is not None and test_idx is not None:
        print(f"  train/test : {len(train_idx)} / {len(test_idx)} rows")

    if data.get("dataset") is not None:
        try:
            nrows = len(data["dataset"])
            print(f"  dataset    : {nrows} rows stored in bundle")
        except Exception:
            print("  dataset    : stored in bundle (length unknown)")

    print("\nUsage example (Python):")
    print("  bundle = load_ml_bundle('path/to/bundle.zip')")

    if predict_name and bundle.predict_fn is not None:
        print(f"  # Predict using helper '{predict_name}'")

        if is_collection:
            print("  # Example: predict all depths for one day from a BRF dataset R")
            print("  pred = bundle.predict(")
            print("      R_dataset,                  # xr.DataArray/xr.Dataset with lat/lon + predictors")
            print("      brt_models=bundle.model,    # dict of models by depth bin")
            if "feature_cols" in meta:
                print("      feature_cols=bundle.meta['feature_cols'],")
            print("      consts={'solar_hour': 12.0, 'type': 1},")
            print("  )  # -> e.g. CHLA(time?, z, lat, lon)")
        else:
            print("  pred = bundle.predict(")
            print("      R_dataset,                  # xr.DataArray/xr.Dataset with lat/lon + predictors")
            print("      brt_model=bundle.model,     # single model")
            if "feature_cols" in meta:
                print("      feature_cols=bundle.meta['feature_cols'],")
            print("  )")
    else:
        if is_collection:
            print("  # This bundle has no stored predict helper.")
            print("  # 'bundle.model' is a collection (e.g. dict) of models:")
            print("  #   e.g. bundle.model['CHLA_0_10'].predict(X)")
        else:
            print("  # This bundle has no stored predict helper; call bundle.model.predict(...) directly.")

    if plot_name and bundle.plot_fn is not None:
        print(f"\n  # Plot using helper '{plot_name}'")
        print("  fig, ax = bundle.plot(pred_da, pred_label='Prediction')")
    else:
        print("\n  # This bundle has no stored plot helper; use your own plotting code.")

    print("")  # final newline


def load_ml_bundle(zip_path):
    """
    Load a bundle created by save_ml_bundle().

    Returns
    -------
    bundle : MLBundle
        Instance with:
          - model, meta, data (dataset + indices)
          - predict_fn, plot_fn (if stored)
          - data["splits"] = dict with X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test : pandas objects
        Train-test splits reconstructed from dataset, feature_cols, y_col,
        train_idx, and test_idx.
    """
    import json
    import zipfile
    import pickle
    import tempfile
    from pathlib import Path

    zip_path = Path(zip_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        meta = json.loads((tmp / "meta.json").read_text())
        model_kind = meta.get("model_kind", "pickle")

        # --- load model ---
        if model_kind == "keras":
            import keras
            model = keras.saving.load_model(tmp / "model.keras")
        else:
            with open(tmp / "model.pkl", "rb") as f:
                model = pickle.load(f)

        # --- load dataset + indices + metadata ---
        with open(tmp / "data.pkl", "rb") as f:
            data = pickle.load(f)

    # --- Reconstruct helpers (including predict/plot) in a shared namespace ---
    helpers_src = meta.get("helpers", {})
    ns = {}
    predict_fn = None
    plot_fn = None

    if helpers_src:
        for name, src in helpers_src.items():
            exec(src, ns, ns)

        pred_name = meta.get("predict_helper_name")
        if pred_name:
            predict_fn = ns.get(pred_name)

        plot_name = meta.get("plot_helper_name")
        if plot_name:
            plot_fn = ns.get(plot_name)

    # --- Rebuild X_train, X_test, y_train, y_test from stored pieces ---
    # Assumes save_ml_bundle stored these in data and/or meta
    dataset = data.get("dataset")
    if dataset is None:
        raise ValueError("`dataset` not found in bundle data.")

    # feature_cols and y_col might be in data or meta
    feature_cols = data.get("feature_cols", meta.get("feature_cols"))
    y_col = data.get("y_col", meta.get("y_col"))

    if feature_cols is None:
        raise ValueError("`feature_cols` not found in data/meta.")

    if y_col is None:
        raise ValueError("`y_col` not found in data/meta.")

    train_idx = data.get("train_idx")
    test_idx = data.get("test_idx")

    if train_idx is None or test_idx is None:
        raise ValueError("`train_idx` and/or `test_idx` not found in data.")

    # Build full X, y
    X = dataset[feature_cols]
    y = dataset[y_col]

    # Use .loc so this works for labels or boolean masks
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # Attach splits back into data for convenience
    data["X_train"] = X_train
    data["X_test"] = X_test
    data["y_train"] = y_train
    data["y_test"] = y_test

    # Build bundle
    bundle = MLBundle(
        model=model,
        meta=meta,
        data=data,
        predict_fn=predict_fn,
        plot_fn=plot_fn,
    )

    _print_bundle_usage(bundle, zip_path)

    # Return bundle
    return bundle



# ---- Make Predictions from Model

# predictions single brt for a single depth
def make_prediction_brt(
    R: xr.Dataset,
    brt_model,
    feature_cols,
    consts=None,
) -> xr.DataArray:
    """
    Predict a single BRT depth-bin field on a lat/lon grid, automatically
    constructing derived features (solar_hour, sin_time/cos_time, x_geo/y_geo/z_geo)
    if they are requested in feature_cols and not already present.

    Parameters
    ----------
    R : xr.Dataset
        Dataset with lat, lon (and possibly time) plus any derived variables.
    brt_model :
        Fitted sklearn-like model with .predict().
    feature_cols : list of str
        Predictor names expected by the model.
    consts : dict, optional
        Constant feature values, e.g. {"solar_hour": 12.0, "type": 1}.

    Returns
    -------
    xr.DataArray
        Prediction on (lat, lon) as 'y_pred'.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    consts = consts or {}

    # ----------
    # Make a working copy
    # ----------
    ds = R.copy()

    # ----------
    # Add derived features if needed
    # ----------

    # --- solar_hour ---
    if "solar_hour" in feature_cols and "solar_hour" not in ds and "solar_hour" not in consts:
        if "add_solar_hour_feature" not in globals():
            raise RuntimeError(
                "Feature 'solar_hour' is required but helper 'add_solar_hour_feature' "
                "is not available.\n"
                "Either define it (or include it in the bundle), provide "
                "consts={'solar_hour': value}, or pre-add 'solar_hour' to your Dataset."
            )
        ds = add_solar_hour_feature(ds)

    # --- seasonal sin/cos ---
    needs_sin = "sin_time" in feature_cols and "sin_time" not in ds and "sin_time" not in consts
    needs_cos = "cos_time" in feature_cols and "cos_time" not in ds and "cos_time" not in consts

    if needs_sin or needs_cos:
        if "add_seasonal_time_features" not in globals():
            raise RuntimeError(
                "Features 'sin_time'/'cos_time' are required but helper "
                "'add_seasonal_time_features' is not available.\n"
                "Define it (or include it in the bundle), or pre-add 'sin_time' "
                "and 'cos_time' to your Dataset."
            )
        ds = add_seasonal_time_features(ds, time="time")  # adjust args if needed

    # --- spherical coords ---
    needs_x = "x_geo" in feature_cols and "x_geo" not in ds and "x_geo" not in consts
    needs_y = "y_geo" in feature_cols and "y_geo" not in ds and "y_geo" not in consts
    needs_z = "z_geo" in feature_cols and "z_geo" not in ds and "z_geo" not in consts

    if needs_x or needs_y or needs_z:
        if "add_spherical_coords" not in globals():
            raise RuntimeError(
                "Spherical-coord features are required but helper 'add_spherical_coords' "
                "is not available.\n"
                "Define it (or include it in the bundle), or pre-add x_geo/y_geo/z_geo."
            )
        ds = add_spherical_coords(ds)

    # ----------
    # Stack and predict
    # ----------
    ds_stack = ds.stack(pixel=("lat", "lon"))
    n_pixel = ds_stack.sizes["pixel"]

    df_cols = {}
    for feat in feature_cols:
        if feat in consts:
            df_cols[feat] = np.full(n_pixel, consts[feat], dtype=float)
        else:
            if feat not in ds_stack:
                raise KeyError(
                    f"Feature '{feat}' was not found in dataset or consts.\n"
                    f"If this is a derived variable, ensure the helper function "
                    f"is available or add it manually."
                )
            # .values.reshape(n_pixel) to flatten all other dims (here just pixel)
            df_cols[feat] = ds_stack[feat].values.reshape(n_pixel)

    df_pred = pd.DataFrame(df_cols, columns=feature_cols)

    # Handle NaNs
    valid_mask = ~df_pred.isna().any(axis=1)
    df_valid = df_pred[valid_mask]

    y_pred_flat = np.full(n_pixel, np.nan, dtype=float)
    if len(df_valid) > 0:
        y_pred_flat[valid_mask.values] = brt_model.predict(df_valid)

    # reshape back
    pred_map = y_pred_flat.reshape(R.sizes["lat"], R.sizes["lon"])

    return xr.DataArray(
        pred_map,
        coords={"lat": R["lat"], "lon": R["lon"]},
        dims=("lat", "lon"),
        name="y_pred",
    )

def predict_all_depths_for_day(
    R,                     # xr.DataArray (lat, lon, wavelength)
    brt_models: dict,      # e.g. {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}
    feature_cols: list,
    consts=None,           # e.g. {"solar_hour": 0, "type": 1}
    chunk_size_lat: int = 100,
    time=None,             # e.g. "2024-07-15" or np.datetime64
    z=None,                # optional override for depth centers
    z_name: str = "z",     # vertical dimension name
    silent: bool = False,  # kept for compatibility; not used right now
):
    """
    Run BRT predictions for all depth bins for a single day.

    Memory-optimized version:
      - predictions stored as float32
      - preallocates (depth, lat, lon) and fills in lat-chunks

    Parameters
    ----------
    R : xr.DataArray
        Rrs on (lat, lon, wavelength). No time dimension.
    brt_models : dict
        Mapping depth-label -> fitted model, e.g.
        {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}.
        The last two underscore-separated tokens are assumed to be
        depth start/end in meters, e.g. "CHLA_0_10" -> 0, 10.
    feature_cols : list of str
        Columns expected by the BRT models. The non-constant subset of these
        must align with the wavelength dimension of R.
    consts : dict, optional
        Feature -> scalar value for constant features
        (e.g. {"solar_hour": 0, "type": 1}).
    chunk_size_lat : int
        Number of latitude indices per chunk.
    time : str or np.datetime64, optional
        If provided, a `time` dimension of length 1 is added to the output.
    z : array-like, optional
        Depth centers (same order as brt_models keys). If not given, centers are
        inferred as (z_start + z_end)/2 from the model name.
    z_name : str, default "z"
        Name of the vertical dimension in the output.
    silent : bool, default False
        Placeholder flag for compatibility; currently not used.

    Returns
    -------
    xr.DataArray
        CHLA prediction with dims:
            (time, z_name, lat, lon)      if `time` provided
            (z_name, lat, lon)           otherwise

        Coordinates:
            z_name             : depth center (m)
            f"{z_name}_start"  : depth bin lower bound (m)
            f"{z_name}_end"    : depth bin upper bound (m)
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    consts = consts or {}

    # Make sure dims are in the expected order
    R = R.transpose("lat", "lon", "wavelength")

    depth_labels = list(brt_models.keys())
    n_depth = len(depth_labels)

    # ---- parse z_start / z_end / z_center from labels like ABC_0_10 ----
    z_start_arr = np.full(n_depth, np.nan, dtype="float32")
    z_end_arr   = np.full(n_depth, np.nan, dtype="float32")
    z_center_arr = np.full(n_depth, np.nan, dtype="float32")

    for i, label in enumerate(depth_labels):
        parts = label.split("_")
        if len(parts) >= 3:
            try:
                z0 = float(parts[-2])
                z1 = float(parts[-1])
                z_start_arr[i] = z0
                z_end_arr[i]   = z1
                z_center_arr[i] = 0.5 * (z0 + z1)
            except ValueError:
                # leave as NaN if parsing fails
                pass

    # override z centers if explicitly provided
    if z is not None:
        z_center_arr = np.asarray(z, dtype="float32")
        if z_center_arr.shape[0] != n_depth:
            raise ValueError(f"len(z)={len(z_center_arr)} does not match number of models={n_depth}")

    nlat = R.sizes["lat"]
    nlon = R.sizes["lon"]
    lat_coord = R["lat"]
    lon_coord = R["lon"]

    # ------- non-constant features must match wavelength axis -------
    non_constant_cols = [c for c in feature_cols if c not in consts]
    nwave = R.sizes["wavelength"]

    if len(non_constant_cols) != nwave:
        raise ValueError(
            f"Number of non-constant features ({len(non_constant_cols)}) "
            f"does not match wavelength dimension ({nwave}).\n"
            f"Non-constant cols: {non_constant_cols}"
        )

    # Check that wavelengths encoded in feature_cols match R["wavelength"]
    try:
        wl_from_cols = np.array(
            [float(col.rsplit("_", 1)[-1]) for col in non_constant_cols],
            dtype=float,
        )
    except ValueError as e:
        raise ValueError(
            "Could not parse wavelengths from feature_cols. "
            "Expected names like 'pace_Rrs_346', 'pace_Rrs_348', etc. "
            f"Got non-constant_cols={non_constant_cols[:5]}..."
        ) from e

    wl_R = np.asarray(R["wavelength"].values, dtype=float)

    if wl_from_cols.shape[0] != wl_R.shape[0] or not np.allclose(wl_from_cols, wl_R, atol=0.01):
        raise ValueError(
            "Mismatch between wavelengths implied by feature_cols and the "
            "R['wavelength'] coordinate.\n"
            f"First few from feature_cols: {wl_from_cols[:5]}\n"
            f"First few from R.wavelength: {wl_R[:5]}"
        )

    # -------- preallocate output array: (depth, lat, lon) as float32 --------
    pred_all_arr = np.full(
        (n_depth, nlat, nlon),
        np.nan,
        dtype=np.float32,
    )

    # ---- chunk over latitude to avoid loading full globe into memory ----
    for start in range(0, nlat, chunk_size_lat):
        if not silent:
            print(f"Starting {start} of {nlat}")

        stop = min(start + chunk_size_lat, nlat)
        R_chunk = R.isel(lat=slice(start, stop))  # (lat_chunk, lon, wavelength)

        # 1. stack lat/lon → pixel
        R2 = R_chunk.stack(pixel=("lat", "lon")).transpose("pixel", "wavelength")
        R2_vals = R2.values  # (n_pixel, n_wavelength)

        # 2. base DataFrame for all models (non-constant features)
        df_base = pd.DataFrame(R2_vals, columns=non_constant_cols)

        # 3. For each depth-model, add constants, filter NaNs, predict, reshape
        for d_idx, depth_label in enumerate(depth_labels):
            model = brt_models[depth_label]

            # Start from the base spectral predictors
            df_pred = df_base.copy()

            # Add constant columns that are actually in feature_cols
            for name, value in consts.items():
                if name in feature_cols:
                    df_pred[name] = value

            # Ensure columns are in the correct order expected by the model
            df_pred = df_pred[feature_cols]

            # Handle NaNs: keep only pixels with complete predictors
            valid_mask = ~df_pred.isna().any(axis=1)
            df_valid = df_pred[valid_mask]

            # Prepare flat prediction array for this lat-chunk (float32)
            y_pred_flat = np.full(df_pred.shape[0], np.nan, dtype=np.float32)

            if len(df_valid) > 0:
                # model.predict may return float64; cast to float32
                y_pred_flat[valid_mask.values] = model.predict(df_valid).astype(np.float32)

            # Reshape back to (lat_chunk, lon)
            nlat_chunk = R_chunk.sizes["lat"]
            y_pred_map = y_pred_flat.reshape(nlat_chunk, nlon)

            # Fill into the preallocated array
            pred_all_arr[d_idx, start:stop, :] = y_pred_map

    if not silent:
        print(f"Starting wrapping")

    # ---- wrap preallocated array into an xarray.DataArray ----
    pred_all = xr.DataArray(
        pred_all_arr,
        coords={
            z_name: z_center_arr,
            "lat": lat_coord,
            "lon": lon_coord,
        },
        dims=(z_name, "lat", "lon"),
        name="CHLA",
    )

    # vertical coordinates
    if not silent:
        print(f"Adding coords")
    pred_all = pred_all.assign_coords(
        {
            z_name: z_center_arr,
            f"{z_name}_start": (z_name, z_start_arr),
            f"{z_name}_end":   (z_name, z_end_arr),
        }
    )

    # optional time dimension
    if time is not None:
        if not silent:
            print(f"Adding time")
        time_val = np.datetime64(time)
        pred_all = pred_all.expand_dims(time=[time_val])

    pred_all.attrs.setdefault(
        "depth_info",
        f"Depth coordinates inferred from brt_models keys of form 'NAME_z0_z1'. "
        f"{z_name} is the bin center; {z_name}_start/{z_name}_end are bin bounds (m)."
    )

    return pred_all

def OLD2_predict_all_depths_for_day(
    R: xr.DataArray,         # (lat, lon, wavelength)
    brt_models: dict,        # e.g. {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}
    feature_cols: list,
    consts=None,             # e.g. {"solar_hour": 0, "type": 1}
    chunk_size_lat: int = 100,
    time=None,               # e.g. "2024-07-15" or np.datetime64
    z: np.ndarray | None = None,   # optional override for depth centers
    z_name: str = "z",       # vertical dimension name
    silent=True # don't print progress
):
    """
    Run BRT predictions for all depth bins for a single day, using the same
    logic as `make_prediction()` but looping over multiple depth-specific models.

    Parameters
    ----------
    R : xr.DataArray
        Rrs on (lat, lon, wavelength). No time dimension.
    brt_models : dict
        Mapping depth-label -> fitted model, e.g.
        {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}.
        The last two underscore-separated tokens are assumed to be
        depth start/end in meters, e.g. "CHLA_0_10" -> 0, 10.
    feature_cols : list of str
        Columns expected by the BRT models. The non-constant subset of these
        must align with the wavelength dimension of R.
    consts : dict, optional
        Feature -> scalar value for constant features
        (e.g. {"solar_hour": 0, "type": 1}).
    chunk_size_lat : int
        Number of latitude indices per chunk.
    time : str or np.datetime64, optional
        If provided, a `time` dimension of length 1 is added to the output.
    z : array-like, optional
        Depth centers (same order as brt_models keys). If not given, centers are
        inferred as (z_start + z_end)/2 from the model name.
    z_name : str, default "z"
        Name of the vertical dimension in the output.

    Returns
    -------
    xr.DataArray
        CHLA prediction with dims:
            (time, z_name, lat, lon)      if `time` provided
            (z_name, lat, lon)           otherwise

        Coordinates:
            z_name             : depth center (m)
            f"{z_name}_start"  : depth bin lower bound (m)
            f"{z_name}_end"    : depth bin upper bound (m)
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    consts = consts or {}

    # Make sure dims are in the expected order
    R = R.transpose("lat", "lon", "wavelength")

    depth_labels = list(brt_models.keys())
    n_depth = len(depth_labels)

    # ---- parse z_start / z_end / z_center from labels like ABC_0_10 ----
    z_start_arr = np.full(n_depth, np.nan, dtype="float32")
    z_end_arr   = np.full(n_depth, np.nan, dtype="float32")
    z_center_arr = np.full(n_depth, np.nan, dtype="float32")

    for i, label in enumerate(depth_labels):
        parts = label.split("_")
        if len(parts) >= 3:
            try:
                z0 = float(parts[-2])
                z1 = float(parts[-1])
                z_start_arr[i] = z0
                z_end_arr[i]   = z1
                z_center_arr[i] = 0.5 * (z0 + z1)
            except ValueError:
                # leave as NaN if parsing fails
                pass

    # override z centers if explicitly provided
    if z is not None:
        z_center_arr = np.asarray(z, dtype="float32")
        if z_center_arr.shape[0] != n_depth:
            raise ValueError(f"len(z)={len(z_center_arr)} does not match number of models={n_depth}")

    nlat = R.sizes["lat"]
    nlon = R.sizes["lon"]
    lat_coord = R["lat"]
    lon_coord = R["lon"]

    # ------- non-constant features must match wavelength axis -------
    non_constant_cols = [c for c in feature_cols if c not in consts]
    nwave = R.sizes["wavelength"]

    if len(non_constant_cols) != nwave:
        raise ValueError(
            f"Number of non-constant features ({len(non_constant_cols)}) "
            f"does not match wavelength dimension ({nwave}).\n"
            f"Non-constant cols: {non_constant_cols}"
        )

    # New: check that the wavelengths encoded in feature_cols match R["wavelength"]
    # Assumes feature names end in the wavelength, e.g. "pace_Rrs_346"
    try:
        wl_from_cols = np.array(
            [float(col.rsplit("_", 1)[-1]) for col in non_constant_cols],
            dtype=float,
        )
    except ValueError as e:
        raise ValueError(
            "Could not parse wavelengths from feature_cols. "
            "Expected names like 'pace_Rrs_346', 'pace_Rrs_348', etc. "
            f"Got non-constant_cols={non_constant_cols[:5]}..."
        ) from e

    wl_R = np.asarray(R["wavelength"].values, dtype=float)

    if wl_from_cols.shape[0] != wl_R.shape[0] or not np.allclose(wl_from_cols, wl_R, atol=0.01):
        raise ValueError(
            "Mismatch between wavelengths implied by feature_cols and the "
            "R['wavelength'] coordinate.\n"
            f"First few from feature_cols: {wl_from_cols[:5]}\n"
            f"First few from R.wavelength: {wl_R[:5]}"
        )

    # collect chunks over depth
    depth_chunks = {label: [] for label in depth_labels}

    # ---- chunk over latitude to avoid loading full globe into memory ----
    for start in range(0, nlat, chunk_size_lat):
        if not silent:
            print(f"Starting {start} of {nlat}")
        stop = min(start + chunk_size_lat, nlat)
        R_chunk = R.isel(lat=slice(start, stop))  # (lat_chunk, lon, wavelength)

        # 1. stack lat/lon → pixel
        R2 = R_chunk.stack(pixel=("lat", "lon")).transpose("pixel", "wavelength")
        R2_vals = R2.values  # (n_pixel, n_wavelength)

        # 2. base DataFrame for all models (non-constant features)
        df_base = pd.DataFrame(R2_vals, columns=non_constant_cols)

        # 3. For each depth-model, add constants, filter NaNs, predict, reshape
        for depth_label, model in brt_models.items():
            # Start from the base spectral predictors
            df_pred = df_base.copy()

            # Add constant columns that are actually in feature_cols
            for name, value in consts.items():
                if name in feature_cols:
                    df_pred[name] = value

            # Ensure columns are in the correct order expected by the model
            df_pred = df_pred[feature_cols]

            # Handle NaNs: keep only pixels with complete predictors
            valid_mask = ~df_pred.isna().any(axis=1)
            df_valid = df_pred[valid_mask]

            # Prepare flat prediction array for this lat-chunk
            y_pred_flat = np.full(df_pred.shape[0], np.nan, dtype=float)

            if len(df_valid) > 0:
                y_pred_flat[valid_mask.values] = model.predict(df_valid)

            # Reshape back to (lat_chunk, lon)
            nlat_chunk = R_chunk.sizes["lat"]
            y_pred_map = y_pred_flat.reshape(nlat_chunk, nlon)

            da_chunk = xr.DataArray(
                y_pred_map,
                coords={"lat": R_chunk["lat"], "lon": lon_coord},
                dims=("lat", "lon"),
                name="CHLA",
            )
            depth_chunks[depth_label].append(da_chunk)

    # ---- stitch each depth over lat, then stack into vertical dimension ----
    if not silent:
        print("Stitching together")
    per_depth = []
    for idx, depth_label in enumerate(depth_labels):
        chunks = depth_chunks[depth_label]
        da = xr.concat(chunks, dim="lat").assign_coords(lat=lat_coord)
        per_depth.append(da.expand_dims({z_name: [idx]}))

    if not silent:
        print("Concatinating together")
    pred_all = xr.concat(per_depth, dim=z_name)  # (z, lat, lon)
    pred_all.name = "CHLA"

    # vertical coordinates
    if not silent:
        print("Add coords")
    pred_all = pred_all.assign_coords(
        {
            z_name: z_center_arr,
            f"{z_name}_start": (z_name, z_start_arr),
            f"{z_name}_end":   (z_name, z_end_arr),
        }
    )

    # optional time dimension
    if time is not None:
        time_val = np.datetime64(time)
        pred_all = pred_all.expand_dims(time=[time_val])

    pred_all.attrs.setdefault(
        "depth_info",
        f"Depth coordinates inferred from brt_models keys of form 'NAME_z0_z1'. "
        f"{z_name} is the bin center; {z_name}_start/{z_name}_end are bin bounds (m)."
    )

    return pred_all


# Combine all BRTs for each depth and make a xr.Dataset of predictions
def OLD_predict_all_depths_for_day(
    R: xr.DataArray,         # (lat, lon, wavelength)
    brt_models: dict,        # e.g. {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}
    feature_cols: list,
    consts=None,
    chunk_size_lat: int = 100,
    time=None,               # e.g. "2024-07-15" or np.datetime64
    z: np.ndarray | None = None,   # optional override for depth centers
    z_name: str = "z",       # vertical dimension name
):
    """
    Run BRT predictions for all depth bins for a single day.

    Parameters
    ----------
    R : xr.DataArray
        Predictor array of Rrs wavelengths from PACE on (lat, lon, wavelength) (no time dimension).
    brt_models : dict
        Mapping depth-label -> fitted model, e.g.
        {"CHLA_0_10": model0, "CHLA_10_20": model1, ...}.
        The last two underscore-separated tokens are assumed to be
        depth start/end in meters, e.g. "CHLA_0_10" -> 0, 10.
    feature_cols : list of str
        Columns expected by the BRT models.
    consts : dict, optional
        Feature -> scalar value for constants (e.g. {"solar_hour": 12.0, "type": 1}).
    chunk_size_lat : int
        Number of latitude indices per chunk.
    time : str or np.datetime64, optional
        Time stamp for this prediction. If provided, a `time` dimension of length 1
        is added to the output.
    z : array-like, optional
        Depth centers (same order as brt_models keys). If not given, centers are
        inferred as (z_start + z_end)/2 from the model name.
    z_name : str, default "z"
        Name of the vertical dimension in the output.

    Returns
    -------
    xr.DataArray
        CHLA prediction with dims:
            (time, z_name, lat, lon)      if `time` provided
            (z_name, lat, lon)           otherwise

        Coordinates:
            z_name         : depth center (m)
            f"{z_name}_start" : depth bin lower bound (m)
            f"{z_name}_end"   : depth bin upper bound (m)
    """
    import numpy as np
    import xarray as xr

    consts = consts or {}
    R = R.transpose("lat", "lon", "wavelength")

    depth_labels = list(brt_models.keys())
    n_depth = len(depth_labels)

    # --- parse z_start / z_end / z_center from labels like ABC_0_10 ---
    z_start_arr = np.full(n_depth, np.nan, dtype="float32")
    z_end_arr   = np.full(n_depth, np.nan, dtype="float32")
    z_center_arr = np.full(n_depth, np.nan, dtype="float32")

    for i, label in enumerate(depth_labels):
        parts = label.split("_")
        if len(parts) >= 3:
            try:
                z0 = float(parts[-2])
                z1 = float(parts[-1])
                z_start_arr[i] = z0
                z_end_arr[i]   = z1
                z_center_arr[i] = 0.5 * (z0 + z1)
            except ValueError:
                # leave as NaN if parsing fails
                pass

    # if user provided z, override centers
    if z is not None:
        z_center_arr = np.asarray(z, dtype="float32")
        if z_center_arr.shape[0] != n_depth:
            raise ValueError(f"len(z)={len(z_center_arr)} does not match number of models={n_depth}")

    nlat = R.sizes["lat"]
    lat_coord = R["lat"]

    depth_chunks = {label: [] for label in depth_labels}

    # --- chunk over latitude ---
    for start in range(0, nlat, chunk_size_lat):
        stop = min(start + chunk_size_lat, nlat)
        R_chunk = R.isel(lat=slice(start, stop))

        for label, model in brt_models.items():
            pred_chunk = make_prediction_brt(
                R_chunk,
                brt_model=model,
                feature_cols=feature_cols,
                consts=consts,
            )
            depth_chunks[label].append(pred_chunk)

    # --- stitch each depth over lat, then stack into vertical dimension ---
    per_depth = []
    for idx, (label, chunks) in enumerate(depth_chunks.items()):
        da = xr.concat(chunks, dim="lat").assign_coords(lat=lat_coord)
        per_depth.append(da.expand_dims({z_name: [idx]}))

    pred_all = xr.concat(per_depth, dim=z_name)  # (z, lat, lon)
    pred_all.name = "CHLA"

    # vertical coordinates
    pred_all = pred_all.assign_coords(
        {
            z_name: z_center_arr,
            f"{z_name}_start": (z_name, z_start_arr),
            f"{z_name}_end":   (z_name, z_end_arr),
        }
    )

    # optional time dimension
    if time is not None:
        time_val = np.datetime64(time)
        pred_all = pred_all.expand_dims(time=[time_val])

    # note about depth inference
    pred_all.attrs.setdefault(
        "depth_info",
        f"Depth coordinates inferred from brt_models keys of form 'NAME_z0_z1'. "
        f"z is the bin center, {z_name}_start/{z_name}_end are bin bounds (m)."
    )

    return pred_all

# ---- Plot predictions

def make_plot_pred_map(
    da,
    pred_label: str = "Prediction",
    cmap_pred: str = "viridis",
    time=None,
    z=None,
    z_start=None,
    z_dim: str = "z",
):
    """
    Plot a prediction slice from an xarray DataArray on a lat/lon grid.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dims:
          - (time, z, lat, lon) or
          - (z, lat, lon) or
          - (lat, lon)
    pred_label : str
        Title and colorbar label.
    cmap_pred : str
        Matplotlib colormap name.
    time :
        Optional time selector. If None and 'time' is a dimension, the first time is used.
    z :
        Optional depth-center selector along `z_dim`. If provided, uses .sel(z_dim=z, method="nearest"). If None, the first z is used.
    z_start :
        Optional lower-bound depth (e.g. 10 for a 10–20 m bin). Only used if
        `z_start` is present as a coordinate and `z` is not given. Selects the
        index whose z_start is closest to this value.
    z_dim : str
        Name of the vertical dimension (default "z").
    """
    # Local imports so this helper is self-contained in the bundle
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    sel = da

    # --- Handle time selection ---
    if "time" in sel.dims:
        if time is None:
            time_val = sel["time"].values[0]
        else:
            time_val = time
        sel = sel.sel(time=time_val)

    # --- Handle vertical selection ---
    if z_dim in sel.dims:
        if z is not None:
            sel = sel.sel({z_dim: z}, method="nearest")
        elif z_start is not None and "z_start" in sel.coords:
            z_start_vals = sel["z_start"].values
            idx = int(np.argmin(np.abs(z_start_vals - z_start)))
            sel = sel.isel({z_dim: idx})
        else:
            sel = sel.isel({z_dim: 0})

    if not (("lat" in sel.dims) and ("lon" in sel.dims)):
        raise ValueError(
            "make_plot_pred_map expects the final slice to have 'lat' and 'lon' dimensions."
        )

    vals = sel.values.ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError("Selected slice is all-NaN; nothing to plot.")

    vmin, vmax = np.nanpercentile(vals, (2, 98))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="0.9")

    im = ax.pcolormesh(
        sel["lon"],
        sel["lat"],
        sel,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_pred,
    )

    # Build a more informative title
    title = pred_label
    pieces = []

    if "time" in da.dims:
        t_val = np.asarray(sel.coords["time"].values).item()
        pieces.append(f"time={np.datetime_as_string(t_val, unit='D')}")

    if z_dim in da.dims:
        if z is not None:
            pieces.append(f"{z_dim}≈{float(sel[z_dim].values):.1f} m")
        elif z_start is not None and "z_start" in sel.coords:
            z0 = float(sel["z_start"].values)
            z1 = float(sel.coords.get("z_end", np.nan))
            if np.isfinite(z1):
                pieces.append(f"{z0:.1f}–{z1:.1f} m")
            else:
                pieces.append(f"{z0:.1f} m bin")
        else:
            zc = float(sel[z_dim].values)
            pieces.append(f"{zc:.1f} m")

    if pieces:
        title = f"{pred_label} ({', '.join(pieces)})"

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
    cbar.set_label(pred_label)

    plt.show()
    return fig, ax

    
## OLD

# Save and Load fitted model
import json, zipfile, tempfile
from pathlib import Path

def save_cnn_bundle(zip_path, model, X_mean, X_std, meta=None):
    """
    Create a single zip containing:
      - model.keras
      - stats.npz  (X_mean, X_std)
      - meta.json  (optional dict)
    """
    tf = _require_keras() 
    
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1) Save model in Keras native format
        model_path = tmp / "model.keras"
        model.save(model_path)
        # 2) Save stats
        np.savez(tmp / "stats.npz", X_mean=X_mean, X_std=X_std)
        # 3) Save meta
        (tmp / "meta.json").write_text(json.dumps(meta or {}))

        # 4) Zip it up
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, arcname="model.keras")
            z.write(tmp / "stats.npz", arcname="stats.npz")
            z.write(tmp / "meta.json", arcname="meta.json")

    return str(zip_path)

def load_cnn_bundle(zip_path, compile=False):
    """
    Load a bundle produced by save_inference_zip().
    Returns: (model, X_mean, X_std, meta_dict)
    """
    tf = _require_keras() 

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Extract all files
        z.extract("model.keras", path=tmp)
        z.extract("stats.npz",  path=tmp)
        # meta.json might be missing in older bundles; handle gracefully
        meta = {}
        if "meta.json" in z.namelist():
            z.extract("meta.json", path=tmp)
            meta = json.loads((tmp / "meta.json").read_text())

        # Load model & stats
        model = tf.keras.models.load_model(tmp / "model.keras", compile=compile)
        stats = np.load(tmp / "stats.npz")
        X_mean, X_std = stats["X_mean"], stats["X_std"]

    return model, X_mean, X_std, meta


## PLOTTING

import numpy as np
import matplotlib.pyplot as plt

def predict_and_plot_date(
    data_xr,
    date,                          # "YYYY-MM-DD" or np.datetime64
    model,
    num_var,                       # list of vars to normalize
    cat_var,                       # list of vars not normalized (e.g., ocean_mask, sin/cos time)
    X_mean, X_std,                 # per-channel stats for num_var (shape [len(num_var)])
    y_var="y",
    mask_var="ocean_mask",
    model_type="cnn",              # "cnn" or "tabular"
    cast_float32=True,
    use_percentiles=False, p_lo=5, p_hi=95,
    cmap="viridis"
):
    """
    Build one-sample input from dataset for a specific date, predict, and plot True vs Pred.
    Works with CNN (map→map) and tabular models (flattened pixels).
    """
    # ---- resolve date index
    date64 = np.datetime64(str(date))
    times = np.asarray(data_xr["time"].values)
    idxs = np.where(times == date64)[0]
    if idxs.size == 0:
        raise ValueError(f"Date {date} not found in dataset time coord.")
    t = int(idxs[0])

    # ---- helper to fetch a variable as (H,W) for that date; broadcast 2D to 3D if needed
    # choose a spatial template (first available among inputs or y)
    tmpl_name = (num_var + cat_var + [y_var])[0]
    tmpl = data_xr[tmpl_name]
    def fetch_2d(varname):
        arr = data_xr[varname]
        if "time" in arr.dims:
            arr_t = arr.isel(time=t)
        else:
            arr_t = arr
        arr_t = arr_t.broadcast_like(tmpl.isel(time=t))  # ensure same H,W
        a = arr_t.values
        if cast_float32:
            a = a.astype("float32", copy=False)
        return a  # (H,W)

    # ---- build channels for this date
    num_chans = []
    for k, vn in enumerate(num_var):
        a = fetch_2d(vn)
        if (X_mean is not None) and (X_std is not None):
            a = (a - X_mean[k]) / (1.0 if X_std[k] == 0 else X_std[k])
            a = np.nan_to_num(a)
        num_chans.append(a)

    cat_chans = []
    for vn in cat_var:
        a = fetch_2d(vn)
        a = np.nan_to_num(a)
        cat_chans.append(a)

    if not (num_chans or cat_chans):
        raise ValueError("No input variables provided.")

    # stack to (H,W,C)
    X_map = np.stack(num_chans + cat_chans, axis=-1)
    H, W, C = X_map.shape

    # ---- ground truth map
    y_true = fetch_2d(y_var)

    # ---- predict
    if model_type == "cnn":
        _ = _require_keras()  # ensure TF present only for cnn path 
        y_pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred[..., 0]
    elif model_type == "tabular":
        y_pred = model.predict(X_map.reshape(-1, C)).reshape(H, W)
    else:
        raise ValueError("model_type must be 'cnn' or 'tabular'.")

    # ---- mask land to NaN (mask_var==0 → land)
    land = (fetch_2d(mask_var) == 0.0)
    y_true = np.where(land, np.nan, y_true)
    y_pred = np.where(land, np.nan, y_pred)

    # ---- color limits
    if use_percentiles:
        stack = np.concatenate([y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_pred)]]) if np.isfinite(y_true).any() and np.isfinite(y_pred).any() else np.array([])
        vmin, vmax = (np.percentile(stack, p_lo), np.percentile(stack, p_hi)) if stack.size else (None, None)
    else:
        vmin = np.nanmin([y_true, y_pred]); vmax = np.nanmax([y_true, y_pred])

    # --- ensure North is up
    lat = np.array(data_xr.lat.values)
    flip_lat = lat[0] > lat[-1]   # True if lat is descending

    if flip_lat:
        y_true = np.flipud(y_true)
        y_pred = np.flipud(y_pred)

    # extent must be (xmin, xmax, ymin, ymax) with increasing y
    lon_min, lon_max = float(data_xr.lon.min()), float(data_xr.lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"True {y_var} — {np.datetime_as_string(date64)}"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"Predicted ({model_type.upper()})"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()
    return y_true, y_pred

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_true_vs_predicted_year_multi(
    data, year,
    models,                 # list of models
    X_mean, X_std,          # per-channel stats for num_var only
    num_var, cat_var,       # lists of variable names
    y_var="y",
    model_types="cnn",      # list like ['cnn','tabular', ...] same length as models
    model_names=None,       # optional names for column titles
    cmap='viridis',
    day=1,
    use_percentiles=True, p_lo=5, p_hi=95
):
    assert len(models) == len(model_types), "models and model_types must have same length"
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    ds = data.sel(time=year)

    # use day-th day of each month
    dates = pd.to_datetime(ds.time.values)
    df = pd.DataFrame({'date': dates, 'dom': dates.day, 'y': dates.year, 'm': dates.month})
    monthly_dates = pd.DatetimeIndex(
        df.groupby(['y','m']).apply(
            lambda g: g.loc[g.dom>=day, 'date'].min() if (g.dom>=day).any() else g['date'].max()
        ).sort_values().values
    )
    n_months = len(monthly_dates)

    lat = ds.lat.values
    lon = ds.lon.values
    flip_lat = lat[0] > lat[-1]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    land_mask = (ds["ocean_mask"].values == 0.0)

    # helper: fetch a 2D array for var at given date; broadcast if var has no time dim
    def fetch_2d(var, date):
        arr = ds[var]
        arr_t = arr.sel(time=date) if "time" in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[y_var].sel(time=date))
        a = arr_t.values.astype("float32", copy=False)
        return a

    # figure: True + one column per model
    ncols = 1 + len(models)
    fig, axs = plt.subplots(n_months, ncols, figsize=(3.2*ncols, 2.2*n_months), constrained_layout=True)
    if n_months == 1:
        axs = np.atleast_2d(axs)  # ensure 2D indexing

    for i, date in enumerate(monthly_dates):
        # Build (H,W,C) input for this date
        chans = []
        for k, v in enumerate(num_var):
            a = fetch_2d(v, date)
            if (X_mean is not None) and (X_std is not None):
                denom = 1.0 if X_std[k] == 0 else X_std[k]
                a = (a - X_mean[k]) / denom
                a = np.nan_to_num(a)
            chans.append(a)
        for v in cat_var:
            a = fetch_2d(v, date)
            chans.append(np.nan_to_num(a))
        X_map = np.stack(chans, axis=-1)
        H, W, C = X_map.shape

        # Truth
        truth = fetch_2d(y_var, date)

        # Predict with each model
        preds = []
        for mdl, mtype in zip(models, model_types):
            if mtype == "cnn":
                _ = _require_keras()  # ensure TF present only for cnn path
                yhat = mdl.predict(X_map[np.newaxis, ...], verbose=0)[0]
                if yhat.ndim == 3 and yhat.shape[-1] == 1:
                    yhat = yhat[..., 0]
            elif mtype == "tabular":
                yhat = mdl.predict(X_map.reshape(-1, C)).reshape(H, W)
            else:
                raise ValueError("model_type must be 'cnn' or 'tabular'.")
            preds.append(yhat)

        # Apply ocean mask and optional north-up flip
        truth_m = np.where(land_mask, np.nan, truth)
        preds_m = [np.where(land_mask, np.nan, p) for p in preds]
        if flip_lat:
            truth_m = np.flipud(truth_m)
            preds_m = [np.flipud(p) for p in preds_m]

        # Shared color limits per row
        all_maps = [truth_m] + preds_m
        if use_percentiles:
            stack = np.concatenate([m[np.isfinite(m)] for m in all_maps if np.isfinite(m).any()]) if any(np.isfinite(m).any() for m in all_maps) else np.array([])
            vmin, vmax = (np.percentile(stack, p_lo), np.percentile(stack, p_hi)) if stack.size else (None, None)
        else:
            vmin = np.nanmin(all_maps); vmax = np.nanmax(all_maps)

        # True panel
        ax = axs[i, 0]
        im = ax.imshow(truth_m, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_title(f"{date.strftime('%Y-%m-%d')} — True", fontsize=9)
        ax.axis('off')

        # Prediction panels with metrics
        for j, (pmap, name) in enumerate(zip(preds_m, model_names), start=1):
            axp = axs[i, j]
            im = axp.imshow(pmap, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
            axp.axis('off')

            # metrics (mask NaNs)
            m = np.isfinite(truth_m) & np.isfinite(pmap)
            if m.any():
                r2 = r2_score(truth_m[m].ravel(), pmap[m].ravel())
                rmse = np.sqrt(np.mean((truth_m[m] - pmap[m])**2))
                axp.set_title(f"{name}\n$R^2$={r2:.2f}, RMSE={rmse:.2f}", fontsize=9)
            else:
                axp.set_title(f"{name}\nno valid pixels", fontsize=9)

    # one colorbar for the last column
    # cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    # fig.colorbar(im, cax=cax, label=y_var)
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import calendar

def plot_metric_by_month(
    data, years, model, X_mean, X_std, num_var, cat_var,
    training_year=None, metric='r2',
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    ymin=None, ymax=None,
    model_type="cnn",
):
    """
    Plot a single monthly performance metric for one model across multiple years.

    This matches the batching/branching style used in `plot_4metric_by_month`:
    it supports CNN models (batched (H,W,C) → (H,W) predictions) and non-CNN
    “tabular/BRT”-style models (per-day (H*W,C) → (H*W) predictions). If
    `model_type == "cnn"`, TensorFlow/Keras is required and is imported lazily
    via `_require_keras()`.

    For each `year`:
      1) Select one representative day per month (the first day present in that
         month).                                   ── same behavior as your original
      2) Build an (H, W, C) feature map by stacking:
         - numerical vars in `num_var`, normalized with (`X_mean`, `X_std`)
           if provided (0 std → no scaling),
         - categorical/aux vars in `cat_var` (no normalization).
      3) Predict for that day:
         - CNN: `model.predict(X_map[None, ...]) → (1,H,W[,1])`.
         - BRT/Tabular: reshape (H,W,C) → (H*W,C), predict, reshape back to (H,W).
      4) Mask land where `mask_var == 0.0` and compute the requested metric:
         - 'r2'  : coefficient of determination over valid pixels
         - 'rmse': root mean squared error
         - 'mae' : mean absolute error
         - 'bias': mean(pred - truth)
         - 'ssim': structural similarity (with optional `ssim_win_size` and
                   Gaussian weighting via `ssim_sigma`). NaNs are filled with
                   per-image means only for SSIM computation.
      5) Plot the monthly metric values with an optional dashed style for the
         `training_year`. `ymin`/`ymax` set common y-limits if provided.

    Parameters
    ----------
    data : xr.Dataset
        Contains `y_name`, `mask_var`, and all variables in `num_var`/`cat_var`,
        with a `time` dimension and (lat, lon) grid.
    years : sequence[int]
        Years to evaluate (e.g., [2019, 2020]).
    model : object
        - If `model_type == "cnn"`: a tf.keras.Model returning (B,H,W[,1]).
        - If `model_type in {"brt","tabular"}`: an estimator with
          `.predict(X_2d)` producing length n_samples predictions.
    X_mean, X_std : array-like or None
        Per-channel stats for numerical variables in `num_var` (len == len(num_var)).
        If None, numerical inputs are not normalized.
    num_var, cat_var : list[str]
        Names of numerical and categorical/aux variables to stack as channels.
    training_year : int or None
        If provided, that year's line is dashed and labeled "(train)".
    metric : {'r2','rmse','mae','bias','ssim'}, default 'r2'
        Which metric to plot per month.
    y_name : str, default 'y'
        Target variable in `data`.
    mask_var : str, default 'ocean_mask'
        Land/ocean mask; pixels with 0.0 are treated as land and masked.
    ssim_win_size : int or None
        SSIM window size (must be odd if provided).
    ssim_sigma : float or None
        If provided, enables Gaussian-weighted SSIM with this sigma.
    ymin, ymax : float or None
        Common y-limits for the plot. If None, Matplotlib defaults are used.
    model_type : {'cnn','brt','tabular'}, default 'cnn'
        Selects the prediction pathway.

    Returns
    -------
    None
        Displays a Matplotlib figure: month (1–12) on x-axis, the chosen metric
        on y-axis, with one line per year.

    Notes
    -----
    - Only one representative day per month is used (first available day).
    - For SSIM, NaNs are filled (just for the SSIM call) with each image's mean;
      `data_range` is derived from the truth field when possible.
    """
    assert metric in ['r2', 'rmse', 'mae', 'bias', 'ssim']

    if model_type == "cnn":
        _ = _require_keras()  # only require TF/Keras for CNN

    def fetch_2d(ds, var, date, like_var):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[like_var].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat','lon'):
            arr_t = arr_t.transpose(..., 'lat', 'lon') if 'time' in arr_t.dims else arr_t.transpose('lat','lon')
        return arr_t.values.astype('float32', copy=False)
        
    metric_by_year_month = {}

    for year in years:
        ds = data.sel(time=year)
        dates = pd.to_datetime(ds.time.values)
        monthly_dates = (
            pd.Series(dates).groupby([dates.year, dates.month]).min().sort_values()
        )

        scores = []
        for date in monthly_dates:
            # build (H,W,C) input for this date
            chans = []
            for k, v in enumerate(num_var):
                a = fetch_2d(ds, v, date, y_name)
                if X_mean is not None and X_std is not None:
                    denom = 1.0 if X_std[k] == 0 else X_std[k]
                    a = (a - X_mean[k]) / denom
                chans.append(np.nan_to_num(a))
            for v in cat_var:
                a = fetch_2d(ds, v, date, y_name)
                chans.append(np.nan_to_num(a))
            X_map = np.stack(chans, axis=-1)

            # predict (branch like plot_4metric_by_month)
            if model_type == "cnn":
                pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred[..., 0]
            elif model_type in ("brt", "tabular"):
                H, W, C = X_map.shape
                pred = model.predict(X_map.reshape(-1, C)).reshape(H, W)
            else:
                raise ValueError("model_type must be 'cnn' or 'brt'/'tabular'.")

            # truth & mask
            truth = fetch_2d(ds, y_name, date, y_name)
            land = (fetch_2d(ds, mask_var, date, y_name) == 0.0)
            pred  = np.where(land, np.nan, pred)
            truth = np.where(land, np.nan, truth)

            # metric
            if metric == 'ssim':
                # fill NaNs for SSIM computation
                t = np.nan_to_num(truth, nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0))
                p = np.nan_to_num(pred,  nan=(np.nanmean(pred)  if np.isfinite(pred).any()  else 0.0))
                # robust data_range
                dr = np.nanmax(truth) - np.nanmin(truth)
                if not np.isfinite(dr) or dr == 0:
                    dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
                # build kwargs safely (don’t pass sigma=None)
                ssim_kwargs = {"data_range": dr}
                if ssim_win_size is not None:
                    ssim_kwargs["win_size"] = int(ssim_win_size)  # must be odd
                if ssim_sigma is not None:
                    ssim_kwargs["gaussian_weights"] = True
                    ssim_kwargs["sigma"] = float(ssim_sigma)
                score = ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)
            else:
                m = ~np.isnan(truth) & ~np.isnan(pred)
                if not m.any():
                    score = np.nan
                elif metric == 'r2':
                    score = r2_score(truth[m].ravel(), pred[m].ravel())
                elif metric == 'rmse':
                    score = float(np.sqrt(np.mean((truth[m] - pred[m])**2)))
                elif metric == 'mae':
                    score = float(mean_absolute_error(truth[m], pred[m]))
                elif metric == 'bias':
                    score = float(np.mean(pred[m] - truth[m]))

            scores.append(score)

        metric_by_year_month[year] = (monthly_dates.dt.month.values, scores)

    # plot
    plt.figure(figsize=(10,5))
    for year, (months, scores) in metric_by_year_month.items():
        label = f"{year} (train)" if year == training_year else year
        style = "--" if year == training_year else "-"
        plt.plot(months, scores, style, marker='o', label=label)

    plt.xlabel("Month")
    plt.ylabel({'r2':"$R^2$",'rmse':"RMSE",'mae':"MAE",'bias':"Bias",'ssim':"SSIM"}[metric])
    plt.title(f"Monthly {metric.upper()} by Year")
    plt.xticks(np.arange(1,13), calendar.month_abbr[1:13])
    plt.legend(); plt.grid(True); plt.tight_layout()
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)
    plt.show()


def plot_4metric_by_month(
    data, years, model, X_mean, X_std, num_var, cat_var,
    training_year=None,
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    ymin=None, ymax=None,
    model_type="cnn",
):
    """
    Compute and plot monthly performance metrics (R², bias, MAE, SSIM) over
    selected days in each month (1, 7, 14, 28) for either a CNN or a BRT model.

    For each year in ``years``:
      1. Select timesteps from ``data`` within that year.
      2. Group timesteps by (year, month).
      3. Within each month, select dates whose day-of-month is in {1, 7, 14, 28}.
      4. For each selected date, build an (H, W, C) feature map using numerical
         variables (``num_var``) and categorical variables (``cat_var``). If
         ``X_mean``/``X_std`` are provided, numerical variables are normalized.
      5. Run the model to obtain predictions for each (H, W) field:
           - If ``model_type == "cnn"``:
               * Stack daily inputs into (B, H, W, C) and call
                 ``model.predict(X_batch)``.
               * Output is expected as (B, H, W) or (B, H, W, 1).
           - If ``model_type == "brt"``:
               * For each day, reshape (H, W, C) → (H*W, C),
                 call ``model.predict(X_flat)``, then reshape predictions
                 back to (H, W).
      6. For each day, mask land using ``mask_var``, drop NaNs and compute:
           * R²
           * Bias (pred - truth)
           * MAE
           * SSIM (with optional window size and Gaussian sigma)
      7. Average daily metrics within each month (via ``np.nanmean``) to obtain
         a monthly value.
      8. Produce a 2×2 panel plot of monthly metrics across all years, with an
         optional highlight for ``training_year``.

    Months that do not contain any of the target days {1, 7, 14, 28} are skipped.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the target and predictor variables.
        Must have a ``time`` dimension and at least the variables in
        ``num_var``, ``cat_var``, ``y_name`` and ``mask_var``.
    years : sequence of int
        Years to evaluate (used with ``data.sel(time=year)``).
    model : object
        - If ``model_type == "cnn"``: a tf.keras.Model (or compatible) that
          accepts (B, H, W, C) and returns (B, H, W) or (B, H, W, 1).
        - If ``model_type == "brt"``: a scikit-learn-like regressor with
          ``predict(X_2d)`` where X_2d has shape (n_samples, n_features).
    X_mean : array-like of float or None
        Per-channel means for numerical variables in ``num_var``.
        Length must match ``len(num_var)`` if not None. If None, no mean/std
        normalization is applied.
    X_std : array-like of float or None
        Per-channel standard deviations for numerical variables in ``num_var``.
        Length must match ``len(num_var)`` if not None. If a std is zero, that
        channel is left unscaled. If None, no mean/std normalization is applied.
    num_var : list of str
        Names of numerical predictor variables in ``data``. Each is fetched,
        broadcast to the target grid, optionally normalized, and stacked as a
        channel.
    cat_var : list of str
        Names of categorical / non-normalized predictor variables in ``data``.
        Each is fetched, broadcast to the target grid, and stacked as a channel
        (no mean/std normalization).
    training_year : int, optional
        Year used for training. If provided, that year's line is dashed and
        labeled "(train)".
    y_name : str, default "y"
        Name of the target variable in ``data``.
    mask_var : str, default "ocean_mask"
        Name of the land/ocean mask variable. Values equal to 0.0 are treated
        as land and masked out.
    ssim_win_size : int, optional
        Window size for SSIM. Must be odd if provided.
    ssim_sigma : float, optional
        Standard deviation for Gaussian-weighted SSIM. If provided, Gaussian
        weights are used.
    ymin, ymax : float, optional
        Common y-limits for all metric subplots. If None, matplotlib defaults.
    model_type : {"cnn", "brt"}, default "cnn"
        Type of model:
          - "cnn": use batched (B, H, W, C) predictions.
          - "brt": predict on flattened features per day and reshape back.

    Notes
    -----
    - SSIM is computed on fields where land is masked and NaNs are filled with
      the mean of valid values for that day (or 0 if none).
    - Daily metric values within a month are averaged via ``np.nanmean``.
    - The resulting figure shows four panels (R², Bias, MAE, SSIM) with month
      on the x-axis (1–12) and one line per year.
    """
    # helper
    def fetch_2d(ds, var, date, like_var):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[like_var].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat', 'lon'):
            arr_t = (
                arr_t.transpose(..., 'lat', 'lon')
                if 'time' in arr_t.dims
                else arr_t.transpose('lat', 'lon')
            )
        return arr_t.values.astype('float32', copy=False)

    # target days within each month
    target_days = {1, 7, 14, 28}

    metrics = ['r2', 'bias', 'mae', 'ssim']
    metric_by_year_month = {m: {} for m in metrics}

    for year in years:
        ds = data.sel(time=year)
        dates = pd.to_datetime(ds.time.values)

        gb = pd.Series(dates).groupby([dates.year, dates.month])

        scores_dict = {m: [] for m in metrics}
        months_list = []

        for (yy, mm), group in gb:
            month_dates = list(group)
            pick = [d for d in month_dates if d.day in target_days]
            if len(pick) == 0:
                continue

            chans_list = []
            truths = []
            lands = []
            pred_list = []

            for date in pick:
                chans = []
                for k, v in enumerate(num_var):
                    a = fetch_2d(ds, v, date, y_name)
                    if X_mean is not None and X_std is not None:
                        denom = 1.0 if X_std[k] == 0 else X_std[k]
                        a = (a - X_mean[k]) / denom
                    chans.append(np.nan_to_num(a))
                for v in cat_var:
                    a = fetch_2d(ds, v, date, y_name)
                    chans.append(np.nan_to_num(a))

                X_map = np.stack(chans, axis=-1)  # (H, W, C)
                truths.append(fetch_2d(ds, y_name, date, y_name))
                lands.append(fetch_2d(ds, mask_var, date, y_name) == 0.0)

                if model_type == "cnn":
                    chans_list.append(X_map)
                elif model_type == "brt":
                    H, W, C = X_map.shape
                    X_flat = X_map.reshape(-1, C)
                    pred_flat = model.predict(X_flat)
                    pred_list.append(pred_flat.reshape(H, W))
                else:
                    raise ValueError(f"Unknown model_type: {model_type!r}")

            # ---- predict batch
            if model_type == "cnn":
                _ = _require_keras()  # ensure TF present only for cnn path
                X_batch = np.stack(chans_list, axis=0)  # (B, H, W, C)
                pred_batch = model.predict(X_batch, verbose=0)
                if pred_batch.ndim == 4 and pred_batch.shape[-1] == 1:
                    pred_batch = pred_batch[..., 0]  # (B, H, W)
            else:  # brt
                pred_batch = np.stack(pred_list, axis=0)  # (B, H, W)

            # ---- metrics
            r2_vals, bias_vals, mae_vals, ssim_vals = [], [], [], []
            for b, date in enumerate(pick):
                truth = np.where(lands[b], np.nan, truths[b])
                pred = np.where(lands[b], np.nan, pred_batch[b])

                m = ~np.isnan(truth) & ~np.isnan(pred)

                if m.any():
                    r2_vals.append(r2_score(truth[m].ravel(), pred[m].ravel()))
                    mae_vals.append(float(mean_absolute_error(truth[m], pred[m])))
                    bias_vals.append(float(np.mean(pred[m] - truth[m])))
                else:
                    r2_vals.append(np.nan)
                    mae_vals.append(np.nan)
                    bias_vals.append(np.nan)

                t = np.nan_to_num(
                    truth,
                    nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0),
                )
                p = np.nan_to_num(
                    pred,
                    nan=(np.nanmean(pred) if np.isfinite(pred).any() else 0.0),
                )
                dr = np.nanmax(truth) - np.nanmin(truth)
                if not np.isfinite(dr) or dr == 0:
                    dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
                ssim_kwargs = {"data_range": dr}
                if ssim_win_size is not None:
                    ssim_kwargs["win_size"] = int(ssim_win_size)
                if ssim_sigma is not None:
                    ssim_kwargs["gaussian_weights"] = True
                    ssim_kwargs["sigma"] = float(ssim_sigma)
                ssim_vals.append(
                    ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)
                )

            months_list.append(mm)
            scores_dict['r2'].append(np.nanmean(r2_vals))
            scores_dict['bias'].append(np.nanmean(bias_vals))
            scores_dict['mae'].append(np.nanmean(mae_vals))
            scores_dict['ssim'].append(np.nanmean(ssim_vals))

        months = np.array(months_list, dtype=int)
        for m in metrics:
            metric_by_year_month[m][year] = (months, scores_dict[m])

    # ---- plotting: 2x2 panel ----
    titles = {'r2': "$R^2$", 'bias': "Bias", 'mae': "MAE", 'ssim': "SSIM"}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = dict(zip(metrics, axs.ravel()))

    for m in metrics:
        ax = axes[m]
        for year, (months, scores) in metric_by_year_month[m].items():
            label = f"{year} (train)" if year == training_year else year
            style = "--" if year == training_year else "-"
            ax.plot(months, scores, style, marker='o', label=label)
        ax.set_title(f"Monthly {titles[m]}")
        ax.set_xlabel("Month")
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(calendar.month_abbr[1:13])
        ax.grid(True)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)

    handles, labels = axes['r2'].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(len(labels), 6),
            frameon=False,
        )
    plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def evaluate_year_batched(
    data, year, model, X_mean, X_std, num_var, cat_var,
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    batch_size=16, model_type='cnn'  # 'cnn' or 'tabular'
):
    """
    Predicts all days in `year` in batches and returns a DataFrame of per-day metrics
    plus a monthly-aggregated summary.

    Returns
    -------
    daily_df : pd.DataFrame with columns ['date','year','month','r2','mae','bias','ssim']
    monthly_df : pd.DataFrame with index (year,month) and mean metrics over days
    """
    ds = data.sel(time=str(year))
    dates = pd.to_datetime(ds.time.values)

    # --- helper to fetch (H,W) at a date; broadcast if var has no time
    def fetch_2d(var, date):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[y_name].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat','lon'):
            arr_t = arr_t.transpose(..., 'lat', 'lon') if 'time' in arr_t.dims else arr_t.transpose('lat','lon')
        return arr_t.values.astype('float32', copy=False)

    # --- build all-day inputs (but predict in mini-batches)
    H, W = int(ds.sizes['lat']), int(ds.sizes['lon'])
    Cn, Cc = len(num_var), len(cat_var)
    C = Cn + Cc

    # We’ll lazily fill a list of batches: [(X_batch, truth_batch, land_batch), ...]
    # to avoid storing the entire year in one big array.
    daily_records = []  # store (date, r2, mae, bias, ssim)
    idx = 0
    while idx < len(dates):
        j = min(idx + batch_size, len(dates))
        batch_dates = dates[idx:j]

        # Build this batch
        X_batch = np.empty((len(batch_dates), H, W, C), dtype='float32')
        truth_batch = np.empty((len(batch_dates), H, W), dtype='float32')
        land_batch = np.empty((len(batch_dates), H, W), dtype=bool)

        for b, date in enumerate(batch_dates):
            chans = []
            # numeric (normalize, fill NaNs with 0 for inputs)
            for k, v in enumerate(num_var):
                a = fetch_2d(v, date)
                denom = 1.0 if X_std[k] == 0 else X_std[k]
                a = (a - X_mean[k]) / denom
                chans.append(np.nan_to_num(a))
            # categorical (just fill NaNs with 0)
            for v in cat_var:
                a = fetch_2d(v, date)
                chans.append(np.nan_to_num(a))
            X_batch[b] = np.stack(chans, axis=-1)

            truth_batch[b] = fetch_2d(y_name, date)
            land_batch[b]  = (fetch_2d(mask_var, date) == 0.0)

        # Predict this batch
        if model_type == 'cnn':
            _ = _require_keras()  # ensure TF present only for cnn path
            pred_batch = model.predict(X_batch, verbose=0)
            if pred_batch.ndim == 4 and pred_batch.shape[-1] == 1:
                pred_batch = pred_batch[..., 0]
        else:  # tabular (e.g., BRT)
            pred_batch = np.empty((len(batch_dates), H, W), dtype='float32')
            for b in range(len(batch_dates)):
                pred_batch[b] = model.predict(X_batch[b][np.newaxis, ...], verbose=0)[0][..., 0]

        # Compute metrics per day
        for b, date in enumerate(batch_dates):
            truth = np.where(land_batch[b], np.nan, truth_batch[b])
            pred  = np.where(land_batch[b], np.nan, pred_batch[b])

            m = ~np.isnan(truth) & ~np.isnan(pred)
            if m.any():
                r2   = r2_score(truth[m].ravel(), pred[m].ravel())
                mae  = float(mean_absolute_error(truth[m], pred[m]))
                bias = float(np.mean(pred[m] - truth[m]))
            else:
                r2 = mae = bias = np.nan

            # SSIM (fill NaNs for SSIM only)
            t = np.nan_to_num(truth, nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0))
            p = np.nan_to_num(pred,  nan=(np.nanmean(pred)  if np.isfinite(pred).any()  else 0.0))
            dr = np.nanmax(truth) - np.nanmin(truth)
            if not np.isfinite(dr) or dr == 0:
                dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
            ssim_kwargs = {"data_range": dr}
            if ssim_win_size is not None:
                ssim_kwargs["win_size"] = int(ssim_win_size)
            if ssim_sigma is not None:
                ssim_kwargs.update(gaussian_weights=True, sigma=float(ssim_sigma))
            ssim_val = ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)

            daily_records.append((date, date.year, date.month, r2, mae, bias, ssim_val))

        idx = j  # next batch

    daily_df = pd.DataFrame(daily_records, columns=["date","year","month","r2","mae","bias","ssim"])
    monthly_df = (daily_df
                  .groupby(["year","month"], as_index=True)[["r2","mae","bias","ssim"]]
                  .mean())

    return daily_df, monthly_df


## MODEL FITTING

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

def train_brt_from_splits(X_train, y_train, feature_names, ocean_channel='ocean_mask',
                          max_samples=300_000, random_state=42,
                          max_depth=6, learning_rate=0.05, max_iter=400,
                          l2_regularization=0.0,
                          grid_shape=None):   # <-- NEW
    T, H, W, C = X_train.shape
    if grid_shape is None:
        grid_shape = (H, W)                  # fallback to training grid

    X2 = X_train.reshape(-1, C)
    y2 = y_train.reshape(-1)

    oce_idx = feature_names.index(ocean_channel)
    valid = (X2[:, oce_idx] > 0.5) & np.isfinite(y2) & np.all(np.isfinite(X2), axis=1)

    X2v, y2v = X2[valid], y2[valid]
    if X2v.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(X2v.shape[0], size=max_samples, replace=False)
        X2v, y2v = X2v[sel], y2v[sel]

    brt = HistGradientBoostingRegressor(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state, validation_fraction=0.1, early_stopping=True
    ).fit(X2v, y2v)

    class BRTWrapper:
        def __init__(self, base, grid_shape):
            self.base = base
            self.grid_shape = tuple(grid_shape)  # (H, W)
        def predict(self, X4, verbose=0):
            X3 = X4[0] if (hasattr(X4, "ndim") and X4.ndim == 4) else X4
            C  = X3.shape[-1]
            flat = self.base.predict(X3.reshape(-1, C))
            H, W = self.grid_shape
            return flat.reshape(1, H, W, 1)  # <-- add batch axis like Keras
        
    return brt, BRTWrapper(brt, grid_shape)


# UTILITIES

def add_latlon_2d(ds):
    import numpy as np
    lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing='ij')
    return ds.assign(
        lat2d=(('lat','lon'), lat2d.astype('float32')),
        lon2d=(('lat','lon'), lon2d.astype('float32')),
    )

def add_sin_coords(ds):
    import numpy as np, xarray as xr
    lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    lonr = np.deg2rad(lon2d); latr = np.deg2rad(lat2d)
    return ds.assign(
        lon_sin=(('lat','lon'), np.sin(lonr).astype('float32')),
        lon_cos=(('lat','lon'), np.cos(lonr).astype('float32')),
        lat_sin=(('lat','lon'), np.sin(latr).astype('float32')),
    )

def add_spherical_coords(obj, lat="lat", lon="lon"):
    """
    Add 3D unit-sphere coordinates (x_geo, y_geo, z_geo) computed from lat/lon.

    - If `obj` is an xarray.Dataset:
        * Assumes `lat` and `lon` are 1D coordinates.
        * Broadcasts to 2D over (lat, lon) and keeps Dask laziness.
        * Returns an xarray.Dataset with x_geo, y_geo, z_geo variables.

    - If `obj` is a pandas.DataFrame:
        * Assumes `lat` and `lon` are columns.
        * Computes per-row x_geo, y_geo, z_geo columns.
        * Returns a new DataFrame (original is not modified in place).

    Parameters
    ----------
    obj : xarray.Dataset or pandas.DataFrame
    lat : str
        Name of latitude coordinate/column in degrees.
    lon : str
        Name of longitude coordinate/column in degrees.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
    """
    import numpy as np, xarray as xr, pandas as pd

    # ---- xarray path --------------------------------------------------------
    if isinstance(obj, xr.Dataset):
        ds = obj

        # 2D lat/lon (lazy if dask)
        lat2d, lon2d = xr.broadcast(ds[lat], ds[lon])   # (lat, lon), (lat, lon)

        # radians (lazy)
        psi = xr.apply_ufunc(np.deg2rad, lat2d, dask="parallelized")
        lam = xr.apply_ufunc(np.deg2rad, lon2d, dask="parallelized")

        x_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.cos, lam, dask="parallelized")
        y_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.sin, lam, dask="parallelized")
        z_geo = xr.apply_ufunc(np.sin, psi, dask="parallelized")

        return ds.assign(
            x_geo=x_geo.astype("float32"),
            y_geo=y_geo.astype("float32"),
            z_geo=z_geo.astype("float32"),
        )

    # ---- pandas path --------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        # radians (vectorized numpy on Series)
        psi = np.deg2rad(df[lat].to_numpy())
        lam = np.deg2rad(df[lon].to_numpy())

        x_geo = np.cos(psi) * np.cos(lam)
        y_geo = np.cos(psi) * np.sin(lam)
        z_geo = np.sin(psi)

        df["x_geo"] = x_geo.astype("float32")
        df["y_geo"] = y_geo.astype("float32")
        df["z_geo"] = z_geo.astype("float32")

        return df

    # ---- unsupported type ---------------------------------------------------
    raise TypeError(
        f"add_spherical_coords expected xarray.Dataset or pandas.DataFrame, "
        f"got {type(obj)}"
    )

def add_seasonal_time_features(obj, ref_var="sst", time="time"):
    """
    Add seasonal time features (sin_time, cos_time) based on day-of-year.

    - If `obj` is an xarray.Dataset:
        * Uses a time coordinate `time` (default 'time').
        * Computes sin_time, cos_time as 3D fields on (time, lat, lon),
          broadcasting using `ref_var` (or infers one).
        * Preserves Dask laziness.

    - If `obj` is a pandas.DataFrame:
        * Uses a datetime-like column `time` (default 'time').
        * Adds per-row float32 columns sin_time and cos_time.

    Parameters
    ----------
    obj : xr.Dataset or pd.DataFrame
    ref_var : str
        For xarray: name of a 3D (time, lat, lon) variable to define the
        broadcast shape. Ignored for pandas.
    time : str
        Name of the time coordinate/column.

    Returns
    -------
    xr.Dataset or pd.DataFrame
    """
    import numpy as np, xarray as xr, pandas as pd
    
    # ---------------- xarray path ----------------
    if isinstance(obj, xr.Dataset):
        ds = obj

        if time not in ds.coords:
            raise ValueError(f"Dataset has no '{time}' coordinate.")

        if ref_var not in ds.data_vars:
            # try to infer a suitable 3D var
            candidates = [
                v
                for v, da in ds.data_vars.items()
                if set(da.dims) >= {"time", "lat", "lon"}
            ]
            if not candidates:
                raise ValueError(
                    "Could not find a 3D (time, lat, lon) variable to broadcast across. "
                    "Pass ref_var='<your_var>' explicitly."
                )
            ref_var = candidates[0]

        # day-of-year (1..366). Works with numpy/cftime calendars.
        doy = ds[time].dt.dayofyear

        # Smooth year length; change to 365/366 for strict calendars if you like
        rad = 2 * np.pi * (doy / 365.25)

        sin_t = xr.apply_ufunc(np.sin, rad, dask="parallelized").astype("float32")
        cos_t = xr.apply_ufunc(np.cos, rad, dask="parallelized").astype("float32")

        # Broadcast time-only arrays to (time, lat, lon) using xarray
        sin_3d, _ = xr.broadcast(sin_t, ds[ref_var])
        cos_3d, _ = xr.broadcast(cos_t, ds[ref_var])

        return ds.assign(
            sin_time=sin_3d,
            cos_time=cos_3d,
        )

    # ---------------- pandas path ----------------
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        if time not in df.columns:
            raise ValueError(f"DataFrame has no '{time}' column.")

        # Ensure datetime, then day-of-year
        time_series = pd.to_datetime(df[time])
        doy = time_series.dt.dayofyear.to_numpy()

        rad = 2 * np.pi * (doy / 365.25)

        sin_t = np.sin(rad).astype("float32")
        cos_t = np.cos(rad).astype("float32")

        df["sin_time"] = sin_t
        df["cos_time"] = cos_t

        return df

    # ---------------- unsupported type ----------------
    raise TypeError(
        f"add_seasonal_time_features expected xarray.Dataset or pandas.DataFrame, "
        f"got {type(obj)}"
    )


def add_solar_time_feature(
    obj,
    time="time",
    lon="lon",
    prefix="solar",
    assume_lon_range="auto",
    ref_var=None,
):
    """
    Add local solar time features to either an xarray.Dataset or pandas.DataFrame.

    Adds:
      - f"{prefix}_hour"      : local solar time in hours [0, 24)
      - f"{prefix}_sin_time"  : sin(2π * hour / 24)
      - f"{prefix}_cos_time"  : cos(2π * hour / 24)

    Local solar time is approximated as:
        solar_hour = (UTC_hour + lon_east / 15) % 24

    Parameters
    ----------
    obj : xr.Dataset or pd.DataFrame
    time : str
        Name of time coord/column.
    lon : str
        Name of longitude coord/column (degrees, either [-180, 180] or [0, 360]).
    prefix : str
        Prefix for created columns/variables:
        "<prefix>_hour", "<prefix>_sin_time", "<prefix>_cos_time".
    assume_lon_range : {"auto", "180", "360"}
        How to interpret longitude range:
        - "auto": if any lon > 180, assume [0, 360] and convert to [-180, 180]
        - "180" : assume already in [-180, 180]
        - "360" : assume in [0, 360], convert to [-180, 180]
    ref_var : str, optional (xarray only)
        For Dataset case, if provided and obj[ref_var] is 2D/3D (e.g. time,lat,lon),
        solar fields will be broadcast to that variable's shape.

    Returns
    -------
    xr.Dataset or pd.DataFrame
    """
    import numpy as np, xarray as xr, pandas as pd
    
    hour_name = f"{prefix}_hour"
    sin_name = f"{prefix}_sin_time"
    cos_name = f"{prefix}_cos_time"

    # ---------------- xarray path ----------------
    if isinstance(obj, xr.Dataset):
        ds = obj

        if time not in ds.coords:
            raise ValueError(f"Dataset has no '{time}' coordinate.")
        if lon not in ds.coords and lon not in ds:
            raise ValueError(f"Dataset has no '{lon}' coordinate/variable.")

        # grab longitude as DataArray
        lon_da = ds[lon] if lon in ds.coords else ds[lon]
        lon_vals = lon_da.astype(float)

        # Handle longitude range (this may touch memory for lon, but lon is usually small)
        if assume_lon_range == "auto":
            lon_np = np.asarray(lon_vals)
            if np.nanmax(lon_np) > 180:
                lon_vals = ((lon_vals + 180) % 360) - 180
        elif assume_lon_range == "360":
            lon_vals = ((lon_vals + 180) % 360) - 180
        # else "180": leave as-is

        # UTC hour from time coord (xarray dt-accessor, dask-friendly)
        t = ds[time].dt
        utc_hour = t.hour + t.minute / 60.0 + t.second / 3600.0  # DataArray 1D in time

        # Broadcast utc_hour and lon to a common grid
        # (e.g. (time, lon) or (time, lat, lon) depending on shapes)
        utc_hour_b, lon_b = xr.broadcast(utc_hour, lon_vals)

        # Compute solar hour
        solar_hour = (utc_hour_b + lon_b / 15.0) % 24.0

        # Cyclical encodings
        rad = 2 * np.pi * (solar_hour / 24.0)
        solar_sin = xr.apply_ufunc(np.sin, rad, dask="parallelized").astype("float32")
        solar_cos = xr.apply_ufunc(np.cos, rad, dask="parallelized").astype("float32")

        # If ref_var given, broadcast onto that variable's shape
        if ref_var is not None and ref_var in ds:
            solar_hour, _ = xr.broadcast(solar_hour, ds[ref_var])
            solar_sin, _ = xr.broadcast(solar_sin, ds[ref_var])
            solar_cos, _ = xr.broadcast(solar_cos, ds[ref_var])

        return ds.assign(
            **{
                hour_name: solar_hour.astype("float32"),
                sin_name: solar_sin,
                cos_name: solar_cos,
            }
        )

    # ---------------- pandas path ----------------
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        if time not in df.columns:
            raise ValueError(f"DataFrame has no '{time}' column.")
        if lon not in df.columns:
            raise ValueError(f"DataFrame has no '{lon}' column.")

        # Ensure datetime with UTC
        df[time] = pd.to_datetime(df[time], utc=True, errors="coerce")

        # Handle longitude range
        lon_series = df[lon].astype(float)

        if assume_lon_range == "auto":
            if (lon_series > 180).any():
                lon_series = ((lon_series + 180) % 360) - 180
        elif assume_lon_range == "360":
            lon_series = ((lon_series + 180) % 360) - 180
        # else "180": leave as-is

        df[lon] = lon_series

        # UTC hour as float
        t = df[time].dt
        utc_hour = t.hour + t.minute / 60.0 + t.second / 3600.0

        # Local solar hour
        solar_hour = (utc_hour + lon_series / 15.0) % 24.0

        # Cyclical encodings
        rad = 2 * np.pi * (solar_hour / 24.0)
        sin_t = np.sin(rad)
        cos_t = np.cos(rad)

        df[hour_name] = solar_hour.astype("float32")
        df[sin_name] = sin_t.astype("float32")
        df[cos_name] = cos_t.astype("float32")

        return df

    # ---------------- unsupported type ----------------
    raise TypeError(
        f"add_solar_time_features expected xarray.Dataset or pandas.DataFrame, "
        f"got {type(obj)}"
    )
    
from scipy.ndimage import distance_transform_edt

def add_distance_to_coast(ds: xr.Dataset,
                          mask_var="ocean_mask",
                          out_name="dist2coast_km") -> xr.Dataset:
    """
    Add distance-to-coast (km) for ocean pixels; 0 on land.
    Assumes lat/lon on a regular grid. Uses EDT with anisotropic sampling.
    """
    
    if mask_var not in ds:
        raise KeyError(f"{mask_var} not found in dataset.")
    if not {"lat","lon"} <= set(ds.coords):
        raise KeyError("Dataset must have 'lat' and 'lon' coordinates.")

    # 2D boolean masks (lat, lon)
    ocean = (ds[mask_var].astype(bool))
    if "time" in ocean.dims:
        ocean2d = ocean.isel(time=0)  # mask is time-invariant in your data
    else:
        ocean2d = ocean

    # Build per-axis sampling (km per pixel). Use mean spacing + mean latitude.
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    dlat = np.abs(np.diff(lat_vals)).mean() if lat_vals.size > 1 else 0.0
    dlon = np.abs(np.diff(lon_vals)).mean() if lon_vals.size > 1 else 0.0
    lat0  = float(np.mean(lat_vals))  # for lon scaling
    km_per_deg = 111.0
    dy_km = dlat * km_per_deg
    dx_km = dlon * km_per_deg * max(np.cos(np.deg2rad(lat0)), 1e-6)  # avoid 0 at poles

    # Distance from ocean -> nearest land (0 on land)
    # EDT expects True for the "non-zero" region to measure distance *to zero* pixels.
    # We want distance to land, so zero = land, one = ocean.
    land2d = ~ocean2d.values
    dist_km = distance_transform_edt(~land2d, sampling=(dy_km, dx_km))  # ~land2d == ocean
    dist_km[land2d] = 0.0  # exactly zero on land

    # Wrap into DataArray and attach
    dist_da = xr.DataArray(dist_km.astype("float32"),
                           dims=("lat","lon"),
                           coords={"lat": ds["lat"], "lon": ds["lon"]},
                           name=out_name,
                           attrs={"units":"km", "long_name":"distance to coast (over ocean)"})
    return ds.assign({out_name: dist_da})

def count_valid_days_by_month(
    ds: xr.Dataset,
    year,
    vars_to_check=("y",),          # str or iterable of str
    mask_var="ocean_mask",         # 1=ocean, 0=land
    nan_max_frac=0.05,             # allow ≤ 5% NaNs over ocean
):
    """
    Return a Series with counts of days per month that pass the NaN filter.
    A day passes iff, for every variable in `vars_to_check`, the number of
    NaNs over the ocean is <= nan_max_frac * ocean_pixels_on_that_day.
    """
    # subset to the year
    dsy = ds.sel(time=str(year))

    # tidy inputs
    if isinstance(vars_to_check, str):
        vars_to_check = [vars_to_check]

    # spatial mask, broadcast to (time, lat, lon)
    ocean = dsy[mask_var].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims(time=dsy["time"])
    # broadcast to the target grid (uses first var as template)
    template = dsy[vars_to_check[0]] if vars_to_check else dsy["y"]
    ocean = ocean.broadcast_like(template)

    # how many ocean pixels each day?
    spatial_dims = [d for d in ocean.dims if d != "time"]
    ocean_count = ocean.sum(dim=spatial_dims)  # (time,)
    nan_thresh_t = nan_max_frac * ocean_count  # per time-step threshold

    # build per-var validity (True if NaN count over ocean <= threshold)
    valid_per_var = []
    for v in vars_to_check:
        if v not in dsy:
            raise KeyError(f"Variable '{v}' not in dataset.")
        arr = dsy[v]
        if "time" not in arr.dims:
            arr = arr.expand_dims(time=dsy["time"])
        arr = arr.broadcast_like(ocean)
        v_nan = xr.apply_ufunc(np.isnan, arr) & ocean
        v_nan_count = v_nan.sum(dim=spatial_dims)  # (time,)
        valid_per_var.append(v_nan_count <= nan_thresh_t)

    # day is valid if all vars pass
    valid_all = xr.concat(valid_per_var, dim="var").all(dim="var")  # (time,) bool

    # group by month and count
    month_idx = pd.to_datetime(valid_all["time"].values).month
    counts = pd.Series(valid_all.values, index=month_idx).groupby(level=0).sum().astype(int)
    # ensure all months present
    counts = counts.reindex(range(1,13), fill_value=0)
    counts.index.name = "month"
    counts.name = f"valid_days_{year}"
    return counts

def pct_missing_by_day_year(
    ds: xr.Dataset,
    year: int,
    var: str = "y",
    mask_var: str = "ocean_mask",   # 1=ocean, 0=land
) -> pd.Series:
    """
    Return a Series with the percent of ocean pixels that are NaN
    for each day in the given year.

    Each value is:
        (# NaN pixels over ocean) / (# ocean pixels) * 100
    for the variable `var`.
    """
    # subset to year explicitly
    dsy = ds.sel(time=ds["time"].dt.year == year)
    if dsy.sizes.get("time", 0) == 0:
        raise ValueError(f"No data found for year {year}.")

    # ocean mask -> bool, broadcast to (time, lat, lon, ...)
    ocean = dsy[mask_var].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims(time=dsy["time"])

    arr = dsy[var]
    if "time" not in arr.dims:
        arr = arr.expand_dims(time=dsy["time"])

    ocean = ocean.broadcast_like(arr)

    # spatial dims (everything except time)
    spatial_dims = [d for d in ocean.dims if d != "time"]

    # counts per day
    ocean_count = ocean.sum(dim=spatial_dims)          # (time,)
    nan_mask = arr.isnull() & ocean
    nan_count = nan_mask.sum(dim=spatial_dims)         # (time,)

    frac = nan_count / ocean_count                     # (time,), 0–1
    pct = (frac * 100).to_series()
    pct.name = f"pct_missing_{var}_{year}"

    return pct


# --- Match-up Code ---
def sample_points_fast(
    ds: xr.Dataset,
    year: int,
    n: int,
    y_name: str = "y",
    mask_var: str = "ocean_mask",
    seed: int = 42,
):
    """
    Vectorized random sampler for chunked xarray/dask datasets (no loops).

    - Draws n random times from ds.sel(time=year)
    - Draws n random *continuous* lat/lon within the coord ranges
    - Snaps those lat/lon to the nearest grid cell (vectorized)
    - Gathers y and mask using a single dask vindex, sets y=NaN where mask==False
    - Returns only rows with finite y

    Returns
    -------
    pd.DataFrame with columns: ['time','lat','lon','y']
      - 'lat','lon' are the original random (continuous) coordinates
        (y is taken from the nearest grid cell).
    """
    import dask.array as da  # local import to keep dependencies scoped

    dsy = ds.sel(time=str(year))
    if dsy.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps found for year {year}.")

    rng = np.random.default_rng(seed)

    # ----- random time indices -----
    T = dsy.sizes["time"]
    t_idx = rng.integers(0, T, size=n)

    # ----- random continuous lat/lon, then snap to nearest grid index -----
    lat_vals = dsy["lat"].values
    lon_vals = dsy["lon"].values

    lat_rand = rng.uniform(lat_vals.min(), lat_vals.max(), size=n)
    lon_rand = rng.uniform(lon_vals.min(), lon_vals.max(), size=n)

    # searchsorted-based nearest-index that works for ascending or descending arrays
    def nearest_index(coord_vals, q):
        asc = coord_vals[0] <= coord_vals[-1]
        base = coord_vals if asc else -coord_vals
        tgt  = q          if asc else -q
        idx = np.searchsorted(base, tgt, side="left")
        idx0 = np.clip(idx - 1, 0, len(coord_vals) - 1)
        idx1 = np.clip(idx,     0, len(coord_vals) - 1)
        pick_right = np.abs(coord_vals[idx1] - q) < np.abs(coord_vals[idx0] - q)
        return np.where(pick_right, idx1, idx0)

    lat_i = nearest_index(lat_vals, lat_rand)
    lon_i = nearest_index(lon_vals, lon_rand)

    # ----- vindex gather for y and mask (single compute) -----
    y_da = dsy[y_name].data  # (time, lat, lon) dask array
    y_s  = y_da.vindex[t_idx, lat_i, lon_i]  # (n,)

    m_da = dsy[mask_var]
    if "time" in m_da.dims:
        m_s = m_da.data.vindex[t_idx, lat_i, lon_i]
    else:
        m_s = m_da.data.vindex[lat_i, lon_i]

    y_np, m_np = da.compute(y_s, m_s)

    # apply mask: keep only ocean
    y_np = np.where(m_np.astype(bool), y_np, np.nan)

    # ----- assemble output; keep the random (continuous) lat/lon -----
    times = pd.to_datetime(dsy["time"].values[t_idx])
    df = pd.DataFrame(
        {"time": times,
         "lat":  lat_rand.astype(float),
         "lon":  lon_rand.astype(float),
         "y":    y_np.astype(float)}
    )

    return df.dropna(subset=["y"]).reset_index(drop=True)

# process one file code
import pandas as pd
import earthaccess
import xarray as xr

def one_file_matches_old(
    f, df,
    ds_lat_name="lat", ds_lon_name="lon", ds_time_name="time", 
    ds_vec_name="wavelength", ds_var_name="Rrs",
    df_lat_name="lat", df_lon_name="lon", df_time_name="time",
    df_var_name="y"
):
    with xr.open_dataset(f, chunks={}, cache=False) as ds:

        # --- Step 1: subset df to the time window in ds
        t_start = pd.to_datetime(ds.attrs["time_coverage_start"], utc=True)
        t_end   = pd.to_datetime(ds.attrs["time_coverage_end"], utc=True)
        df_times = pd.to_datetime(df[df_time_name], utc=True)
        # df points that are in this record
        df_record = df[(df_times >= t_start) & (df_times < t_end)]

        # Stop if no points in df are in the record
        if df_record.empty: return None, None

        # --- Step 2: Get an array of Rrs values for the lat/lon in df_day
        # will need index and values later
        lat_idx = ds.get_index(ds_lat_name)
        lon_idx = ds.get_index(ds_lon_name)
        lat_vals = ds[ds_lat_name].values
        lon_vals = ds[ds_lon_name].values
        # Get the lat/lon vals in df_day
        df_lats = df_record[df_lat_name].to_numpy(dtype=float)
        df_lons = df_record[df_lon_name].to_numpy(dtype=float)
        # Use pandas indexer to quickly find indices for vals nearest points in df_record
        lat_i = lat_idx.get_indexer(df_lats, method="nearest")
        lon_i = lon_idx.get_indexer(df_lons, method="nearest")

        # If record has only 10-100 points to match. This will be faster than .vindex
        def sample_few_points(ds, lats, lons, var_name=ds_var_name):
            import numpy as np
            ds_var = ds[var_name]  # (lat, lon, wavelength)
            spectra = [
                ds_var.sel(lat=i, lon=j, method="nearest").values
                for i, j in zip(lats, lons)
            ]
            return np.stack(spectra, axis=0)
        
        var_vals=sample_few_points(ds, df_lats, df_lons)    
        
        # --- Step 4: build a dataframe with our points and data in the xr.DataSet ---
        data = {
            ds_time_name: pd.to_datetime(df_record[df_time_name], utc=True),
            ds_lat_name:  lat_vals[lat_i],
            ds_lon_name:  lon_vals[lon_i],
            df_var_name: df_record[df_var_name].to_numpy(),
            "df_lat": df_lats,
            "df_lon": df_lons
            }

        # Get the wavelength values for our col names
        if not ds_vec_name == None:
            vec_vals = ds[ds_vec_name].values
            for j, v in enumerate(vec_vals):
                label = int(v) # make integer for nicer label
                col_name = f"{ds_var_name}_{label}"
                data[col_name] = var_vals[:, j].astype(float)
        else:
            data[ds_var_name] = var_vals[:].astype(float)

        return df_record, pd.DataFrame(data)


from typing import Optional, Tuple
import xarray as xr
import pandas as pd
import numpy as np

def one_file_matches(
    f: "earthaccess.store.EarthAccessFile",
    df: pd.DataFrame,
    ds_lat_name: str = "lat",
    ds_lon_name: str = "lon",
    ds_time_name: str = "time",
    ds_vec_name: Optional[str] = "wavelength",
    ds_var_name: str = "Rrs",
    ds_vec_sel = None,
    df_lat_name: str = "lat",
    df_lon_name: str = "lon",
    df_time_name: str = "time",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Match Argo point observations to a single PACE L2/L3 file and extract
    colocated satellite values and metadata.

    Parameters
    ----------
    f : file-like object
        An earthaccess/open file-like handle for a single PACE granule
        (as returned by `earthaccess.open`). This object is passed directly
        to `xr.open_dataset` to read the granule.
    df : pandas.DataFrame
        A DataFrame containing Argo observations. Must include columns for
        time, latitude, longitude, and the target Argo variable to be matched.

    ds_lat_name, ds_lon_name, ds_time_name : str, optional
        Names of the latitude, longitude, and time variables in the PACE
        dataset. Default: "lat", "lon", "time".

    ds_vec_name : str or None, optional
        Name of the spectral dimension in the PACE dataset (e.g. "wavelength").
        If not None, matched satellite spectra are returned with one column per
        wavelength. If None, only a single variable is extracted.

    ds_vec_sel : value or None, optional
        Value of the spectral dimension in the PACE dataset (e.g. "wavelength")
        to select. If None, matched satellite spectra are returned with one column per
        wavelength. If given, only a single variable is extracted for that value.

    ds_var_name : str, optional
        Name of the satellite variable to extract from the PACE dataset
        (e.g. "Rrs" or "chlor_a").

    df_lat_name, df_lon_name, df_time_name : str, optional
        Column names in `df` for latitude, longitude, and time.

    df_var_name : str, optional
        Column name in `df` for the Argo variable being matched.

    Returns
    -------
    df_record : pandas.DataFrame or None
        A subset of `df` containing only the Argo observations whose timestamps
        fall within the PACE file's temporal coverage window. Returns None if
        no observations fall within this window.

    pts : pandas.DataFrame or None
        A DataFrame containing colocated satellite values for each matched Argo
        observation, including:
            - matched PACE pixel coordinates
            - the Argo variable value
            - spectral satellite values (if `ds_vec_name` is provided)
            - PACE temporal metadata (`pace_t_start`, `pace_t_end`)
            - the PACE file name (`pace_file`)
        Returns None if no points were matched.

Examples
    --------
    >>> import earthaccess
    >>> import pandas as pd
    >>>
    >>> # Log in and search for PACE granules (simplified example)
    >>> earthaccess.login()
    >>> results = earthaccess.search(
    ...     short_name="PACE_OCI_L3M_DAY_IOP",
    ...     temporal=("2024-03-05", "2024-03-06"),
    ...     bounding_box=(-180, -90, 180, 90),
    ... )
    >>> files = earthaccess.open(results, pqdm_kwargs={"disable": True})
    >>>
    >>> # Load Argo matchup candidates
    >>> df_argo = pd.read_parquet("tutorial_data/chl_argo_points.parquet")
    >>>
    >>> # Match a single PACE file to the Argo DataFrame
    >>> df_record, pts = one_file_matches(
    ...     files[0],
    ...     df_argo,
    ...     ds_vec_name=None,            # e.g. non-spectral variable like "chlor_a"
    ...     ds_var_name="chlor_a",
    ...     df_var_name="argo_chl"
    ... )
    
    Notes
    -----
    - Time matching uses a half-open interval: `[t_start, t_end)`.
    - Spatial matching is performed using nearest-neighbor lookup on the PACE
      latitude/longitude grid.
    - This function does not load full PACE granules into memory; only metadata
      and the required pixel values are accessed.
    - Returned DataFrames are aligned row-by-row: each row in `pts` corresponds
      to the same row in `df_record`.
    """
    with xr.open_dataset(f, chunks={}, cache=False) as ds:

        # --- time window in ds ---
        t_start = pd.to_datetime(ds.attrs["time_coverage_start"], utc=True)
        t_end   = pd.to_datetime(ds.attrs["time_coverage_end"], utc=True)

        # filename / product name
        fname = ds.attrs.get("product_name", None)
        if fname is None:
            src = ds.encoding.get("source", "")
            fname = src.split("/")[-1] if "/" in src else src

        df_times = pd.to_datetime(df[df_time_name], utc=True)
        # Use 24 hours so as not to cut off data if the window is small
        t_end_24 = t_start + pd.Timedelta(hours=24)
        df_record = df[(df_times >= t_start) & (df_times < t_end_24)]

        if df_record.empty:
            return None, None

        # --- spatial index / nearest lat-lon ---
        lat_idx = ds.get_index(ds_lat_name)
        lon_idx = ds.get_index(ds_lon_name)
        lat_vals = ds[ds_lat_name].values
        lon_vals = ds[ds_lon_name].values

        df_lats = df_record[df_lat_name].to_numpy(dtype=float)
        df_lons = df_record[df_lon_name].to_numpy(dtype=float)

        lat_i = lat_idx.get_indexer(df_lats, method="nearest")
        lon_i = lon_idx.get_indexer(df_lons, method="nearest")

        def sample_few_points(ds, lats, lons, var_name=ds_var_name):
            ds_var = ds[var_name]  # e.g. (lat, lon, wavelength)
            spectra = [
                ds_var.sel(lat=i, lon=j, method="nearest").values
                for i, j in zip(lats, lons)
            ]
            return np.stack(spectra, axis=0)

        var_vals = sample_few_points(ds, df_lats, df_lons)

        n = len(df_record)

        # --- build dataframe ---
        data = {
            # PACE file metadata per row
            f"pace_{ds_var_name}_file":  np.full(n, fname),
            f"pace_{ds_var_name}_t_start": np.full(n, t_start),
            f"pace_{ds_var_name}_t_end":   np.full(n, t_end),
            f"pace_{ds_var_name}_lat": lat_vals[lat_i],
            f"pace_{ds_var_name}_lon": lon_vals[lon_i],
        }

        if ds_vec_name is not None:
            vec_vals = ds[ds_vec_name].values
            if ds_vec_sel is not None:
                m = vec_vals == ds_vec_sel
                vec_vals = vec_vals[m]
            for j, v in enumerate(vec_vals):
                label = int(v)
                col_name = f"pace_{ds_var_name}_{label}"
                data[col_name] = var_vals[:, j].astype(float)
        else:
            col_name = f"pace_{ds_var_name}"
            data[col_name] = var_vals[:].astype(float)

        pts = pd.DataFrame(data)
        return df_record, pts

        
def matchup_dask(
    ds: xr.Dataset,
    samples: np.array,
    n: int,
    y_name: str = "y",
    var_name: np.array = ["lat", "lon"]
):
    """
    Vectorized random sampler for chunked xarray/dask datasets (no loops).

    - Draws n random times from ds.sel(time=year)
    - Draws n random *continuous* lat/lon within the coord ranges
    - Snaps those lat/lon to the nearest grid cell (vectorized)
    - Gathers y and mask using a single dask vindex, sets y=NaN where mask==False
    - Returns only rows with finite y

    Returns
    -------
    pd.DataFrame with columns: ['time','lat','lon','y']
      - 'lat','lon' are the original random (continuous) coordinates
        (y is taken from the nearest grid cell).
    """
    import dask.array as da  # local import to keep dependencies scoped

    dsy = ds.sel(time=str(year))
    if dsy.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps found for year {year}.")

    rng = np.random.default_rng(seed)

    # ----- random time indices -----
    T = dsy.sizes["time"]
    t_idx = rng.integers(0, T, size=n)

    # ----- random continuous lat/lon, then snap to nearest grid index -----
    lat_vals = dsy["lat"].values
    lon_vals = dsy["lon"].values

    lat_rand = rng.uniform(lat_vals.min(), lat_vals.max(), size=n)
    lon_rand = rng.uniform(lon_vals.min(), lon_vals.max(), size=n)

    # searchsorted-based nearest-index that works for ascending or descending arrays
    def nearest_index(coord_vals, q):
        asc = coord_vals[0] <= coord_vals[-1]
        base = coord_vals if asc else -coord_vals
        tgt  = q          if asc else -q
        idx = np.searchsorted(base, tgt, side="left")
        idx0 = np.clip(idx - 1, 0, len(coord_vals) - 1)
        idx1 = np.clip(idx,     0, len(coord_vals) - 1)
        pick_right = np.abs(coord_vals[idx1] - q) < np.abs(coord_vals[idx0] - q)
        return np.where(pick_right, idx1, idx0)

    lat_i = nearest_index(lat_vals, lat_rand)
    lon_i = nearest_index(lon_vals, lon_rand)

    # ----- vindex gather for y and mask (single compute) -----
    y_da = dsy[y_name].data  # (time, lat, lon) dask array
    y_s  = y_da.vindex[t_idx, lat_i, lon_i]  # (n,)

    m_da = dsy[mask_var]
    if "time" in m_da.dims:
        m_s = m_da.data.vindex[t_idx, lat_i, lon_i]
    else:
        m_s = m_da.data.vindex[lat_i, lon_i]

    y_np, m_np = da.compute(y_s, m_s)

    # apply mask: keep only ocean
    y_np = np.where(m_np.astype(bool), y_np, np.nan)

    # ----- assemble output; keep the random (continuous) lat/lon -----
    times = pd.to_datetime(dsy["time"].values[t_idx])
    df = pd.DataFrame(
        {"time": times,
         "lat":  lat_rand.astype(float),
         "lon":  lon_rand.astype(float),
         "y":    y_np.astype(float)}
    )

    return df.dropna(subset=["y"]).reset_index(drop=True)


# ---- Helper code for STAC Json

import json
from pathlib import Path
from datetime import datetime

def load_or_create_collection(path):
    path = Path(path)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    # Create a new skeleton if it doesn't exist
    return {
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": "fish-pace-2025-tutorial-data",
        "description": "Datasets used in the Fish-PACE 2025 tutorials (tutorial_data/).",
        "license": "Open access where specified by each dataset's source; see per-item metadata.",
        "links": [],
        "extent": {
            "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
            "temporal": {"interval": [["2024-01-01T00:00:00Z", None]]}
        },
        "keywords": ["tutorials"],
        "providers": [],
        "summaries": {},
        "assets": {},
        "item_assets": {},
        "items": []  # <- we'll keep items here
    }

def add_or_update_item(
    collection,
    item_id,
    asset_href,
    *,
    title,
    description,
    start_datetime=None,
    end_datetime=None,
    extra_properties=None
):
    items = collection.setdefault("items", [])
    extra_properties = extra_properties or {}

    # Try to find existing item
    existing = next((it for it in items if it.get("id") == item_id), None)

    props = {
        "title": title,
        "description": description,
        "created": datetime.utcnow().isoformat() + "Z",
    }
    if start_datetime or end_datetime:
        props["start_datetime"] = start_datetime
        props["end_datetime"] = end_datetime
    props.update(extra_properties)

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "properties": props,
        "geometry": None,     # you could add a bbox/point later if you want
        "bbox": None,
        "assets": {
            "data": {
                "href": asset_href,
                "type": "application/x-parquet",
                "roles": ["data"],
                "title": title
            }
        },
        "links": []
    }

    if existing is not None:
        # update in-place
        idx = items.index(existing)
        items[idx] = item
    else:
        items.append(item)

    return collection

def save_collection(collection, path):
    path = Path(path)
    with path.open("w") as f:
        json.dump(collection, f, indent=2)

import json
from pathlib import Path

import json
from pathlib import Path
from urllib.parse import urlparse

def is_url(href: str) -> bool:
    return bool(urlparse(href).scheme and urlparse(href).netloc)

def stac_to_readme(
    stac_path,
    readme_path="README.md",
    repo_raw_base=None
):
    """
    Convert a STAC Collection with an 'items' list into a human-readable README.

    Parameters
    ----------
    stac_path : str or Path
        Path to STAC collection JSON.
    readme_path : str or Path
        Output README path. Default: README.md.
    repo_raw_base : str, optional
        Base URL for constructing public URLs for relative asset hrefs.
        Example: 'https://raw.githubusercontent.com/fish-pace/2025-tutorials/main'
    """
    stac_path = Path(stac_path)
    with stac_path.open() as f:
        collection = json.load(f)

    items = collection.get("items", [])
    coll_providers = collection.get("providers", [])

    readme_lines = []
    readme_lines.append("# Tutorial Data Catalog\n")
    readme_lines.append(
        "This README is autogenerated from the STAC catalog.\n"
        f"Source STAC file: `{stac_path}`\n\n"
    )
    readme_lines.append("---\n")

    for i, item in enumerate(items, start=1):
        props = item.get("properties", {})
        assets = item.get("assets", {})

        title = props.get("title", item.get("id"))
        readme_lines.append(f"## {i}. {title}\n")

        # File Nme
        file_name_ = props.get("file_name")
        if file_name_:
            readme_lines.append(f"**File name:** {file_name_}\n")

        readme_lines.append(f"**ID:** `{item.get('id')}`\n")

        # Description
        desc = props.get("description")
        if desc:
            readme_lines.append(f"**Description:**\n{desc}\n")

        # Time coverage
        if "start_datetime" in props or "end_datetime" in props:
            readme_lines.append(
                "**Time range:** "
                f"{props.get('start_datetime', 'N/A')} → "
                f"{props.get('end_datetime', 'N/A')}\n"
            )

        # Spatial
        if item.get("bbox"):
            lon_min, lat_min, lon_max, lat_max = item["bbox"]
            readme_lines.append("**Geospatial coverage:**\n")
            readme_lines.append(f"- Latitude: {lat_min} to {lat_max}\n")
            readme_lines.append(f"- Longitude: {lon_min} to {lon_max}\n")

        # License
        license_ = props.get("license") or collection.get("license")
        if license_:
            readme_lines.append(f"**License:** {license_}\n")

        # Providers
        if coll_providers:
            readme_lines.append("**Providers:**\n")
            for p in coll_providers:
                name = p.get("name", "Unknown")
                url = p.get("url")
                if url:
                    readme_lines.append(f"- {name} ({url})\n")
                else:
                    readme_lines.append(f"- {name}\n")

        # Creation notebook
        nb_path = props.get("tutorial_notebook")
        if nb_path:
            readme_lines.append("**Creation / provenance notebook:**\n")
            readme_lines.append(f"- [{nb_path}]({nb_path})\n")

        # Data assets
        if assets:
            readme_lines.append("**Data file(s):**\n")
            for k, a in assets.items():
                href = a.get("href")
                if href is None:
                    continue

                # Construct public URL if needed
                if repo_raw_base and not is_url(href):
                    public_url = repo_raw_base.rstrip("/") + "/" + href.lstrip("/")
                else:
                    public_url = href

                readme_lines.append(f"- `{href}`\n")
                if public_url != href:
                    readme_lines.append(f"  \n  Public URL: {public_url}\n")

                # Python example
                readme_lines.append("  **Python load example:**\n")

                if href.endswith(".parquet"):
                    readme_lines.append("  ```python")
                    readme_lines.append("  import pandas as pd")
                    readme_lines.append(f"  url = '{public_url}'")
                    readme_lines.append("  df = pd.read_parquet(url)")
                    readme_lines.append("  df.head()\n")
                    readme_lines.append("  # You can read metadata with pyarrow")
                    readme_lines.append("  import fsspec")
                    readme_lines.append("  import pyarrow.parquet as pq")
                    readme_lines.append("  with fsspec.open(url, 'rb') as f:")
                    readme_lines.append("      t = pq.read_table(f)")
                    readme_lines.append("  t.schema.metadata")
                    readme_lines.append("  ```\n\n")

                elif href.endswith((".nc", ".nc4", ".zarr")):
                    readme_lines.append("  ```python\n")
                    readme_lines.append("  import xarray as xr\n")
                    readme_lines.append(f"  url = '{public_url}'\n")
                    readme_lines.append("  ds = xr.open_dataset(url)\n")
                    readme_lines.append("  ds\n")
                    readme_lines.append("  ```\n\n")

                else:
                    readme_lines.append("  ```python\n")
                    readme_lines.append("  url = '{public_url}'\n")
                    readme_lines.append("  # Load using appropriate library\n")
                    readme_lines.append("  ```\n\n")

        readme_lines.append("\n---\n")

    # Write README.md
    readme_path = Path(readme_path)
    with readme_path.open("w") as f:
        f.write("\n".join(readme_lines))

    print(f"README.md written to {readme_path}")
