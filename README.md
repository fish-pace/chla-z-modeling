# CHLA-Z: Global Chlorophyll-a Vertical Distribution (0–200 m)

**Tags:** ocean color · chlorophyll-a · Bio-Argo · PACE · machine learning · NOAA

---

## Description

**CHLA-Z** is a global gridded dataset providing estimates of chlorophyll-a concentration as a function of depth (0–200 m) on a regular latitude/longitude grid, along with derived vertical-structure metrics.

The product is generated using a **boosted regression tree (BRT)** model trained on **PACE OCI Level-3 mapped remote-sensing reflectances (Rrs)** and **in-situ chlorophyll profile observations** from **Bio-Argo** and **OOI**.

### Dataset available via NOAA Open Data Dissemination (NODD) on Google Cloud Storage

```
gs://nmfs_odp_nwfsc/CB/fish-pace-datasets/chla-z
```

Public HTTPS access (no authentication required):  
https://storage.googleapis.com/nmfs_odp_nwfsc/CB/fish-pace-datasets/chla-z/

> Note: Google Cloud Console “Browse” links require login even for public buckets.  
> Use HTTPS links or example files for anonymous access.

---

## GitHub repository

* **Repo:** https://github.com/fish-pace/chla-z
* [Example notebook](https://github.com/fish-pace/chla-z/blob/main/notebooks/examples.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/fish-pace/chla-z/blob/main/notebooks/examples.ipynb)
* [Dataset metadata (`chla-z.json`)](https://github.com/fish-pace/chla-z/blob/main/chla-z.json)

---

### Dataset contents

- **Core variable:**  
  `CHLA` (`time`, `z`, `lat`, `lon`) — chlorophyll-a concentration (mg m⁻³)

- **Derived layers:**  
  - `CHLA_int_0_200` (mg m⁻²)  
  - `CHLA_peak` (mg m⁻³)  
  - `CHLA_peak_depth` (m)  
  - `CHLA_depth_center_of_mass` (m)

- **Vertical grid:** 10 m bin centers with bounds `z_start` / `z_end`  
- **Spatial grid:** global regular lat/lon (~4 km nominal; 0.041666668° step)  
- **Temporal resolution:** daily  
- **Formats:**  
  - Zarr v3 (cloud-optimized)  
  - NetCDF (one file per day)

> **Note on “4 km”:**  
> This dataset is on a regular latitude/longitude grid (0.041666668°).  
> North–south spacing is approximately constant; east–west spacing decreases with latitude (cos(lat)).

---

## Talks (draft stage)

* [Talk for HyperDawgs Dec 12, 2025](https://fish-pace.github.io/chla-z-modeling/text_and_talks/chla_depth_profiles.html#/title-slide)
* [Talk 2 in prep](https://fish-pace.github.io/chla-z-modeling/text_and_talks/chla_depth_talk_2.html#/title-slide)

---

## Python examples

### Stream Zarr from Google Cloud Storage

```python
import xarray as xr

zarr_url = "gcs://nmfs_odp_nwfsc/CB/fish-pace-datasets/chla-z/zarr"
ds = xr.open_zarr(
    zarr_url,
    consolidated=False,
    storage_options={"token": "anon"}
)

# Example: time series at a point (nearest grid cell)
pt = ds["CHLA"].sel(lon=-155, lat=20, method="nearest")
pt = pt.isel(z=0)  # surface
pt.sel(time=slice("2024-03-01", "2024-04-01")).plot()
```

### Stream a single NetCDF day

```python
import xarray as xr

url = "gcs://nmfs_odp_nwfsc/CB/fish-pace-datasets/chla-z/netcdf/chla_z_20240305_v2.nc"
ds = xr.open_dataset(
    url,
    engine="h5netcdf",
    storage_options={"token": "anon"}
)

# Plot surface CHLA for that day
ds["CHLA"].isel(time=0, z=0).plot()
```

---

## R example (NetCDF)

> **Note:** R typically does not stream NetCDF directly from `gs://`.  
> Download via HTTPS first.

```r
library(terra)

url <- "https://storage.googleapis.com/nmfs_odp_nwfsc/CB/fish-pace-datasets/chla-z/netcdf/chla_z_20240305_v2.nc"
download.file(url, "chla_z_20240305_v2.nc", mode = "wb")

r <- rast("chla_z_20240305_v2.nc", subds = "CHLA")  # 20 z layers
surf <- r[[1]]  # surface CHLA

plot(log10(surf), main = "log10 Surface CHLA (z = 1)")
```

## Suggested citation

> Holmes, E. (2025). *CHLA-Z: Global chlorophyll-a vertical distribution (0–200 m) derived from PACE OCI and Bio-Argo using a BRT model*.  
> NOAA Fisheries (NMFS). Zarr v3 / NetCDF. (draft)

---

## Status

**Research / draft product.**  
Validation and uncertainty characterization are in progress.  
Variable names, grid, and access paths are expected to remain stable; algorithm details may evolve.

---

## License and attribution

Inputs include **PACE OCI Level-3 mapped remote-sensing reflectances** (NASA/GSFC/OBPG) and **Bio-Argo** profile observations.

Dataset and code are provided under **CC-BY 4.0** unless otherwise noted.

---

### NOAA Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is
provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of
Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed
by all applicable Federal law. Any reference to specific commercial products, processes, or services by service
mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or
favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a
DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by
DOC or the United States Government.
