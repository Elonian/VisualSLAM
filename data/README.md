Place Project 3 datasets in this layout:

- data/dataset00/dataset00.npy
- data/dataset00/dataset00_imgs.npy
- data/dataset02/dataset02.npy
- data/dataset02/dataset02_imgs.npy

Only the datasets that are actually present need to exist. The scripts auto-discover available `datasetXX` folders when `--datasets` is omitted.

The `.npy` file should contain keys:
`v_t`, `w_t`, `timestamps`, `features` (optional for dataset02),
`K_l`, `K_r`, `extL_T_imu`, `extR_T_imu`.

Accepted image keys inside `datasetXX_imgs.npy` include `left/right`, `cam0/cam1`, and `cam_imgs_L/cam_imgs_R`.
