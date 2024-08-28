# walnut

## Data

Download the entire Centrum Wiskunde & Informatica (CWI) Walnut Dataset and construct the ground truth volume with the following:

```zsh
zsh data.sh
```

**Note:** This dataset takes ~10 hrs to download and requires ~300 GB (48 walnuts with ~3,600 high-resolution X-ray projections per walnut). Constructing the ground truth volume takes ~90 min / walnut on an NVIDIA TITAN Xp.

Optionally, you can construct ground truth volumes with a subset of projections using the following command (in this example, we subsample the input data by 25X):

```zsh
srun python utils/construct_ground_truth.py -d data -s 25
```

This runs in about ~4 min / walnut on an NVIDIA TITAN Xp.

## getting source and target points from the dataset

the raw data contains files like `scan_geom_corrected.geom` --- these specify the geometry of each projection image

each row of this file contains 12 numbers that specify the geometry of a particular image

these numbers can be turned into `source` and `target` pointset that can be passed to DiffDRR with the following script:

```python
import numpy as np
import torch

projs_rows = 972  # Image height
projs_cols = 768  # Image width

tube = 1  # The second acquisition is closest to a central cone
vecs = np.loadtxt(f"tubeV{tube}/scan_geom_corrected.geom")

idx = 0  # Parse the parameters of the first acquisition
src = vecs[idx, :3]  # X-ray source
det = vecs[idx, 3:6]  # Center of the detector plane
u = vecs[idx, 6:9]  # Basis vector one of the detector plane
v = vecs[idx, 9:12]  # Basis vector two of the detector plane

src = torch.from_numpy(src).to(torch.float32)
det = torch.from_numpy(det).to(torch.float32)
u = torch.from_numpy(u).to(torch.float32)
v = torch.from_numpy(v).to(torch.float32)

# Create a canonical basis for the detector plane
rows = torch.arange(-projs_rows // 2, projs_rows // 2) + 0.5 if projs_rows % 2 == 0 else 1.0
cols = torch.arange(-projs_cols // 2, projs_cols // 2) + 0.5 if projs_cols % 2 == 0 else 1.0

# Change of basis to u and v from the dataset
i, j = torch.meshgrid(rows, cols, indexing="ij")
x = torch.einsum("ij, n -> ijn", j, u)
y = torch.einsum("ij, n -> ijn", i, v)

# Move the center of the detector plane to `det`
source = src
target = det + x + y
```
