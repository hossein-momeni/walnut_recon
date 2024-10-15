import time
from pathlib import Path

import astra
import click
import imageio.v2 as imageio
import nibabel as nib
import numpy as np
import submitit
from tqdm import tqdm
import pandas as pd


class AcceleratedGradientPlugin(astra.plugin.base):
    """
    Accelerated Gradient Descend a la Nesterov.

    MinConstraint : constrain values to at least this (optional)
    MaxConstraint : constrain values to at most this (optional)
    """

    astra_name = "AGD-PLUGIN"

    def initialize(self, cfg, liptschitz=1, MinConstraint=None, MaxConstraint=None):
        self.W = astra.OpTomo(cfg["ProjectorId"])
        self.vid = cfg["ReconstructionDataId"]
        self.sid = cfg["ProjectionDataId"]
        self.min_constraint = MinConstraint
        self.max_constraint = MaxConstraint

        try:
            v = astra.data2d.get_shared(self.vid)
            s = astra.data2d.get_shared(self.sid)
            self.data_mod = astra.data2d
        except Exception:
            v = astra.data3d.get_shared(self.vid)
            s = astra.data3d.get_shared(self.sid)
            self.data_mod = astra.data3d

        self.liptschitz = self.power_iteration(self.W, 10)
        self.nu = 1 / self.liptschitz

        self.ATy = self.W.BP(s)
        self.obj_func = None
        print("→ plugin initialized")

    def power_iteration(self, A, num_simulations):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = np.random.rand(A.shape[1])
        b_k1_norm = 1

        print("→ running power iteration to determine step size")
        for i in range(num_simulations):

            # calculate the matrix-by-vector product Ab
            b_k1 = A.T * A * b_k

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def run(self, its):
        v = self.data_mod.get_shared(self.vid)
        s = self.data_mod.get_shared(self.sid)
        W = self.W
        ATy = self.ATy
        x_apgd = v
        nu = self.nu

        # New variables
        t_acc = 1
        x_old = x_apgd.copy()
        NRMx = np.zeros_like(v)  # normal operator A'*A
        NRMx_old = NRMx.copy()
        gradient = NRMx - ATy

        self.obj_func = np.zeros(its)

        print(
            f"→ running {str(its)} iterations of Accelerated Gradient descent",
        )
        for i in tqdm(range(its), desc="Reconstruction iteration"):

            tau = (t_acc - 1) / (t_acc + 2)
            t_acc = t_acc + 1

            # Compute descent direction
            descent_direction = (
                gradient - tau / nu * (x_apgd - x_old) + tau * (NRMx - NRMx_old)
            )

            # update x
            x_old[:] = x_apgd[:]
            x_apgd -= nu * descent_direction
            if self.min_constraint is not None or self.max_constraint is not None:
                x_apgd.clip(min=self.min_constraint, max=self.max_constraint, out=v)

            # Compute all other updates
            Wx = W.FP(x_apgd)
            NRMx_old[:] = NRMx[:]
            NRMx = W.BP(Wx)
            gradient = NRMx - ATy
            self.obj_func[i] = 0.5 * np.linalg.norm(Wx - s) ** 2

            if self.obj_func[i] > np.min(self.obj_func[0 : i + 1]):
                t_acc = 1
                # restart acceleration
                print("→ acceleration restarted!")


def trafo(image):
    """
    A transformation to apply to each image. Converts an image from the
    raw scanner output to the form described by the projection geometry.
    """
    return np.transpose(np.flipud(image))


def load(
    walnut_id,
    proj_rows,
    proj_cols,
    subsample,
    orbits_to_recon,
    geometry_filename="scan_geom_corrected.geom",
    dark_filename="di000000.tif",
    flat_filenames=["io000000.tif", "io000001.tif"],
):
    """Load and preprocess raw projection data."""

    # Create a numpy array to geometry projection data
    projs = np.zeros((proj_rows, 0, proj_cols), dtype=np.float32)

    # And create a numpy array to projection geometry
    vecs = np.zeros((0, 12), dtype=np.float32)
    orbit = range(0, 1200, subsample)
    n_projs_orbit = len(orbit)

    # Projection file indices, reversed due to portrait mode acquisition
    projs_idx = range(1200, 0, -subsample)

    # Read the images and geometry from each acquisition
    for orbit_id in orbits_to_recon:

        # Load the scan geometry
        orbit_datapath = Path(f"/data/vision/polina/scratch/walnut/data/Walnut{walnut_id}/Projections") / f"tubeV{orbit_id}"
        vecs_orbit = np.loadtxt(orbit_datapath / f"{geometry_filename}")
        vecs = np.concatenate((vecs, vecs_orbit[orbit]), axis=0)

        # Load flat-field and dark-fields
        dark = trafo(imageio.imread(orbit_datapath / dark_filename))
        flat = np.zeros((2, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(flat_filenames):
            flat[idx] = trafo(imageio.imread(orbit_datapath / fn))
        flat = np.mean(flat, axis=0)

        # Load projection data directly on the big projection array
        projs_orbit = np.zeros((n_projs_orbit, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(tqdm(projs_idx, desc=f"Loading images (tube {orbit_id})")):
            projs_orbit[idx] = trafo(
                imageio.imread(orbit_datapath / f"scan_{fn:06}.tif")
            )

        # Preprocess the projection data
        projs_orbit -= dark
        projs_orbit /= flat - dark
        np.log(projs_orbit, out=projs_orbit)
        np.negative(projs_orbit, out=projs_orbit)

        # Permute data to ASTRA convention
        projs_orbit = np.transpose(projs_orbit, (1, 0, 2))
        projs = np.concatenate((projs, projs_orbit), axis=1)
        del projs_orbit

    projs = np.ascontiguousarray(projs)
    return projs, vecs


def reconstruct(walnut_id, n_views, algorithm, n_itrs, proj_rows=972, proj_cols=768):

    subsample = 1200 // n_views
    projs, vecs = load(walnut_id, proj_rows, proj_cols, subsample=subsample, orbits_to_recon=[2])
    assert len(vecs) == n_views

    # Specify the size of the reconstructed volume (in voxels) and
    # the size of a cubic voxel (in mm)
    voxel_per_mm = 10
    vol_sz = 3 * (50 * voxel_per_mm + 1,)
    vox_sz = 1 / voxel_per_mm
    vol_rec = np.zeros(vol_sz, dtype=np.float32)
    
    # Specify the geometry of the reconstructed volume
    vol_geom = astra.create_vol_geom(vol_sz)
    vol_geom["option"]["WindowMinX"] = vol_geom["option"]["WindowMinX"] * vox_sz
    vol_geom["option"]["WindowMaxX"] = vol_geom["option"]["WindowMaxX"] * vox_sz
    vol_geom["option"]["WindowMinY"] = vol_geom["option"]["WindowMinY"] * vox_sz
    vol_geom["option"]["WindowMaxY"] = vol_geom["option"]["WindowMaxY"] * vox_sz
    vol_geom["option"]["WindowMinZ"] = vol_geom["option"]["WindowMinZ"] * vox_sz
    vol_geom["option"]["WindowMaxZ"] = vol_geom["option"]["WindowMaxZ"] * vox_sz
    
    # Specify the geometry of the cone-beam projector
    proj_geom = astra.create_proj_geom("cone_vec", proj_rows, proj_cols, vecs)
    
    # Register the volume and projection geometries
    vol_id = astra.data3d.link("-vol", vol_geom, vol_rec)
    proj_id = astra.data3d.link("-sino", proj_geom, projs)
    projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)
    
    # Create an ASTRA configuration for reconstruction
    if algorithm == "fdk":
        cfg_agd = astra.astra_dict("FDK_CUDA")
    elif algorithm == "cgls":
        cfg_agd = astra.astra_dict("CGLS3D_CUDA")
    elif algorithm == "sirt":
        cfg_agd = astra.astra_dict("SIRT3D_CUDA")
    elif algorithm == "nesterov":
        astra.plugin.register(AcceleratedGradientPlugin)
        cfg_agd = astra.astra_dict("AGD-PLUGIN")
    else:
        raise ValueError

    start = time.time()
    cfg_agd["ProjectionDataId"] = proj_id
    cfg_agd["ReconstructionDataId"] = vol_id
    cfg_agd["ProjectorId"] = projector_id
    cfg_agd["option"] = {}
    cfg_agd["option"]["MinConstraint"] = 0
    alg_id = astra.algorithm.create(cfg_agd)
    
    # Run reconstruct and return the reconstructed volume
    astra.algorithm.run(alg_id, n_itrs)
    stop = time.time()
    
    volume = astra.data3d.get(vol_id)
    affine = np.eye(4)
    affine[[0, 1, 2], [0, 1, 2]] = vox_sz
    affine[:3, 3] = (-np.array(vol_sz) + 1) / voxel_per_mm / 2
    nifti = nib.Nifti1Image(volume, affine)
    
    # Clean up ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)
    astra.projector.delete(projector_id)
    astra.functions.clear()

    return nifti, stop - start


def main(walnut_id, n_views, algorithm):
    if algorithm == "fdk":
        n_itrs = 1
    elif algorithm == "cgls":
        n_itrs = 20
    elif algorithm == "sirt":
        n_itrs = 500
    elif algorithm == "nesterov":
        n_itrs = 50
    else:
        raise ValueError

    nifti, runtime = reconstruct(walnut_id, n_views, algorithm, n_itrs)
    savepath = Path(f"csvs/baseline_runtime/Walnut{walnut_id}_{n_views}_{algorithm}.csv")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    # nib.save(nifti, savepath)
    df = pd.DataFrame([walnut_id, n_views, algorithm, runtime])
    df.to_csv(savepath, index=False)


if __name__ == "__main__":

    walnut_ids = list(range(3, 43))
    n_views = [30]
    algorithms = ["fdk", "cgls", "sirt", "nesterov"]

    walnut_id = []
    n_view = []
    algorithm = []
    for w in walnut_ids:
        for n in n_views:
            for a in algorithms:
                if Path(f"csvs/baseline_runtime/Walnut{w}_{n}_{a}.csv").exists():
                    continue
                walnut_id.append(w)
                n_view.append(n)
                algorithm.append(a)

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="walnut",
        gpus_per_node=1,
        mem_gb=10,
        slurm_array_parallelism=len(walnut_id),
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, walnut_id, n_view, algorithm)
