import numpy as np
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio
from torchvision.transforms import Resize
import torch.nn.functional as F

from diffdrr.data import read
from diffdrr.drr import DRR

from .construct_ground_truth import load as load_gt


def get_source_target_vec(vecs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the source and target vectors from the projection geometry.
    """
    projs_rows = 972  # Image height
    projs_cols = 768  # Image width

    sources = []
    targets = []
    for idx in range(len(vecs)):
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
        x = torch.einsum("ij, n -> ijn", j, -u)
        y = torch.einsum("ij, n -> ijn", i, v)

        # Move the center of the detector plane to `det`
        source = src
        target = det + x + y
        source = source.expand(target.shape)
        sources.append(source.flip([1,2]))
        targets.append(target.flip([1,2]))
    return sources, targets


def trafo(image: np.ndarray) -> np.ndarray:
    """
    A transformation to apply to each image. Converts an image from the
    raw scanner output to the form described by the projection geometry.
    """
    return np.transpose(np.flipud(image))


def load(
    datapath: Path,
    proj_rows: int,
    proj_cols: int,
    n_views: int, # number of views per orbit
    orbits_to_recon=[1, 2, 3],
    geometry_filename="scan_geom_corrected.geom",
    dark_filename="di000000.tif",
    flat_filenames=["io000000.tif", "io000001.tif"],
    half_orbit=False,
):
    """Load and preprocess raw projection data."""

    # Create a numpy array to geometry projection data
    projs = np.zeros((0, proj_rows, proj_cols), dtype=np.float32)

    # And create a numpy array to projection geometry
    vecs = np.zeros((0, 12), dtype=np.float32)
    if half_orbit:
        orbit = np.linspace(0, 600 - 1, n_views, endpoint=False, dtype=int)
    else:
        orbit = np.linspace(0, 1200 - 1, n_views, endpoint=False, dtype=int)
    n_projs_orbit = len(orbit)

    # Projection file indices, reversed due to portrait mode acquisition
    # projs_idx = range(1200, 0, -subsample) # this is the original method
    # if half_orbit:
    #     projs_idx = np.linspace(1200, 600, n_views, endpoint=False, dtype=int)
    # else:
    #     projs_idx = np.linspace(1200, 0, n_views, endpoint=False, dtype=int)

    


    # Read the images and geometry from each acquisition
    for orbit_id in orbits_to_recon:

        # Load the scan geometry
        orbit_datapath = datapath / f"tubeV{orbit_id}"
        vecs_orbit = np.loadtxt(orbit_datapath / f"{geometry_filename}")
        vecs_orbit = np.flip(vecs_orbit, axis=0)
        vecs = np.concatenate((vecs, vecs_orbit[orbit]), axis=0)

        # Load flat-field and dark-fields
        dark = trafo(imageio.imread(orbit_datapath / dark_filename))
        flat = np.zeros((2, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(flat_filenames):
            flat[idx] = trafo(imageio.imread(orbit_datapath / fn))
        flat = np.mean(flat, axis=0)

        # Load projection data directly on the big projection array
        projs_orbit = np.zeros((n_projs_orbit, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(tqdm(orbit, desc=f"Loading images (tube {orbit_id})")):
            projs_orbit[idx] = trafo(
                imageio.imread(orbit_datapath / f"scan_{fn:06}.tif")
            )

        # Preprocess the projection data
        projs_orbit -= dark
        projs_orbit /= flat - dark
        np.log(projs_orbit, out=projs_orbit)
        np.negative(projs_orbit, out=projs_orbit)

        projs = np.concatenate((projs, projs_orbit), axis=0)
        del projs_orbit

    projs = np.ascontiguousarray(projs)
    return projs, vecs


class TVLoss3D(torch.nn.Module):
    """
    Total variation loss for 3D data
    """
    def __init__(self, TVLoss_weight=1, norm='l2'):
        super(TVLoss3D, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.norm = norm

    def forward(self, x):
        if self.norm == 'None':
            return 0
        batch_size = x.size()[0]
        d_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        
        count_d = self._tensor_size(x[:,:,1:,:,:])
        count_h = self._tensor_size(x[:,:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,:,1:])
        
        if self.norm == 'l2':
            d_tv = torch.pow((x[:,:,1:,:,:] - x[:,:,:d_x-1,:,:]), 2).sum()
            h_tv = torch.pow((x[:,:,:,1:,:] - x[:,:,:,:h_x-1,:]), 2).sum()
            w_tv = torch.pow((x[:,:,:,:,1:] - x[:,:,:,:,:w_x-1]), 2).sum()
            return self.TVLoss_weight * 3 * (d_tv/count_d + h_tv/count_h + w_tv/count_w) / batch_size


        elif self.norm == 'l1':
            d_tv = torch.sum(torch.abs(x[:,:,1:,:,:] - x[:,:,:d_x-1,:,:]))
            h_tv = torch.sum(torch.abs(x[:,:,:,1:,:] - x[:,:,:,:h_x-1,:]))
            w_tv = torch.sum(torch.abs(x[:,:,:,:,1:] - x[:,:,:,:,:w_x-1]))
            return self.TVLoss_weight * 3 * (d_tv/count_d + h_tv/count_h + w_tv/count_w) / batch_size
            
        elif self.norm == 'nl1':
            loss = F.smooth_l1_loss(x[:,:,1:,:,:], x[:,:,:-1,:,:], reduction='mean').double() +\
                   F.smooth_l1_loss(x[:,:,:,1:,:], x[:,:,:,:-1,:], reduction='mean').double() +\
                   F.smooth_l1_loss(x[:,:,:,:,1:], x[:,:,:,:,:-1], reduction='mean').double()
            loss /= 3
            return self.TVLoss_weight * loss / batch_size
        
        elif self.norm == 'vl1': # same as l1, but consistent with vivek's implementation
            delx = x.diff(dim=-3).abs().mean()
            dely = x.diff(dim=-2).abs().mean()
            delz = x.diff(dim=-1).abs().mean()
            return self.TVLoss_weight * (delx + dely + delz) / 3 
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4] 


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    Adopted from https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


class Reconstruction(torch.nn.Module):
    """
    Main reconstruction module
    """
    def __init__(self, subject, device, drr_params: dict, shift=6, density_regulator='sigmoid'):
        super().__init__()
        self.drr_params = drr_params
        self._density = torch.nn.Parameter(torch.zeros(*subject.volume.shape, device=device)[0])
        self.drr = DRR(
            subject,
            sdd=drr_params['sdd'],
            height=drr_params['height'],
            width=drr_params['width'],
            delx=drr_params['delx'],
            renderer=drr_params['renderer'],
            patch_size=drr_params['patch_size'],
        ).to(device)
        
        if density_regulator == 'None':
            self.density_regulator = lambda x: x
        elif density_regulator == 'clamp':
            self.density_regulator = lambda x: dclamp(x, 0, 1)
        elif density_regulator == 'softplus':
            self.density_regulator = torch.nn.Softplus(10,200)
        elif density_regulator == 'sigmoid':
            self.density_regulator = lambda x: torch.sigmoid(x - shift)

    def forward(self, source, target, **kwargs):
        # source = self.drr.affine_inverse(source)
        # target = self.drr.affine_inverse(target)
        if self.drr_params['renderer'] == 'trilinear':
            kwargs['n_points'] = self.drr_params['n_points']
        img = self.drr.renderer(
            self.density,
            source,
            target,
            **kwargs,
        )

        img *= ((target - source) * self.drr.affine.matrix[0].diag()[:3]).norm(dim=-1).unsqueeze(1)
        return img
    
    @property
    def density(self):
        return self.density_regulator(self._density)
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, walnut_id, tube=[2], downsample=1, poses=30, half_orbit=False):
        main_dir = Path(f'/data/vision/polina/scratch/walnut/data/Walnut{walnut_id}/')
        dir = Path(f'/data/vision/polina/scratch/walnut/data/Walnut{walnut_id}/Projections/')
        
        # self.gt_projs, vecs = load_gt(dir, 972, 768, int(1200/poses), orbits_to_recon=tube)
        self.gt_projs, vecs = load(dir, 972, 768, poses, orbits_to_recon=tube, half_orbit=half_orbit)

        self.gt_projs = torch.tensor(self.gt_projs)#.permute(1,0,2)
        self.sources, self.targets = get_source_target_vec(vecs)
        self.sources = torch.stack(self.sources)
        self.targets = torch.stack(self.targets)
        self.subject = read(main_dir / 'gt.nii.gz')

        if downsample != 1:
            Warning("downsampling is not complete yet!!!")
            resizer = Resize((972//downsample, 768//downsample))
            self.gt_projs = resizer(self.gt_projs)
            self.sources = resizer(self.sources.permute(0,3,1,2)).permute(0,2,3,1)
            self.targets = resizer(self.targets.permute(0,3,1,2)).permute(0,2,3,1)


        self.sources = self.sources.reshape(1, -1, 3)
        self.targets = self.targets.reshape(1, -1, 3)
        self.gt_projs = self.gt_projs.reshape(1, 1, -1)

       
    def __len__(self):
        return len(self.gt_projs)
    
    def __getitem__(self, idx):
        return self.gt_projs[idx], self.sources[idx], self.targets[idx]

    def get_data(self):
        return self.gt_projs, self.sources, self.targets, self.subject
    

class FastTensorDataLoader:

    def __init__(
        self, source, target, pixels, subject, batch_size=None, shuffle=True, pin_memory=True
    ):
        assert source.shape[1] == target.shape[1] == pixels.shape[2]
        self.subject = subject
        self.source = source
        self.target = target
        self.pixels = pixels
        if pin_memory:
            self.pin_memory() # for faster data transfer to GPU, set to false for better memory management

        self.batch_size = batch_size if batch_size is not None else self.__len__()
        self.shuffle = shuffle if batch_size is not None else False

        self.n_batches, remainder = divmod(self.__len__(), self.batch_size)
        self.n_batches += 1 if remainder > 0 else 0
        # if False: # masking the images (not working yet)
        #     threshold = 1e-2
        #     valid_idx = torch.nonzero(pixels.flatten() > threshold, as_tuple=True)
        #     self.source = self.source[:, valid_idx[0]]
        #     self.target = self.target[:, valid_idx[0]]
        #     self.pixels = self.pixels[..., valid_idx[0]]



    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.__len__(), dtype=torch.int32, device='cuda').cpu()
            self.source = self.source[:, indices]
            self.target = self.target[:, indices]
            self.pixels = self.pixels[..., indices]
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        source = self.source[:, self.idx : self.idx + self.batch_size]
        target = self.target[:, self.idx : self.idx + self.batch_size]
        pixels = self.pixels[..., self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return source, target, pixels

    def __len__(self):
        return self.source.shape[1]

    def pin_memory(self):
        self.source = self.source.pin_memory()
        self.target = self.target.pin_memory()
        self.pixels = self.pixels.pin_memory()

    def apply_function(self, func, device='cuda'):
        idx = 0
        while idx < self.__len__():
            self.source[:, idx : idx + self.batch_size] = func(self.source[:, idx : idx + self.batch_size].to(device)).cpu()
            self.target[:, idx : idx + self.batch_size] = func(self.target[:, idx : idx + self.batch_size].to(device)).cpu()
            idx += self.batch_size


def z_norm(x):
    return (x - x.mean()) / x.std()