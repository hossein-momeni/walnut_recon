import wandb
import torch
from tqdm import tqdm
from monai.metrics import SSIMMetric, PSNRMetric
from monai.losses import LocalNormalizedCrossCorrelationLoss
import argparse
from datetime import datetime
from torchmetrics.regression import PearsonCorrCoef
from pathlib import Path
import time

from utils.helper import dclamp, Reconstruction, Dataset, FastTensorDataLoader, TVLoss3D, z_norm


normalize = lambda x: (x - x.min()) / (x.max() - x.min())
def initialize(walnut_id, poses, downsample=1, batch_size=1_600_000, half_orbit=False):
    projections, sources, targets, subject = Dataset(walnut_id=walnut_id, downsample=downsample, poses=poses, half_orbit=half_orbit).get_data()
    return FastTensorDataLoader(sources, targets, projections, subject, batch_size=batch_size)


def optimize(walnut_id, poses, downsample, batch_size, n_itr, lr, lr_tv, shift, loss_fn, drr_params, density_regulator, tv_type, half_orbit, drr_scale):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    print(f"Using device: {device}")

    dataloader = initialize(walnut_id=walnut_id, poses=poses, downsample=downsample, batch_size=batch_size, half_orbit=half_orbit)
    recon = Reconstruction(dataloader.subject, device, drr_params, shift, density_regulator) 

    optimizer = torch.optim.Adam(recon.parameters(), lr=lr)
    if loss_fn == "l1":
        criterion = torch.nn.L1Loss()
    elif loss_fn == "l2":
        criterion = torch.nn.MSELoss()
    elif loss_fn == "pcc":
        Warning("Using PCC loss, work in progress")
        criterion = PearsonCorrCoef()
    elif loss_fn == 'ncc':
        KeyError("NCC loss not implemented")
        criterion = LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
        
    else:
        raise ValueError(f"Unrecognized loss function : {loss_fn}")
    
    
    subject_volume = dataloader.subject.volume.data.cuda()

    # max_val = subject_volume.max()
    max_val = (subject_volume).max()
    ssim_calc = SSIMMetric(3, max_val)
    psnr_calc = PSNRMetric(max_val)
    pcc_calc = PearsonCorrCoef().to(device)
    mse_calc = torch.nn.MSELoss()
    # ncc_calc = LocalNormalizedCrossCorrelationLoss(spatial_dims=3)

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=n_itr, power=1.0)
    gamma = 0.95
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    losses = []
    ssims = []
    psnrs = []
    pccs = []
    time_deltas = []
    for itr in (pbar := tqdm(range(n_itr), ncols=100)):
        start_time = time.perf_counter()
        for source, target, gt in dataloader:
            optimizer.zero_grad()
            est = recon(source.cuda(), target.cuda())
            loss = criterion(drr_scale * est, gt.cuda())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        end_time = time.perf_counter()
        time_deltas.append(end_time - start_time)
        pbar.set_description(f"loss : {loss.item():.06f}")
        lr_scheduler.step()
        # ssim = ssim_calc(recon.density[None, None], subject_volume[None])
        psnr = psnr_calc(recon.density[None, None], subject_volume[None])
        pcc = pcc_calc(recon.density.flatten(), subject_volume.flatten())
        mse = mse_calc(recon.density[None, None], subject_volume[None])
        # ncc = ncc_calc(recon.density[None, None], subject_volume[None]).cpu()
        # ssims.append(ssim.item())
        psnrs.append(psnr.item())
        pccs.append(pcc.item())
    
        wandb.log({"loss": loss.item(), "psnr": psnr.item(), 'pcc': pcc.item(), 'vol_mse': mse, 'lr_decay': lr_scheduler.get_last_lr()[0]})
    ssims.append(ssim_calc(recon.density[None, None], subject_volume[None]).item())
    tvs = []
    return recon.density, losses, tvs, ssims, psnrs, pccs, time_deltas
    

def run(
        walnut_id,
        poses,
        downsample,
        batch_size,
        half_orbit=False,
        n_itr=100,
        lr=1e-1,
        lr_tv=1e2,
        shift=5.0,
        loss_fn="l1",
        drr_params={'renderer': 'trilinear', 'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None, 'n_points': 500},
        density_regulator='sigmoid',
        tv_type='vl1',
        drr_scale=1.0,
        proj_name='dynamic_tv_scaled',
        **kwargs,
):
    drr_params['n_points'] = kwargs.get('n_points', 500)
    drr_params['renderer'] = kwargs.get('renderer', 'trilinear')
    # proj_name = 'dynamic_tv_scaled'
    now_time = datetime.now().strftime("%m-%d__%H:%M")
    wandb.login() # replace your wandb key here!
    wandb.init(
        project=proj_name,
        config={
            "walnut_id": walnut_id,
            "poses": poses,
            "downsample": downsample,
            "batch_size": batch_size,
            "n_itr": n_itr,
            "lr": lr,
            "lr_tv": lr_tv,
            "shift": shift,
            "loss_fn": loss_fn,
            "drr_params": drr_params,
            "density_regulator": density_regulator,
            "tv_type": tv_type,
            "half_orbit": half_orbit,
            "drr_scale": drr_scale,
            "lr_scheduler_decay": 1,
        },
        name = f"w{walnut_id}_{poses}_{drr_params['renderer']}_{now_time}",
        )
    density, losses, tvs, set_ssim, set_psnr, set_pcc, set_times = optimize(
        walnut_id,
        poses,
        downsample,
        batch_size,
        n_itr,
        lr, 
        lr_tv, 
        shift, 
        loss_fn, 
        drr_params, 
        density_regulator, 
        tv_type,
        half_orbit,
        drr_scale
    )

    total_time = sum(set_times)
    wandb.run.summary['total_time'] = total_time / 60
    wandb.run.summary['ssim'] = set_ssim[-1]
    save_loc = Path(f'/data/vision/polina/scratch/walnut/results/{proj_name}')
    save_loc.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'tensors':{
                'est': density.cpu(), 
            },
            'metrics':{
                'loss': losses,
                'tv': tvs,
                'ssim': set_ssim,
                'psnr': set_psnr,
                'pcc': set_pcc,
                'tota_time': total_time,
                'time_delta': set_times,
            },
            'hyperparameters':{
                "walnut_id": walnut_id,
                "poses": poses,
                "downsample": downsample,
                "batch_size": batch_size,
                "n_itr": n_itr,
                "lr": lr,
                "lr_tv": lr_tv,
                "shift": shift,
                "loss_fn": loss_fn,
                "drr_params": drr_params,
                "density_regulator": density_regulator,
                "tv_type": tv_type,
                "half_orbit": half_orbit,
            }
        },
        save_loc / f'walnut{walnut_id}_{poses}_{drr_params['renderer']}_{wandb.run.id}.pt',

    )


def main(**kwargs):
    run(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Run optimization on walnut data")
    parser.add_argument("--walnut_id", type=int, default=1)
    # parser.add_argument("--walnut_id", type=int, required=True)
    parser.add_argument("--poses", type=int, default=30)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2_000_000)
    parser.add_argument("--n_itr", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--lr_tv", type=float, default=15)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--loss_fn", type=str, default="l1")
    parser.add_argument("--renderer", type=str, default='trilinear')
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--drr_params", type=dict, default={'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None}, required=False)
    parser.add_argument("--density_regulator", type=str, default='sigmoid')
    parser.add_argument("--tv_type", type=str, default='vl1')
    parser.add_argument("--half_orbit", type=bool, default=False)
    parser.add_argument("--drr_scale", type=float, default=1.0)
    parser.add_argument("--proj_name", type=str, default='grand_experiment')
    args = parser.parse_args()
    main(**vars(args))
    
