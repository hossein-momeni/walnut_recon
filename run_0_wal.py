import wandb
import torch
from tqdm import tqdm
from monai.metrics import SSIMMetric, PSNRMetric
import argparse
from datetime import datetime


from utils.helper import dclamp, get_source_target_vec, load, Reconstruction, Dataset, FastTensorDataLoader, TVLoss3D
        

normalize = lambda x: (x - x.min()) / (x.max() - x.min())
def initialize(walnut_id, poses, downsample=1, batch_size=1_600_000):
    projections, sources, targets, subject = Dataset(walnut_id=walnut_id, downsample=downsample, poses=poses).get_data()
    return FastTensorDataLoader(sources, targets, projections, subject, batch_size=batch_size)


def optimize(walnut_id, poses, downsample, batch_size, n_itr, lr, lr_tv, shift, loss_fn, drr_params, density_regulator, tv_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = initialize(walnut_id=walnut_id, poses=poses, downsample=downsample, batch_size=batch_size)
    recon = Reconstruction(dataloader.subject, device, drr_params, shift, density_regulator)
    tv_calc = TVLoss3D(lr_tv, tv_type)
 

    optimizer = torch.optim.Adam(recon.parameters(), lr=lr)
    if loss_fn == "l1":
        criterion = torch.nn.L1Loss()
    elif loss_fn == "l2":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unrecognized loss function : {loss_fn}")
    
    
    subject_volume = dataloader.subject.volume.data.cuda()

    # max_val = subject_volume.max()
    max_val = 1.0
    ssim_calc = SSIMMetric(3, max_val)
    psnr_calc = PSNRMetric(max_val)


    losses = []
    tvs = []
    ssims = []
    psnrs = []
    for itr in (pbar := tqdm(range(n_itr), ncols=100)):
        for source, target, gt in dataloader:
            optimizer.zero_grad()
            est = recon(source.cuda(), target.cuda())
            tv_norm = tv_calc(recon.density[None, None])
            loss = criterion(est, gt.cuda()) + tv_norm
            loss.backward()
            optimizer.step()
            pbar.set_description(f"loss : {loss.item():.06f} tv : {tv_norm.item():06f}")
            losses.append(loss.item())
            tvs.append(tv_norm.item())
        ssim = ssim_calc(normalize(recon.density[None, None]), normalize(subject_volume[None]))
        psnr = psnr_calc(normalize(recon.density[None, None]), normalize(subject_volume[None]))
        ssims.append(ssim)
        psnrs.append(psnr)
        wandb.log({"loss": loss.item(), "tv_loss": tv_norm.item(), "ssim": ssim, "psnr": psnr})
    return recon.density, losses, tvs, ssims, psnrs
    

def run(
        walnut_id,
        poses,
        downsample=1,
        batch_size=1_600_000,
        n_itr=100,
        lr=1e-1,
        lr_tv=1e2,
        shift=5.0,
        loss_fn="l1",
        drr_params={'renderer': 'trilinear', 'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None, 'n_points': 500},
        density_regulator='sigmoid',
        tv_type='vl1'
):
    now_time = datetime.now().strftime("%m-%d__%H:%M")
    wandb.login(key='611398a058ddff8103235b675346fafee38358d5')
    settings = wandb.Settings(job_name=f"{poses}_{now_time}")
    wandb.init(
        project="walnut",
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
        },
        settings=settings
        )
    density, losses, tvs, set_ssim, set_psnr = optimize(
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
    )
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
            }
        },
        f'/data/vision/polina/scratch/walnut/results/walnut{walnut_id}_{poses}_{wandb.run.id}.pt',
    )


def main(**kwargs):
    run(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Run optimization on walnut data")
    parser.add_argument("--walnut_id", type=int, default=1)
    # parser.add_argument("--walnut_id", type=int, required=True)
    parser.add_argument("--poses", type=int, default=4)
    # parser.add_argument("--poses", type=int, nargs="+", required=True)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1_600_000)
    parser.add_argument("--n_itr", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--lr_tv", type=float, default=1e2)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--loss_fn", type=str, default="l1")
    parser.add_argument("--drr_params", type=dict, default={'renderer': 'trilinear', 'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None, 'n_points': 500}, required=False)
    parser.add_argument("--density_regulator", type=str, default='sigmoid')
    parser.add_argument("--tv_type", type=str, default='vl1')
    args = parser.parse_args()
    main(**vars(args))
    
