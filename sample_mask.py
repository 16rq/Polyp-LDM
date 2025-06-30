import argparse
import os
import sys
import glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import cv2
import pandas as pd

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_Noise
from ldm.models.diffusion.plms import PLMSSampler
import csv
# change working space
os.chdir("/data6/ryqiu/latent-diffusion/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
### txt2img with modification.
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def count_connected_components(image_path):
    image = cv2.imread(image_path, 0)  # 以灰度图像读入
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, _ = cv2.connectedComponents(thresh)
    return num_labels - 1  # 减去背景

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="/data6/ryqiu/latent-diffusion/NoiseDiscriminator/output"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=500,
        help="number of ldm sampling steps",
    )

    parser.add_argument( 
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/rdm/rdm768x768/model.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval-augmented-diffusion/768x768.yaml",
        help="path to config which constructs model",
    )

    opt = parser.parse_args()
    opt.n_samples = 1
    opt.prompt = '1'
    opt.config = '/data6/ryqiu/latent-diffusion/configs/autoencoder/autoencoder_kl_32x32x4_rimage_onlyDecoder.yaml'
    # opt.config = "/data6/ryqiu/latent-diffusion/configs/latent-diffusion/config_satge1_txt2mask_baseline.yaml"
    opt.ckpt = "/data6/ryqiu/latent-diffusion/ldm/logs/2024-10-28T19-58-49_config_ldm1_GMS_color/checkpoints/epoch=000157.ckpt"
    opt.outdir = '/data6/ryqiu/latent-diffusion/outputs/{}/mask/'.format(opt.prompt)
    parent_dir = '/data6/ryqiu/latent-diffusion/outputs/'
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # 
    opt.gpus = [0,]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        # sampler = DDIMSampler(model) # with
        sampler = DDIMSampler(model) # w/o noisediscriminant


    outpath = opt.outdir
    prompt = opt.prompt

    os.makedirs(outpath, exist_ok=True)
    base_count = len(os.listdir(outpath))

    all_samples = list()
    polyp_counts = []

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            

            for n in trange(400, desc="Sampling"):
                start = 0
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H // 8, opt.W // 8]
                samples_ddim, _ = sampler.sample(prompt = opt.prompt,S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)


                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    image_path = os.path.join(outpath, '{}n{}.png'.format(prompt, start+n))                  
                    Image.fromarray(x_sample.astype(np.uint8)).save(image_path)
                    polyp_count = count_connected_components(image_path)
                    polyp_counts.append({"Image": n, "Polyp Count": polyp_count})
                    base_count += 1
                all_samples.append(x_samples_ddim)

    df = pd.DataFrame(polyp_counts)
    df.to_excel(os.path.join(parent_dir, "polyp_{}.xlsx".format(prompt)), index=False)
    
    # Save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
