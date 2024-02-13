import os
import argparse
from glob import glob

import torch

from pipeline_cds import CDSPipeline

def load_model(args):
    if not args.v5:
        sd_version = "CompVis/stable-diffusion-v1-4"
    else:
        sd_version = "runwayml/stable-diffusion-v1-5"
    stable = CDSPipeline.from_pretrained(sd_version)
    return stable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='sample/cat1.png', help="img file path")
    parser.add_argument('--prompt', type=str, help="source(reference) prompt")
    parser.add_argument('--trg_prompt', type=str, nargs='+', help="target prompt")
    parser.add_argument('--save_path', type=str, default='results', help="save directory")
    parser.add_argument('--w_cut', type=float, default=3.0, help="weight coefficient for cut loss term")
    parser.add_argument('--w_dds', type=float, default=1.0, help="weight coefficient for dds loss term")
    parser.add_argument('--patch_size', type=int, nargs='+', default=[1,2], help="size of patches")
    parser.add_argument('--n_patches', type=int, default=256, help="number of patches")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--v5', action='store_true', default=False)
    args = parser.parse_args()

    # Prepare model
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_model(args)

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    img_files = sorted(glob(os.path.join(args.img_path, '*.jpg')))

    # Inference
    for img_file in img_files:
        print(img_file)
        
        result = stable(
            img_path=img_file,
            prompt=args.prompt,
            trg_prompt=args.trg_prompt,
            generator=generator,
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            save_path=args.save_path,
        )

        # Save result
        result.save(os.path.join(args.save_path, os.path.basename(img_file)))

if __name__ == '__main__':    
    main()
