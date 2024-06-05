import os
import argparse
from glob import glob

import torch

from utils.utils import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='sample/cat1.png', help="img file path")
    parser.add_argument('--prompt', type=str, help="source(reference) prompt")
    parser.add_argument('--trg_prompt', type=str, nargs='+', help="target prompt")
    parser.add_argument('--num_inference_steps', type=int, default=200, help="inference steps")
    parser.add_argument('--save_path', type=str, default='results', help="save directory")
    parser.add_argument('--experiment_name', type=str, default='experiment_1', help="save directory")
    parser.add_argument('--w_cut', type=float, default=3.0, help="weight coefficient for cut loss term")
    parser.add_argument('--w_dds', type=float, default=1.0, help="weight coefficient for dds loss term")
    parser.add_argument('--patch_size', type=int, nargs='+', default=[1,2], help="size of patches")
    parser.add_argument('--n_patches', type=int, default=256, help="number of patches")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--v5', action='store_true', default=False)
    parser.add_argument("--torch_dtype", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="dtype for less vram memory")
    
    parser.add_argument('--portion_update', action='store_true', default=False)
    parser.add_argument('--use_cut', action='store_true', default=False)
    parser.add_argument('--structure_conserve', action='store_true', default=False)
    parser.add_argument('--ssim_coeff', type=float, default=1000.0)
    args = parser.parse_args()

    # Prepare model
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_model(args)

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    if os.path.isdir(args.img_path):
        img_files = sorted(glob(os.path.join(args.img_path, '*.jpg')))
    else:
        img_files = [args.img_path]
    
    save_path = os.path.join(args.save_path, os.path.basename(args.img_path).split('.')[0], args.experiment_name)
    # Inference
    for img_file in img_files:
        print(img_file)
        
        result, result_replace = stable(
            img_path=img_file,
            prompt=args.prompt,
            trg_prompt=args.trg_prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            save_path=save_path,
            
            portion_update=args.portion_update,
            use_cut=args.use_cut,
            structure_conserve=args.structure_conserve,
            ssim_coeff=args.ssim_coeff,
        )

        # Save result
        result.save(os.path.join(save_path,'else', os.path.basename(img_file)))
        result_replace.save(os.path.join(save_path,'replace_luminance', os.path.basename(img_file)))

        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            f.write(f'{args.prompt}\n')
            f.write(f'{args.trg_prompt}\n')
            f.write(f'portion_update: {args.portion_update}\n')
            f.write(f'use_cut: {args.use_cut}\n')
            f.write(f'structure_conserve: {args.structure_conserve}\n')
            f.write(f'ssim_coeff: {args.ssim_coeff}\n')

if __name__ == '__main__':    
    main()
