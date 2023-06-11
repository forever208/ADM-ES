"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from guided_diffusion.image_datasets import load_data


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_path = args.model_path
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    logger.log(f"loading checkpoint: {model_path}")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"creating data loader from {args.data_dir}...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    all_labels = []
    x_t_stochas_part = {}

    for timestep in range(550, 1000, 50):
        num_iteration = 0
        logger.log(f" ")
        logger.log(f"computing pred_x_t distribution for timestep {timestep}...")

        with th.no_grad():
            while num_iteration * args.batch_size < args.num_samples:
                num_iteration += 1

                # call forward diffusion to get x_T based on x_0
                x_0, _ = next(data)
                x_0 = x_0.to(dist_util.dev())
                ones = th.ones(x_0.shape[0]).long().to(dist_util.dev())
                t = ones * timestep
                epsilon = th.randn_like(x_0)
                x_t = diffusion.q_sample(x_0, t, epsilon)

                model_kwargs = {}
                if args.class_cond:
                    classes = th.randint(
                        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                    )
                    model_kwargs["y"] = classes
                sample_fn = (
                    diffusion.p_sample if not args.use_ddim else diffusion.ddim_sample
                )

                out = sample_fn(
                    model,
                    x_t,
                    t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    x_0=x_0,
                )
                x_t_prev = out["sample"]
                sqrt_alpha_bar_x0 = out["sqrt_alpha_bar_x0"]
                stochas_part = x_t_prev - sqrt_alpha_bar_x0  # x_t_prev: (sqrt_one_minus_alpha_bar) * N(0, I)
                stochas_part = stochas_part.contiguous().cpu().numpy()

                if str(timestep-1) in x_t_stochas_part.keys():
                    x_t_stochas_part[str(timestep-1)] = np.concatenate((x_t_stochas_part[str(timestep-1)], stochas_part),
                                                                       axis=0)
                else:
                    x_t_stochas_part[str(timestep-1)] = stochas_part
                logger.log(f"x_t_stochas_part for {timestep-1} with shape: {x_t_stochas_part[str(timestep-1)].shape}")

                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"x_t_stochas_part.npz")
        np.savez(out_path, **x_t_stochas_part)
        logger.log(f"x_t_stochas_part saving to {out_path}")

    dist.barrier()
    logger.log("computing complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        forward_t=1000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
