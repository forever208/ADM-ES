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
    eps_dict = {}

    for timestep in range(0, diffusion.betas.shape[0], 1):
        num_iteration = 0
        logger.log(f" ")
        logger.log(f"computing pred eps for timestep {timestep}...")

        with th.no_grad():
            while num_iteration * args.batch_size < args.num_samples:
                num_iteration += 1

                # call forward diffusion to get x_t based on x_0
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

                pred_eps = out["eps"]
                pred_eps = pred_eps.contiguous().cpu().numpy()

                # compute the l2-norm for each image
                l2_norms = []
                for n in range(pred_eps.shape[0]):
                    image = pred_eps[n, :, :, :]
                    image = image.reshape(3, -1)
                    l2_norm = np.linalg.norm(image, 'fro')
                    l2_norms.append(l2_norm)
                eps_l2_norm = sum(l2_norms) / len(l2_norms)
                eps_l2_norm = np.array(eps_l2_norm).reshape([1])

                if str(timestep) in eps_dict.keys():
                    eps_dict[str(timestep)] = np.concatenate((eps_dict[str(timestep)], eps_l2_norm), axis=0)
                else:
                    eps_dict[str(timestep)] = eps_l2_norm
                logger.log(f"eps norm: {eps_l2_norm} at timestep {timestep}")

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
        # out_path = os.path.join(logger.get_dir(), f"eps_each_20allsteps.npz")
        # np.savez(out_path, **eps_error_dict)
        # logger.log(f"eps saving to {out_path}")

        l2_norms_ls = []
        for t in eps_dict:
            l2_norms_ls.append(eps_dict[t].mean())
            logger.log(f"avg eps l2 norm: {eps_dict[t].mean()} at {t} step")

        logger.log(f"eps l2 norm: {np.array(l2_norms_ls)}")
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
