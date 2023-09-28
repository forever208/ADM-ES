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
    logger.log(f"timesteps: {args.timestep_respacing}")
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

    logger.log("sampling...")
    all_images = []
    # all_x_0 = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:

        # call forward diffusion to get x_T based on x_0
        x_0, _ = next(data)
        x_0 = x_0.to(dist_util.dev())
        ones = th.ones(x_0.shape[0]).long().to(dist_util.dev())
        t = ones * args.forward_t
        epsilon = th.randn_like(x_0)
        x_T = diffusion.q_sample(x_0, t, epsilon)

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            x_T=x_T,
            forward_t=args.forward_t,
            x_0=x_0,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    if dist.get_rank() == 0:
        # compute x_t dist variance error
        timesteps = []
        gt_std = []
        pred_std = []
        delta_t = []

        logger.log(f"array keys:{diffusion.x_t_stochas_part.keys()}")
        for t, array in diffusion.x_t_stochas_part.items():
            t = int(t)
            # noise schedule of x_t
            ground_truth_std = diffusion.sqrt_one_minus_alphas_cumprod
            logger.log(f"gt x_{t} std:{ground_truth_std[t]}")

            means = []
            stds = []
            data = diffusion.x_t_stochas_part[str(t)]

            # compute the mean and std for each timestep t among 50k predictions
            for c in range(data.shape[1]):
                for h in range(data.shape[2]):
                    for w in range(data.shape[3]):
                        pixel_data = data[:, c, h, w]
                        mean = np.mean(pixel_data)
                        std = np.std(pixel_data)
                        means.append(mean)
                        stds.append(std)
            pred_x_t_std = sum(stds) / len(stds)
            logger.log(f"pred x_{t} std:{pred_x_t_std}, mean:{sum(means) / len(means)}")

            if int(t + 1) not in timesteps:
                timesteps.append(int(t + 1))
                gt_std.append(ground_truth_std[t])
                pred_std.append(pred_x_t_std)
                delta_t.append((pred_x_t_std - ground_truth_std[t])**2)
                logger.log(f"timestep {t} added into plot")
                logger.log(f"")

        logger.log(f"timesteps: {timesteps}")
        logger.log(f"ground truth std: {gt_std}")
        logger.log(f"pred std: {pred_std}")
        logger.log(f"delta_t: {delta_t}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        forward_t=19,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
