"""Synthetic image generation using Stable Diffusion."""

import argparse
import os
import time

import torch
import torchvision.transforms.functional as tfunc
from diffusers import (
    DDIMScheduler,
    DPMSolverSinglestepScheduler,
    StableUnCLIPImg2ImgPipeline,
)
from torchvision import datasets, transforms

from data_generation.icgan.data_utils import utils as data_utils


class StableGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        # model
        self.model = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip-small",
            torch_dtype=torch.float16,
            variation="fp16",
        )

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        self.model.to(device)
        print(f"Using device: {device}")

        if opt.dpm:
            self.model.scheduler = DPMSolverSinglestepScheduler.from_config(
                self.model.scheduler.config, rescale_betas_zero_snr=True
            )
        else:
            self.model.scheduler = DDIMScheduler.from_config(
                self.model.scheduler.config,
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )

        print("Scheduler:", self.model.scheduler)

        # image size
        self.height = self.opt.img_size
        self.width = self.opt.img_size

        # inference steps
        self.num_inference_steps = self.opt.steps

        self.eta = self.opt.ddim_eta

        self.generator = torch.Generator()
        self.generator.manual_seed(self.opt.image_version)

    def generate(self, input_image, n_sample_per_image=1):
        synth_images = self.model(
            input_image,
            eta=self.eta,
            num_images_per_prompt=n_sample_per_image,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,
        ).images
        trans = transforms.Resize(size=(self.height, self.width))
        return [trans(img) for img in synth_images]


class ImageNetWithFilenames(datasets.ImageNet):
    def __getitem__(self, index):
        filename, _ = self.imgs[index]
        image = self.loader(os.path.join(self.root, filename))
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "filename": filename}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument("--img_size", type=int, default=224, help="image saving size")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--dpm",
        action="store_true",
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="use ddim sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for inference",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Number of shards to split the dataset.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        help="Index of the shard",
    )
    parser.add_argument(
        "--image_version",
        type=int,
        help="Version and seed for generated images.",
    )
    parser.add_argument(
        "--counter",
        type=int,
        help="Counter.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size.",
    )
    opt = parser.parse_args()
    print(opt)

    if opt.outdir is not None:
        os.makedirs(opt.outdir, exist_ok=True)

    transform_list = [
        data_utils.CenterCropLongEdge(),
        transforms.Resize(size=(opt.img_size, opt.img_size)),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform_list)

    imagenet_dir = "/scratch/ssd004/datasets/imagenet256"

    imagenet_dataset = ImageNetWithFilenames(
        root=imagenet_dir, split="train", transform=transform
    )
    dl_generator = torch.Generator()
    dl_generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        generator=dl_generator,
    )

    stable_generator = StableGenerator(opt)
    n = len(imagenet_dataset)
    print(f"Total number of images: {n}")
    counter = 0
    for i, dct in enumerate(data_loader):
        counter += 1
        if counter > opt.counter:
            break
        if i % opt.num_shards == opt.shard_index:
            images = dct["image"]
            filenames = dct["filename"]
            start = time.time()
            image_list = []
            for j in range(images.size(0)):
                image = tfunc.to_pil_image(images[j])
                image_list.append(image)
            generated_images = stable_generator.generate(
                image_list,
                n_sample_per_image=1,
            )
            _save_images(
                filenames, generated_images, imagenet_dir, opt.outdir, opt.image_version
            )
            end = time.time()
            print(
                f"Generated {len(generated_images)} images in time: {end-start} seconds."
            )

    print("Program finished!")


def _save_images(file_list, images, old_prefix, new_prefix, image_version):
    for filepath, img in zip(file_list, images):
        # Replace the prefix
        new_filepath = filepath.replace(old_prefix, new_prefix)

        # Extract directory and create if it doesn't exist
        directory = os.path.dirname(new_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Extract filename and extension
        filename, extension = os.path.splitext(os.path.basename(filepath))

        new_filename = f"{filename}_{image_version}{extension}"
        new_file_path = os.path.join(directory, new_filename)
        img.save(new_file_path)


if __name__ == "__main__":
    main()
