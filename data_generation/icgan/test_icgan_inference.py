import torch
import os
import numpy as np
from config import get_config
from icgan_inference import ICGANInference
from pytorch_pretrained_biggan import convert_to_images


def save(out, torch_format=True):
    if torch_format:
        with torch.no_grad():
            out = out.cpu().numpy()
    img = convert_to_images(out)[0]
    return img


def save_images(all_outs, num_samples, size):
    row_i, col_i, i_im = 0, 0, 0
    all_images_mosaic = np.zeros(
        (3, size * (int(np.sqrt(num_samples))), size * (int(np.sqrt(num_samples))))
    )
    for j in range(len(all_outs)):
        all_images_mosaic[
            :, row_i * size : row_i * size + size, col_i * size : col_i * size + size
        ] = all_outs[j]
        if row_i == int(np.sqrt(num_samples)) - 1:
            row_i = 0
            if col_i == int(np.sqrt(num_samples)) - 1:
                col_i = 0
            else:
                col_i += 1
        else:
            row_i += 1
        i_im += 1

    name = "output_image.png"
    pil_image = save(all_images_mosaic[np.newaxis, ...], torch_format=False)
    pil_image.save(os.path.join("output", name))


def main():
    # Initialize the RCDMInference class
    config = get_config()
    icgan_inference = ICGANInference(config)

    input_image_tensor = icgan_inference.preprocess_input_image(
        input_image_path="../rcdm/images/Jules.jpeg", size=256
    )

    # Run inference
    generated_images = icgan_inference.run_inference(
        input_image_tensor=input_image_tensor
    )

    # Save the generated images to disk
    save_images(generated_images, num_samples=16, size=256)


if __name__ == "__main__":
    main()
