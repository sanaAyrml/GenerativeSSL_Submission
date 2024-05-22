import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    config.num_samples = 16
    config.truncation = 0.7
    config.stochastic_truncation = False
    config.noise_size = 128
    config.batch_size = 4
    config.experiment_name = "icgan_biggan_imagenet_res256"
    config.feat_ext_path = "swav_pretrained.pth.tar"
    config.size = 256
    config.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    config.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return config
