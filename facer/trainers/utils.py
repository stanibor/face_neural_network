from typing import Dict, Union

import torch

from facer.models.backbone import resnet_by_name


def load_model_from_training_checkpoint(checkpoint_path: Dict, device: Union[torch.device, str]="cpu"):
    checkpoint = torch.load(checkpoint_path)
    hparams = checkpoint['hyper_parameters']
    model_type = hparams['model_type']

    model_params = {k: hparams[k] for k in ('pool_size', 'levels', 'hidden_channels')}
    backbone = resnet_by_name(hparams['backbone_'])

    model = model_type(output_shape=(68, 2), backbone=backbone, **model_params).to(device)
    best_state = checkpoint['state_dict']
    best_state = {k.replace('model.', ''): v for k, v in best_state.items()}
    # best_state = torch.load("best_model_state.pt")
    model.load_state_dict(best_state)

    return model
