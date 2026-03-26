import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerA(nnUNetTrainer):
    """
    Trainer class for network A.
    Kept separate so nnU-Net writes A and B checkpoints into different result folders.
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerB(nnUNetTrainer):
    """
    Trainer class for network B.
    Kept separate so nnU-Net writes A and B checkpoints into different result folders.
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
