import argparse
import multiprocessing
import os
import shutil
from time import time
from typing import Optional, Tuple, Union

import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, save_json
from torch import GradScaler, autocast

from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context, empty_cache


def _select_checkpoint_for_resume(output_folder: str) -> Optional[str]:
    candidates = ("checkpoint_latest.pth", "checkpoint_final.pth", "checkpoint_best.pth")
    for name in candidates:
        ckpt = join(output_folder, name)
        if isfile(ckpt):
            return ckpt
    return None


def _extract_main_logits(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _to_probabilities(logits: torch.Tensor, use_sigmoid: bool) -> torch.Tensor:
    return torch.sigmoid(logits) if use_sigmoid else torch.softmax(logits, dim=1)


def _compute_containment_loss(
    output_a,
    output_b,
    class_idx_a: int,
    class_idx_b: int,
    use_sigmoid_a: bool,
    use_sigmoid_b: bool,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_a = _extract_main_logits(output_a)
    logits_b = _extract_main_logits(output_b)

    probs_a = _to_probabilities(logits_a, use_sigmoid_a)
    probs_b = _to_probabilities(logits_b, use_sigmoid_b)

    if class_idx_a >= probs_a.shape[1]:
        raise ValueError(
            f"class_idx_a={class_idx_a} is out of range for A output channels ({probs_a.shape[1]})."
        )
    if class_idx_b >= probs_b.shape[1]:
        raise ValueError(
            f"class_idx_b={class_idx_b} is out of range for B output channels ({probs_b.shape[1]})."
        )

    pa = probs_a[:, class_idx_a]
    pb = probs_b[:, class_idx_b]
    violation = torch.relu(pa - pb - margin)
    containment_loss = (violation ** 2).mean()
    violation_rate = (violation > 0).float().mean()
    return containment_loss, violation_rate


def _write_metadata_files(trainer) -> None:
    maybe_mkdir_p(trainer.output_folder_base)
    save_json(trainer.plans_manager.plans, join(trainer.output_folder_base, "plans.json"), sort_keys=False)
    save_json(trainer.dataset_json, join(trainer.output_folder_base, "dataset.json"), sort_keys=False)

    fp = join(trainer.preprocessed_dataset_folder_base, "dataset_fingerprint.json")
    if isfile(fp):
        shutil.copy(fp, join(trainer.output_folder_base, "dataset_fingerprint.json"))


def _cleanup_augmenters(*augmenters) -> None:
    for aug in augmenters:
        if isinstance(aug, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            aug._finish()


def _train_dual(
    trainer_a,
    trainer_b,
    warmup_epochs: int,
    total_epochs: int,
    lambda_containment: float,
    ramp_epochs: int,
    containment_margin: float,
    class_idx_a: int,
    class_idx_b: int,
    save_every: int,
    run_final_validation: bool,
    export_validation_probabilities: bool,
) -> None:
    trainer_a.num_epochs = total_epochs
    trainer_b.num_epochs = total_epochs

    if trainer_a.enable_deep_supervision != trainer_b.enable_deep_supervision:
        raise RuntimeError(
            "A and B must use the same deep supervision setting because joint training uses one shared dataloader."
        )
    if class_idx_a < 0 or class_idx_b < 0:
        raise ValueError("class_idx_a and class_idx_b must be >= 0.")

    if not trainer_a.was_initialized:
        trainer_a.initialize()
    if not trainer_b.was_initialized:
        trainer_b.initialize()

    # Set mirroring axes so checkpoints can be used directly for inference.
    trainer_a.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    trainer_b.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    trainer_a.set_deep_supervision_enabled(trainer_a.enable_deep_supervision)
    trainer_b.set_deep_supervision_enabled(trainer_b.enable_deep_supervision)
    if class_idx_a >= trainer_a.label_manager.num_segmentation_heads:
        raise ValueError(
            f"class_idx_a={class_idx_a} is out of range. "
            f"A has {trainer_a.label_manager.num_segmentation_heads} segmentation heads."
        )
    if class_idx_b >= trainer_b.label_manager.num_segmentation_heads:
        raise ValueError(
            f"class_idx_b={class_idx_b} is out of range. "
            f"B has {trainer_b.label_manager.num_segmentation_heads} segmentation heads."
        )

    _write_metadata_files(trainer_a)
    _write_metadata_files(trainer_b)

    # Use one shared dataloader so A and B always see the same augmented patch.
    dataloader_train, dataloader_val = trainer_a.get_dataloaders()

    if trainer_a.local_rank == 0:
        trainer_a.dataset_class.unpack_dataset(
            trainer_a.preprocessed_dataset_folder,
            overwrite_existing=False,
            num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
            verify=True,
        )

    start_epoch = max(int(trainer_a.current_epoch), int(trainer_b.current_epoch))
    trainer_a.current_epoch = start_epoch
    trainer_b.current_epoch = start_epoch

    if trainer_a.device.type == "cuda":
        scaler = GradScaler("cuda")
    else:
        scaler = None

    best_val_a = float("inf")
    best_val_b = float("inf")

    def containment_weight(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return lambda_containment
        progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        return lambda_containment * max(0.0, min(1.0, progress))

    trainer_a.print_to_log_file(
        f"[DualTrain] start_epoch={start_epoch}, warmup_epochs={warmup_epochs}, total_epochs={total_epochs}, "
        f"lambda={lambda_containment}, ramp_epochs={ramp_epochs}, class_idx_a={class_idx_a}, class_idx_b={class_idx_b}"
    )
    trainer_b.print_to_log_file(
        f"[DualTrain] start_epoch={start_epoch}, warmup_epochs={warmup_epochs}, total_epochs={total_epochs}, "
        f"lambda={lambda_containment}, ramp_epochs={ramp_epochs}, class_idx_a={class_idx_a}, class_idx_b={class_idx_b}"
    )

    try:
        for epoch in range(start_epoch, total_epochs):
            epoch_t0 = time()
            lam = containment_weight(epoch)
            phase = "independent" if epoch < warmup_epochs else "joint"

            trainer_a.network.train()
            trainer_b.network.train()
            trainer_a.lr_scheduler.step(epoch)
            trainer_b.lr_scheduler.step(epoch)

            tr_total = 0.0
            tr_sup_a = 0.0
            tr_sup_b = 0.0
            tr_cont = 0.0
            tr_violation = 0.0

            for _ in range(trainer_a.num_iterations_per_epoch):
                batch = next(dataloader_train)
                data = batch["data"].to(trainer_a.device, non_blocking=True)
                target = batch["target"]
                if isinstance(target, list):
                    target = [i.to(trainer_a.device, non_blocking=True) for i in target]
                else:
                    target = target.to(trainer_a.device, non_blocking=True)

                trainer_a.optimizer.zero_grad(set_to_none=True)
                trainer_b.optimizer.zero_grad(set_to_none=True)

                amp_ctx = (
                    autocast(trainer_a.device.type, enabled=True)
                    if trainer_a.device.type == "cuda"
                    else dummy_context()
                )
                with amp_ctx:
                    out_a = trainer_a.network(data)
                    out_b = trainer_b.network(data)
                    loss_sup_a = trainer_a.loss(out_a, target)
                    loss_sup_b = trainer_b.loss(out_b, target)
                    loss_cont, violation_rate = _compute_containment_loss(
                        out_a,
                        out_b,
                        class_idx_a,
                        class_idx_b,
                        trainer_a.label_manager.has_regions,
                        trainer_b.label_manager.has_regions,
                        containment_margin,
                    )
                    loss = loss_sup_a + loss_sup_b + lam * loss_cont

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(trainer_a.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainer_a.network.parameters(), 12)
                    scaler.unscale_(trainer_b.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainer_b.network.parameters(), 12)
                    scaler.step(trainer_a.optimizer)
                    scaler.step(trainer_b.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer_a.network.parameters(), 12)
                    torch.nn.utils.clip_grad_norm_(trainer_b.network.parameters(), 12)
                    trainer_a.optimizer.step()
                    trainer_b.optimizer.step()

                tr_total += float(loss.detach().cpu().item())
                tr_sup_a += float(loss_sup_a.detach().cpu().item())
                tr_sup_b += float(loss_sup_b.detach().cpu().item())
                tr_cont += float(loss_cont.detach().cpu().item())
                tr_violation += float(violation_rate.detach().cpu().item())

            n_tr = float(trainer_a.num_iterations_per_epoch)
            tr_total /= n_tr
            tr_sup_a /= n_tr
            tr_sup_b /= n_tr
            tr_cont /= n_tr
            tr_violation /= n_tr

            trainer_a.network.eval()
            trainer_b.network.eval()

            va_total = 0.0
            va_sup_a = 0.0
            va_sup_b = 0.0
            va_cont = 0.0
            va_violation = 0.0

            with torch.no_grad():
                for _ in range(trainer_a.num_val_iterations_per_epoch):
                    batch = next(dataloader_val)
                    data = batch["data"].to(trainer_a.device, non_blocking=True)
                    target = batch["target"]
                    if isinstance(target, list):
                        target = [i.to(trainer_a.device, non_blocking=True) for i in target]
                    else:
                        target = target.to(trainer_a.device, non_blocking=True)

                    amp_ctx = (
                        autocast(trainer_a.device.type, enabled=True)
                        if trainer_a.device.type == "cuda"
                        else dummy_context()
                    )
                    with amp_ctx:
                        out_a = trainer_a.network(data)
                        out_b = trainer_b.network(data)
                        loss_sup_a = trainer_a.loss(out_a, target)
                        loss_sup_b = trainer_b.loss(out_b, target)
                        loss_cont, violation_rate = _compute_containment_loss(
                            out_a,
                            out_b,
                            class_idx_a,
                            class_idx_b,
                            trainer_a.label_manager.has_regions,
                            trainer_b.label_manager.has_regions,
                            containment_margin,
                        )
                        loss = loss_sup_a + loss_sup_b + lam * loss_cont

                    va_total += float(loss.detach().cpu().item())
                    va_sup_a += float(loss_sup_a.detach().cpu().item())
                    va_sup_b += float(loss_sup_b.detach().cpu().item())
                    va_cont += float(loss_cont.detach().cpu().item())
                    va_violation += float(violation_rate.detach().cpu().item())

            n_va = float(trainer_a.num_val_iterations_per_epoch)
            va_total /= n_va
            va_sup_a /= n_va
            va_sup_b /= n_va
            va_cont /= n_va
            va_violation /= n_va

            trainer_a.current_epoch = epoch
            trainer_b.current_epoch = epoch

            msg = (
                f"[DualTrain][Epoch {epoch + 1}/{total_epochs}] phase={phase}, lambda={lam:.6f}, "
                f"train(total={tr_total:.4f}, A={tr_sup_a:.4f}, B={tr_sup_b:.4f}, contain={tr_cont:.4f}, "
                f"viol={tr_violation:.4f}), "
                f"val(total={va_total:.4f}, A={va_sup_a:.4f}, B={va_sup_b:.4f}, contain={va_cont:.4f}, "
                f"viol={va_violation:.4f}), "
                f"lrA={trainer_a.optimizer.param_groups[0]['lr']:.6f}, lrB={trainer_b.optimizer.param_groups[0]['lr']:.6f}, "
                f"time={time() - epoch_t0:.2f}s"
            )
            trainer_a.print_to_log_file(msg, also_print_to_console=True)
            trainer_b.print_to_log_file(msg, also_print_to_console=False)

            if not trainer_a.disable_checkpointing and (epoch + 1) % save_every == 0 and epoch != (total_epochs - 1):
                trainer_a.save_checkpoint(join(trainer_a.output_folder, "checkpoint_latest.pth"))
                trainer_b.save_checkpoint(join(trainer_b.output_folder, "checkpoint_latest.pth"))

            if not trainer_a.disable_checkpointing and va_sup_a < best_val_a:
                best_val_a = va_sup_a
                trainer_a.save_checkpoint(join(trainer_a.output_folder, "checkpoint_best.pth"))
            if not trainer_b.disable_checkpointing and va_sup_b < best_val_b:
                best_val_b = va_sup_b
                trainer_b.save_checkpoint(join(trainer_b.output_folder, "checkpoint_best.pth"))

        trainer_a.current_epoch = total_epochs - 1
        trainer_b.current_epoch = total_epochs - 1
        trainer_a.save_checkpoint(join(trainer_a.output_folder, "checkpoint_final.pth"))
        trainer_b.save_checkpoint(join(trainer_b.output_folder, "checkpoint_final.pth"))

        latest_a = join(trainer_a.output_folder, "checkpoint_latest.pth")
        latest_b = join(trainer_b.output_folder, "checkpoint_latest.pth")
        if isfile(latest_a):
            os.remove(latest_a)
        if isfile(latest_b):
            os.remove(latest_b)

        trainer_a.print_to_log_file("[DualTrain] Training finished. checkpoint_final.pth written for A and B.")
        trainer_b.print_to_log_file("[DualTrain] Training finished. checkpoint_final.pth written for A and B.")

    finally:
        _cleanup_augmenters(dataloader_train, dataloader_val)
        empty_cache(trainer_a.device)

    if run_final_validation:
        trainer_a.print_to_log_file("[DualTrain] Running final full validation for A...")
        trainer_a.perform_actual_validation(save_probabilities=export_validation_probabilities)
        trainer_b.print_to_log_file("[DualTrain] Running final full validation for B...")
        trainer_b.perform_actual_validation(save_probabilities=export_validation_probabilities)


def run_training_dual_containment(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_a_name: str = "nnUNetTrainerA",
    trainer_b_name: str = "nnUNetTrainerB",
    plans_identifier: str = "nnUNetPlans",
    warmup_epochs: int = 500,
    total_epochs: int = 1000,
    class_idx_a: int = 1,
    class_idx_b: int = 1,
    lambda_containment: float = 0.2,
    ramp_epochs: int = 50,
    containment_margin: float = 0.0,
    pretrained_weights_a: Optional[str] = None,
    pretrained_weights_b: Optional[str] = None,
    continue_joint: bool = False,
    disable_checkpointing: bool = False,
    run_final_validation: bool = False,
    export_validation_probabilities: bool = False,
    save_every: int = 50,
    device: torch.device = torch.device("cuda"),
):
    if isinstance(fold, str):
        if fold != "all":
            fold = int(fold)

    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0.")
    if warmup_epochs > total_epochs:
        raise ValueError("warmup_epochs cannot be larger than total_epochs.")
    if continue_joint and (pretrained_weights_a is not None or pretrained_weights_b is not None):
        raise RuntimeError("Cannot set continue_joint together with pretrained weights.")

    trainer_a = get_trainer_from_args(
        str(dataset_name_or_id), configuration, fold, trainer_a_name, plans_identifier, device=device
    )
    trainer_b = get_trainer_from_args(
        str(dataset_name_or_id), configuration, fold, trainer_b_name, plans_identifier, device=device
    )

    trainer_a.disable_checkpointing = disable_checkpointing
    trainer_b.disable_checkpointing = disable_checkpointing

    if not trainer_a.was_initialized:
        trainer_a.initialize()
    if not trainer_b.was_initialized:
        trainer_b.initialize()

    if continue_joint:
        ckpt_a = _select_checkpoint_for_resume(trainer_a.output_folder)
        ckpt_b = _select_checkpoint_for_resume(trainer_b.output_folder)
        if ckpt_a is None or ckpt_b is None:
            raise RuntimeError(
                f"continue_joint=True but resume checkpoint missing. Found: A={ckpt_a}, B={ckpt_b}"
            )
        trainer_a.load_checkpoint(ckpt_a)
        trainer_b.load_checkpoint(ckpt_b)
        if trainer_a.current_epoch != trainer_b.current_epoch:
            raise RuntimeError(
                f"Resume epoch mismatch: A={trainer_a.current_epoch}, B={trainer_b.current_epoch}. "
                "Please align checkpoints before continuing."
            )
    else:
        if pretrained_weights_a is not None:
            load_pretrained_weights(trainer_a.network, pretrained_weights_a, verbose=True)
        if pretrained_weights_b is not None:
            load_pretrained_weights(trainer_b.network, pretrained_weights_b, verbose=True)
        trainer_a.current_epoch = 0
        trainer_b.current_epoch = 0

    _train_dual(
        trainer_a=trainer_a,
        trainer_b=trainer_b,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        lambda_containment=lambda_containment,
        ramp_epochs=ramp_epochs,
        containment_margin=containment_margin,
        class_idx_a=class_idx_a,
        class_idx_b=class_idx_b,
        save_every=save_every,
        run_final_validation=run_final_validation,
        export_validation_probabilities=export_validation_probabilities,
    )


def run_training_dual_containment_entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID.")
    parser.add_argument("configuration", type=str, help="nnU-Net configuration, e.g. 3d_fullres.")
    parser.add_argument("fold", type=str, help='Fold [0..4] or "all".')
    parser.add_argument("-trA", type=str, default="nnUNetTrainerA", help="Trainer class for network A.")
    parser.add_argument("-trB", type=str, default="nnUNetTrainerB", help="Trainer class for network B.")
    parser.add_argument("-p", type=str, default="nnUNetPlans", help="Plans identifier.")
    parser.add_argument("--warmup_epochs", type=int, default=500, help="Epochs with independent training.")
    parser.add_argument("--total_epochs", type=int, default=1000, help="Total epochs.")
    parser.add_argument("--class_idx_A", type=int, default=1, help="Class/channel index i in A.")
    parser.add_argument("--class_idx_B", type=int, default=1, help="Class/channel index j in B.")
    parser.add_argument("--lambda_containment", type=float, default=0.2, help="Max weight of containment loss.")
    parser.add_argument(
        "--ramp_epochs",
        type=int,
        default=50,
        help="Linear warmup length for containment weight after warmup_epochs.",
    )
    parser.add_argument("--containment_margin", type=float, default=0.0, help="Margin for A-in-B containment.")
    parser.add_argument("-pretrained_weights_A", type=str, default=None, help="Optional pretrained checkpoint for A.")
    parser.add_argument("-pretrained_weights_B", type=str, default=None, help="Optional pretrained checkpoint for B.")
    parser.add_argument("--continue_joint", action="store_true", help="Resume joint training for both A and B.")
    parser.add_argument("--disable_checkpointing", action="store_true", help="Disable checkpoint writing.")
    parser.add_argument("--run_final_validation", action="store_true", help="Run full validation at training end.")
    parser.add_argument(
        "--npz",
        action="store_true",
        help="If final validation is enabled, save probability maps (.npz).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Write checkpoint_latest every N epochs.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        help="cpu, cuda or mps (do not use this to select GPU id).",
    )
    args = parser.parse_args()

    if args.device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"-device must be cpu, cuda or mps. Got: {args.device}")

    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    run_training_dual_containment(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_a_name=args.trA,
        trainer_b_name=args.trB,
        plans_identifier=args.p,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        class_idx_a=args.class_idx_A,
        class_idx_b=args.class_idx_B,
        lambda_containment=args.lambda_containment,
        ramp_epochs=args.ramp_epochs,
        containment_margin=args.containment_margin,
        pretrained_weights_a=args.pretrained_weights_A,
        pretrained_weights_b=args.pretrained_weights_B,
        continue_joint=args.continue_joint,
        disable_checkpointing=args.disable_checkpointing,
        run_final_validation=args.run_final_validation,
        export_validation_probabilities=args.npz,
        save_every=args.save_every,
        device=device,
    )


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    run_training_dual_containment_entry()
