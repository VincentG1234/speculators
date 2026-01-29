import argparse
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader
from transformers import LlamaConfig
from transformers.utils import cached_file
from transformers.models.auto.configuration_auto import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.data import (
    Eagle3SampleFileDataset,
    create_collate_fn,
    split_files,
    standardize_data_v0,
    standardize_data_v1,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed

# DRAFTER MODEL HYPARAMETERS
NORM_BEFORE_RESIDUAL = True

# Dataloader
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
NOISE_STD = 0.05


def setup_dataloader(
    file_list: list[str],
    world_size: int,
    local_rank: int,
    add_noise: bool = True,
    data_format_version: int = 1,
) -> DataLoader:
    """Setup dataloader for training.
    Args:
        file_list: List of file paths to load data from.
        world_size: Number of processes in the distributed training.
        local_rank: Rank of the current process.
        add_noise: Whether to add noise to the data.
        data_format_version: Version of the data format. Default is 1.
    Returns:
        DataLoader: Dataloader for training.
    """
    if add_noise:
        noise_transform = AddUniformNoise(
            std=NOISE_STD, tensors=("hidden_states", "verifier_last_hidden_states")
        )
    else:
        noise_transform = None

    standardize_fn = (
        standardize_data_v1 if data_format_version == 1 else standardize_data_v0
    )

    dataset = Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=args.total_seq_len,
        transform=noise_transform,
        standardize_fn=standardize_fn,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=args.total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
        collate_fn=create_collate_fn(args.total_seq_len),
        persistent_workers=True,
    )


def create_transformer_layer_config(
    verifier_name_or_path: str, num_layers: int
) -> LlamaConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    transformer_layer_config = LlamaConfig(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=verifier_config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=verifier_config.num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        hidden_act=verifier_config.hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=getattr(verifier_config, "head_dim", None),
    )
    transformer_layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    return transformer_layer_config


def main(args: argparse.Namespace):
    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
    device = torch.device(local_rank)

    if args.pretrained_draft_model_path:
        # Fine-tuning mode
        if args.d2t_path or args.t2d_path:
            raise ValueError(
                "Cannot specify d2t/t2d paths when using pretrained model. "
                "Mappings will be loaded from the model checkpoint."
            )

        speculator_config = Eagle3SpeculatorConfig.from_pretrained(
            args.pretrained_draft_model_path
        )

        # Update verifier in config if provided in args
        if args.verifier_name_or_path:
             speculator_config.speculators_config.verifier.name_or_path = args.verifier_name_or_path

        # Manually load t2d/d2t from checkpoint to initialize model structure correctly
        t2d, d2t = None, None
        state_dict = None
        
        # Try to find model file (local or hub)
        model_path = args.pretrained_draft_model_path
        st_path = Path(model_path) / "model.safetensors"
        bin_path = Path(model_path) / "pytorch_model.bin"
        
        file_to_load = None
        if st_path.exists():
            file_to_load = st_path
        elif bin_path.exists():
            file_to_load = bin_path
        else:
            # Try getting from hub cache
            try:
                file_to_load = cached_file(model_path, "model.safetensors")
                if file_to_load is None:
                    file_to_load = cached_file(model_path, "pytorch_model.bin")
            except Exception:
                pass

        if file_to_load:
            file_to_load = str(file_to_load)
            if file_to_load.endswith(".safetensors"):
                state_dict = load_safetensors_file(file_to_load)
            else:
                state_dict = torch.load(file_to_load, map_location="cpu")
            
            t2d = state_dict.get("t2d")
            d2t = state_dict.get("d2t")
            
        if t2d is not None:
            t2d = t2d.to(device)
        if d2t is not None:
            d2t = d2t.to(device)

        # Check for vocab size mismatch only if t2d is NOT found
        if t2d is None:
            verifier_config = AutoConfig.from_pretrained(speculator_config.speculators_config.verifier.name_or_path)
            if hasattr(verifier_config, "text_config"):
                verifier_config = verifier_config.text_config
            
            if verifier_config.vocab_size != speculator_config.draft_vocab_size:
                print(f"WARNING: Verifier vocab size ({verifier_config.vocab_size}) does not match "
                      f"draft vocab size ({speculator_config.draft_vocab_size}) and no mapping found. "
                      "Updating draft vocab size to match verifier.")
                speculator_config.draft_vocab_size = verifier_config.vocab_size
        else:
             print("INFO: Found t2d/d2t mapping in checkpoint. Using mapped vocabulary.")
            
        # Create model with mappings (like from scratch training)
        draft_model = Eagle3DraftModel(config=speculator_config, t2d=t2d, d2t=d2t)
        
        # Load pretrained weights manually
        if file_to_load and state_dict:
            # Filter out keys that should not be loaded
            keys_to_ignore = ["verifier"]
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not any(ignore_key in k for ignore_key in keys_to_ignore)
            }
            
            # Load weights with strict=False to ignore missing keys
            missing_keys, unexpected_keys = draft_model.load_state_dict(
                filtered_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"INFO: Missing keys when loading checkpoint (expected): {missing_keys}")
            if unexpected_keys:
                print(f"WARNING: Unexpected keys when loading checkpoint: {unexpected_keys}")
                
            print(f"INFO: Successfully loaded pretrained weights from {file_to_load}")
        else:
            print("WARNING: Could not find checkpoint file. Model will be initialized from scratch.")
        
        if speculator_config.eagle_aux_hidden_state_layer_ids:
             print(f"INFO: Pretrained model uses layer IDs: {speculator_config.eagle_aux_hidden_state_layer_ids}")
             print("Ensure your training data is generated with these same layer IDs!")

    else:
        # Scratch training mode
        # Load t2d and d2t tensors if provided
        if args.d2t_path or args.t2d_path:
            if not (args.d2t_path and args.t2d_path):
                raise ValueError(
                    "Both t2d and d2t must be provided together, or both must be omitted. "
                    f"Got t2d={'provided' if args.t2d_path is not None else 'not provided'}"
                    f"d2t={'provided' if args.d2t_path is not None else 'not provided'}"
                )
            d2t = torch.from_numpy(np.load(args.d2t_path)).to(device)
            t2d = torch.from_numpy(np.load(args.t2d_path)).to(device)
            draft_vocab_size = d2t.shape[0]
        else:
            d2t = None
            t2d = None
            # When vocab mapping is not provided, use the full verifier vocab
            verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
            if hasattr(verifier_config, "text_config"):
                verifier_config = verifier_config.text_config
            draft_vocab_size = verifier_config.vocab_size

        # Setup speculator config
        transformer_layer_config = create_transformer_layer_config(
            args.verifier_name_or_path, args.num_layers
        )

        speculator_config = Eagle3SpeculatorConfig(
            transformer_layer_config=transformer_layer_config,
            draft_vocab_size=draft_vocab_size,
            norm_before_residual=NORM_BEFORE_RESIDUAL,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        proposal_type="greedy",
                        speculative_tokens=args.ttt_steps,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=args.verifier_name_or_path,
                    architectures=["LlamaForCausalLM"],
                ),
            ),
        )

        # Setup draft model
        draft_model = Eagle3DraftModel(config=speculator_config, t2d=t2d, d2t=d2t)

    # Setup dataloaders
    train_files, val_files = split_files(args.data_path, ratio=0.9)
    train_loader = setup_dataloader(
        train_files,
        world_size,
        local_rank,
        add_noise=True,
        data_format_version=args.data_format_version,
    )
    val_loader = setup_dataloader(
        val_files,
        world_size,
        local_rank,
        add_noise=False,
        data_format_version=args.data_format_version,
    )

    # Setup trainer
    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={
            "use_off_policy_tokens": args.use_off_policy_tokens,
            "ttt_steps": args.ttt_steps,
            "ttt_step_loss_decay": args.ttt_step_loss_decay,
        },
        val_call_kwargs={
            "use_off_policy_tokens": False,
            "ttt_steps": args.ttt_steps,
            "ttt_step_loss_decay": args.ttt_step_loss_decay,
        },
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_num_cosine_cycles=args.scheduler_num_cosine_cycles,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    maybe_destroy_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="",
        help="One of 'trackio', 'wandb', 'tensorboard' or comma separated list of them",
    )
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--data-format-version", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--pretrained-draft-model-path",
        type=str,
        default=None,
        help="Path to pretrained draft model to finetune",
    )
    parser.add_argument(
        "--use-off-policy-tokens",
        action="store_true",
        default=False,
        help="Use off-policy tokens during training (required for regenerated data)",
    )
    # lr scheduler
    parser.add_argument("--scheduler-type", type=str, default="linear")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--scheduler-num-cosine-cycles", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --nnodes=1 --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
