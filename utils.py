import torch 
import diffusion 
import model 
import dataset 
import torchvision
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple
import sys
import json
from pathlib import Path


def get_train_val_test_loaders(
    batch_size: int = 128,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    dataset_path: str = './data',
    dataset_name: str = 'cifar10',
    image_size: int = 32,
):
    """
    Get the train, val, and test loaders for the CIFAR10 dataset

    Args:
        batch_size: the batch size for the loaders
        num_workers: the number of workers for the loaders
        prefetch_factor: the prefetch factor for the loaders
        dataset_path: the path to the dataset
    Returns:
        train_loader: the train loader
        val_loader: the val loader
        test_loader: the test loader
    """
    ds = dataset_name.lower().strip()
    if ds == 'cifar10':
        data_module = dataset.CIFAR10Dataset(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_path=dataset_path,
        )
    elif ds in {'flowers', 'oxford-flowers', 'flowers102', 'flower'}:
        flowers_root = str(Path(dataset_path) / 'flowers') if 'flowers' not in dataset_path.lower() else dataset_path
        data_module = dataset.FlowersDataset(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_path=flowers_root,
            image_size=image_size if image_size is not None else 64,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    train_loader = data_module.get_trainloader()
    val_loader = data_module.get_valloader()
    test_loader = data_module.get_testloader()
    return train_loader, val_loader, test_loader


def save_images(images: torch.Tensor, path: str = None, show: bool = True):
    """
    Unnormalize the images and convert them to a grid of images and save it to the path. Expects images to be of shape [B, C, H, W].

    Args:
        images: the images to save
        path: the path to save the images
        show: whether to show the images
    """
    images = (images.clamp(-1, 1) + 1) / 2                         # to [0,1]
    grid = torchvision.utils.make_grid(images)
    if path is not None:
        _ensure_dir(path)
        torchvision.utils.save_image(grid, path)

    grid = grid.detach().cpu().permute(1, 2, 0).numpy()
    if show:
        plt.imshow(grid); plt.axis('off'); plt.show(); plt.close()
    return grid
    

def _ensure_dir(path: str) -> None:
    p = Path(path)
    dir_path = p.parent if p.suffix else p
    dir_path.mkdir(parents=True, exist_ok=True)


def save_training_state(
    checkpoint_path: str,
    *,
    model_instance: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    conditioner_instance: Optional[model.Conditioner] = None,
    ema_instance: Optional[diffusion.EMA] = None,
    diffusion_instance: Optional[diffusion.Image_Diffusion] = None,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a training checkpoint. Any of model/optimizer/EMA/diffusion may be omitted.

    Returns the path where the checkpoint was written.
    """
    
    _ensure_dir(checkpoint_path)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "framework": {
            "torch_version": torch.__version__,
            "python_version": sys.version,
        },
    }

    if model_instance is not None:
        state["model_state"] = {
            "state_dict": f"{checkpoint_path}/model_state_dict.pt",
            "hparams": {
                "in_channels": int(getattr(model_instance, "in_channels")),
                "out_channels": int(getattr(model_instance, "out_channels")),
                "embedding_dim": int(getattr(model_instance, "embedding_dim")),
                "base_channels": int(getattr(model_instance, "base_channels")),
                "channel_mults": tuple(getattr(model_instance, "channel_mults")),
                "channels": getattr(model_instance, "channels"),
                "num_res_blocks_enc": getattr(model_instance, "num_res_blocks_enc"),
                "num_res_blocks_dec": getattr(model_instance, "num_res_blocks_dec"),
                "downsample_at": getattr(model_instance, "downsample_at"),
                "attn_stages": getattr(model_instance, "attn_stages"),
                "dec_attn_stages": getattr(model_instance, "dec_attn_stages"),
                "bottleneck_attn": getattr(model_instance, "bottleneck_attn"),
                "attn_num_heads": getattr(model_instance, "attn_num_heads"),
                "dec_attn_num_heads": getattr(model_instance, "dec_attn_num_heads"),
                "bottleneck_num_heads": getattr(model_instance, "bottleneck_num_heads"),
                "attn_block_index": getattr(model_instance, "attn_block_index"),
                "num_groups": int(getattr(model_instance, "num_groups")),
            },
        }
        torch.save(model_instance.state_dict(), f"{checkpoint_path}/model_state_dict.pt")

    if optimizer is not None:
        state["optimizer_state"] = {
            "state_dict": f"{checkpoint_path}/optimizer_state_dict.pt",
            "class": optimizer.__class__.__name__,
        }
        torch.save(optimizer.state_dict(), f"{checkpoint_path}/optimizer_state_dict.pt")

    if conditioner_instance is not None:
        state["conditioner_state"] = {
            "state_dict": f"{checkpoint_path}/conditioner_state_dict.pt",
            "hparams": {
                "d_time": int(getattr(conditioner_instance, "d_time")),
                "d_embedding": int(getattr(conditioner_instance, "d_embedding")),
                "num_classes": int(getattr(conditioner_instance, "num_classes")),
                "p_dropout": float(getattr(conditioner_instance, "p_dropout")),
            },
        }
        torch.save(conditioner_instance.state_dict(), f"{checkpoint_path}/conditioner_state_dict.pt")

    if ema_instance is not None:
        state["ema_state"] = {
            "state_dict": f"{checkpoint_path}/ema_state_dict.pt",
            "num_updates": getattr(ema_instance, "num_updates", 0),
            "decay": getattr(ema_instance, "decay", None),
            "warmup": getattr(ema_instance, "warmup", None),
        }
        torch.save(ema_instance.ema_model.state_dict(), f"{checkpoint_path}/ema_state_dict.pt")

    if diffusion_instance is not None:
        state["diffusion_state"] = {
            "hparams": {
                "num_timesteps": getattr(diffusion_instance, "T", None),
                "beta_start": float(diffusion_instance.betas[0].item()) if hasattr(diffusion_instance, "betas") else None,
                "beta_end": float(diffusion_instance.betas[-1].item()) if hasattr(diffusion_instance, "betas") else None,
            }
        }

    if extra:
        state["extra"] = extra

    json.dump(state, open(f"{checkpoint_path}/state.json", "w"), indent=4)
    return checkpoint_path



def load_from_checkpoint(
    checkpoint_path: str,
    *,
    device: str = "cpu",
) -> Tuple[Optional[torch.nn.Module], Optional[torch.optim.Optimizer], Optional[diffusion.EMA], Optional[model.Conditioner], Optional[diffusion.Image_Diffusion]]:
    """
    Reconstruct model, optimizer, EMA, conditioner, and diffusion directly from a checkpoint saved by save_training_state.

    Returns (model | None, optimizer | None, ema | None, conditioner | None, diffusion | None), all moved to the specified device when constructed.
    """
    try:
        ckpt: Dict[str, Any] = json.load(open(f"{checkpoint_path}/state.json", "r"))
    except Exception:
        print(f"Training state not found: {checkpoint_path}/state.json")
        return None, None, None, None, None

    model_instance: Optional[torch.nn.Module] = None
    optimizer_instance: Optional[torch.optim.Optimizer] = None
    conditioner_instance: Optional[model.Conditioner] = None
    ema_instance: Optional[diffusion.EMA] = None
    diffusion_instance: Optional[diffusion.Image_Diffusion] = None

    # Diffusion
    diff_state = ckpt.get("diffusion_state", {})
    diff_h = diff_state.get("hparams", {}) if isinstance(diff_state, dict) else {}
    if isinstance(diff_h, dict) and diff_h:
        diffusion_instance = diffusion.Image_Diffusion(
            num_timesteps=int(diff_h["num_timesteps"]),
            beta_start=float(diff_h["beta_start"]),
            beta_end=float(diff_h["beta_end"]),
            device=device,
        )

    # Model (needs model_state with hparams and state_dict)
    model_state = ckpt.get("model_state", {})
    if isinstance(model_state, dict) and model_state.get("state_dict") is not None and isinstance(model_state.get("hparams"), dict):
        mh = model_state["hparams"]
        model_instance = model.UNet(device=device, **mh)
        model_instance.load_state_dict(torch.load(model_state["state_dict"], map_location="cpu"))  # type: ignore[arg-type]

    # Optimizer (needs optimizer_state with class and state_dict and model_instance)
    optim_state = ckpt.get("optimizer_state", {})
    if model_instance is not None and isinstance(optim_state, dict) and optim_state.get("state_dict") is not None and optim_state.get("class") is not None:
        name_lc = str(optim_state["class"]).lower()
        optimizer_cls_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }
        OptimCls = optimizer_cls_map.get(name_lc)
        if OptimCls is not None:
            optimizer_instance = OptimCls(model_instance.parameters(), lr=0.0)  # lr overridden by state dict
            optimizer_instance.load_state_dict(torch.load(optim_state["state_dict"], map_location="cpu"))  # type: ignore[arg-type]

    # Conditioner (needs conditioner_state with hparams and state_dict)

    conditioner_state = ckpt.get("conditioner_state", {})
    if isinstance(conditioner_state, dict) and conditioner_state.get("state_dict") is not None and isinstance(conditioner_state.get("hparams"), dict):
        ch = conditioner_state["hparams"]
        conditioner_instance = model.Conditioner(device=device, **ch)
        conditioner_instance.load_state_dict(torch.load(conditioner_state["state_dict"], map_location="cpu"))  # type: ignore[arg-type]

    # EMA (needs model_instance and ema_state)
    ema_state = ckpt.get("ema_state", {})
    if model_instance is not None and isinstance(ema_state, dict) and ema_state.get("state_dict") is not None:
        ema_decay = ema_state.get("decay", 0.999)
        ema_warmup = ema_state.get("warmup", 0)
        ema_instance = diffusion.EMA(model_instance, decay=float(ema_decay), device=device, warmup=int(ema_warmup))
        ema_instance.ema_model.load_state_dict(torch.load(ema_state["state_dict"], map_location="cpu"))  # type: ignore[arg-type]
        ema_instance.num_updates = int(ema_state.get("num_updates", 0))

    return model_instance, optimizer_instance, ema_instance, conditioner_instance, diffusion_instance

