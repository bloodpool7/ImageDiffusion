import torch
import model
import diffusion 
import utils
from pathlib import Path
import json


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    # Choose dataset and image size
    dataset_name = 'flowers'  # 'cifar10' or 'flowers'
    train_loader, _, _ = utils.get_train_val_test_loaders(
        batch_size=32, 
        num_workers=4 if device == 'cuda' else 8, 
        dataset_name=dataset_name,
        image_size=64
    )
    unet = model.UNet(
        in_channels=3, 
        out_channels=3, 
        base_channels=64, 
        embedding_dim=512, 
        channel_mults=(1, 2, 4, 8), 
        device=device
    )
                      
    conditioner = model.Conditioner(d_time=128, d_embedding=512, num_classes=5, p_dropout=0.1, device=device) # 5 classes: daisy, dandelion, rose, sunflower, tulip
    diffusion_model = diffusion.Image_Diffusion(device=device)
    ema = diffusion.EMA(unet, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), 2e-4)

    # Training schedule and checkpointing settings
    num_epochs = 2400
    sample_every = 300
    save_every = 400
    ckpt_dir = Path('./checkpoints/flowers-conditional-64x64')

    # Optional resume: set a path or leave as None
    resume_path = ckpt_dir / 'last'  # e.g., os.path.join(ckpt_dir, 'last.pt')
    resume_path = None
    if resume_path:
        loaded_model, loaded_opt, loaded_ema, loaded_cond, loaded_diff = utils.load_from_checkpoint(
            resume_path,
            device=device,
        )
        if loaded_model is not None:
            unet = loaded_model
        if loaded_opt is not None:
            optimizer = loaded_opt
        if loaded_ema is not None:
            ema = loaded_ema
        if loaded_diff is not None:
            diffusion_model = loaded_diff
        if loaded_cond is not None:
            conditioner = loaded_cond
        try:
            ckpt = json.load(open(resume_path / "state.json", "r"))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
        except Exception:
            print("Error getting checkpoint json")
            start_epoch = 0
    else:
        start_epoch = 0

    print(f"Starting training from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        loss = diffusion_model.train_one_epoch(unet, train_loader, optimizer, ema, conditioner, device=device, epoch=epoch)

        val_loss = None
        if epoch % sample_every == 0:
            # val_loss = diffusion_model.evaluate(ema.ema_model, test_loader, device=device)
            out = diffusion_model.sample_ddim(ema.ema_model, conditioner, num_samples=16, device=device, image_shape=(3, 64, 64))
            utils.save_images(out, ckpt_dir / "samples" / f"samples_epoch_{epoch}.png", show=False)

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            utils.save_training_state(
                ckpt_dir / f"epoch_{epoch}",
                model_instance=unet,
                optimizer=optimizer,
                ema_instance=ema,
                diffusion_instance=diffusion_model,
                conditioner_instance=conditioner,
                epoch=epoch,
                extra={"train_loss": float(loss), "val_loss": float(val_loss) if val_loss is not None else None},
            )
        # Also save/update a rolling "last" checkpoint for convenience
        utils.save_training_state(
            ckpt_dir / "last",
            model_instance=unet,
            optimizer=optimizer,
            ema_instance=ema,
            diffusion_instance=diffusion_model,
            conditioner_instance=conditioner,
            epoch=epoch,
        )
    
    print("Training completed")


if __name__ == "__main__":
    main()