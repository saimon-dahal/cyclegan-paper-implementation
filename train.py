import os

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.configs import settings
from src.data import denormalize, get_dataloaders
from src.models import Discriminator, Generator
from src.utils import (
    GANLoss,
    ImageBuffer,
    LRScheduler,
    cycle_consistency_loss,
    identity_loss,
)


def save_checkpoint(epoch, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D, path):
    torch.save(
        {
            "epoch": epoch,
            "G_A2B": G_A2B.state_dict(),
            "G_B2A": G_B2A.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        },
        path,
    )


def load_checkpoint(path, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D):
    checkpoint = torch.load(path)
    G_A2B.load_state_dict(checkpoint["G_A2B"])
    G_B2A.load_state_dict(checkpoint["G_B2A"])
    D_A.load_state_dict(checkpoint["D_A"])
    D_B.load_state_dict(checkpoint["D_B"])
    opt_G.load_state_dict(checkpoint["opt_G"])
    opt_D.load_state_dict(checkpoint["opt_D"])
    return checkpoint["epoch"]


def weights_init(m):
    """Normal Weight initialization"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def train():
    # Setup - Mac GPU support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Data loaders
    train_loader, _ = get_dataloaders(
        root_dir=settings.data_root,
        batch_size=settings.batch_size,
        img_size=settings.img_size,
        num_workers=settings.num_workers,
    )

    # Models
    G_A2B = Generator(
        settings.input_channels, settings.output_channels, settings.n_residual_blocks
    ).to(device)

    G_B2A = Generator(
        settings.input_channels, settings.output_channels, settings.n_residual_blocks
    ).to(device)

    D_A = Discriminator(settings.input_channels).to(device)
    D_B = Discriminator(settings.input_channels).to(device)

    # Initialize weights
    G_A2B.apply(weights_init)
    G_B2A.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

    # Optimizers
    # Seperate optimizers cause both are adversaries
    # Adam maintains moving averages
    opt_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=settings.lr,
        betas=(settings.beta1, 0.999),
    )

    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=settings.lr,
        betas=(settings.beta1, 0.999),
    )

    # Learning rate schedulers
    lr_scheduler_G = LRScheduler(opt_G, settings.n_epochs, settings.n_epochs // 2)
    lr_scheduler_D = LRScheduler(opt_D, settings.n_epochs, settings.n_epochs // 2)

    # Loss functions
    criterion_GAN = GANLoss().to(device)

    # Image buffers
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    # TensorBoard
    writer = SummaryWriter("runs/cyclegan")

    # Training loop
    global_step = 0

    for epoch in range(settings.n_epochs):
        epoch_G_loss = 0
        epoch_D_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{settings.n_epochs}")

        for _, batch in enumerate(pbar):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # =================== Train Generators ===================
            opt_G.zero_grad()

            # Identity loss
            identity_A = G_B2A(real_A)
            loss_identity_A = (
                identity_loss(real_A, identity_A) * settings.lambda_identity
            )

            identity_B = G_A2B(real_B)
            loss_identity_B = (
                identity_loss(real_B, identity_B) * settings.lambda_identity
            )

            # GAN loss
            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, True)

            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, True)

            # Cycle consistency loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = (
                cycle_consistency_loss(real_A, recovered_A) * settings.lambda_cycle
            )

            recovered_B = G_A2B(fake_A)
            loss_cycle_B = (
                cycle_consistency_loss(real_B, recovered_B) * settings.lambda_cycle
            )

            # Total generator loss
            loss_G = (
                loss_identity_A
                + loss_identity_B
                + loss_GAN_A2B
                + loss_GAN_B2A
                + loss_cycle_A
                + loss_cycle_B
            )

            loss_G.backward()
            opt_G.step()

            # =================== Train Discriminators ===================
            opt_D.zero_grad()

            # Discriminator A
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, True)

            fake_A_buffer_out = fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = D_A(fake_A_buffer_out.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, False)

            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

            # Discriminator B
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, True)

            fake_B_buffer_out = fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = D_B(fake_B_buffer_out.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, False)

            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            # Total discriminator loss
            loss_D = loss_D_A + loss_D_B

            loss_D.backward()
            opt_D.step()

            # Update progress bar
            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()

            pbar.set_postfix(
                {"G_loss": f"{loss_G.item():.4f}", "D_loss": f"{loss_D.item():.4f}"}
            )

            # Log to tensorboard
            if global_step % 100 == 0:
                writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
                writer.add_scalar("Loss/Discriminator", loss_D.item(), global_step)
                writer.add_scalar(
                    "Loss/G_identity",
                    (loss_identity_A + loss_identity_B).item(),
                    global_step,
                )
                writer.add_scalar(
                    "Loss/G_GAN", (loss_GAN_A2B + loss_GAN_B2A).item(), global_step
                )
                writer.add_scalar(
                    "Loss/G_cycle", (loss_cycle_A + loss_cycle_B).item(), global_step
                )

            global_step += 1

        # Update learning rates
        lr_G = lr_scheduler_G.step(epoch)
        lr_D = lr_scheduler_D.step(epoch)

        print(
            f"Epoch {epoch + 1} - G_loss: {epoch_G_loss / len(train_loader):.4f}, "
            f"D_loss: {epoch_D_loss / len(train_loader):.4f}, LR: {lr_G:.6f}"
        )

        # Save samples
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake_B = G_A2B(real_A[:4])
                fake_A = G_B2A(real_B[:4])

                real_A_img = denormalize(real_A[:4])
                real_B_img = denormalize(real_B[:4])
                fake_A_img = denormalize(fake_A)
                fake_B_img = denormalize(fake_B)

                comparison = torch.cat(
                    [real_A_img, fake_B_img, real_B_img, fake_A_img], dim=0
                )

                vutils.save_image(
                    comparison, f"samples/epoch_{epoch + 1}.png", nrow=4, padding=2
                )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                epoch + 1,
                G_A2B,
                G_B2A,
                D_A,
                D_B,
                opt_G,
                opt_D,
                f"checkpoints/checkpoint_epoch_{epoch + 1}.pth",
            )

    # Save final model
    save_checkpoint(
        settings.n_epochs,
        G_A2B,
        G_B2A,
        D_A,
        D_B,
        opt_G,
        opt_D,
        "checkpoints/final_model.pth",
    )

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
