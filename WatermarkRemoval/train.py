import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from unet_model import UNet  # Assuming UNet is in unet_model.py
from composite_loss import composite_loss  # Your composite loss function
from torch.utils.data import Dataset
from PIL import Image
import os
import optuna
from optuna.pruners import MedianPruner
from torch.utils.tensorboard import SummaryWriter

def objective(trial):
    # Hyperparameters to be optimized
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    beta = trial.suggest_float("beta", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)

    # Model, optimizer, and other setup here
    model = UNet(n_channels=3, n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (watermark, watermark_free) in enumerate(train_loader):
            watermark = watermark.to(device)
            watermark_free = watermark_free.to(device)

            optimizer.zero_grad()
            outputs = model(watermark)
            loss, _, _, _ = composite_loss(outputs, watermark_free, alpha, beta, gamma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i+1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        epoch_loss /= len(train_loader)
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return the metric of interest (e.g., validation loss)
    # For simplicity, I'm using training loss here
    return epoch_loss

class WatermarkRemovalDataset(Dataset):
    def __init__(self, watermarked_dir, watermark_free_dir, transform=None):
        self.watermarked_dir = watermarked_dir
        self.watermark_free_dir = watermark_free_dir
        self.transform = transform

        # Assuming each file in watermarked_dir match each file in watermark_free_dir
        self.filenames = os.listdir(watermarked_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        watermarked_file = os.path.join(self.watermarked_dir, self.filenames[idx])
        watermark_free_file = os.path.join(self.watermark_free_dir, os.path.splitext(self.filenames[idx])[0] + '.jpg')

        watermarked_img = Image.open(watermarked_file)
        watermark_free_img = Image.open(watermark_free_file)

        if self.transform:
            watermarked_img = self.transform(watermarked_img)
            watermark_free_img = self.transform(watermark_free_img)

        return watermarked_img, watermark_free_img


# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 20
batch_size = 16
learning_rate = 1.254037045488747e-05
alpha = 0.25730441769093637
beta = 0.2604620107676715
gamma = 0.12656891482151053

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor()
    ])

train_dataset = WatermarkRemovalDataset(
    watermarked_dir='../Data/CLWD/train/Watermarked_image',
    watermark_free_dir='../Data/CLWD/train/Watermark_free_image',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = UNet(n_channels=3, n_classes=3).to(device)  # Adjust n_channels and n_classes based on your data

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# study = optuna.create_study(direction="minimize", pruner=MedianPruner())
# study.optimize(objective, n_trials=10)

# # Best hyperparameters
# print("Best trial:")
# trial = study.best_trial
# print(f" - Learning Rate: {trial.params['learning_rate']}")
# print(f" - Alpha: {trial.params['alpha']}")
# print(f" - Beta: {trial.params['beta']}")

writer = SummaryWriter('runs/watermark_removal_experiment')

# Training Loop
for epoch in range(num_epochs):
    for i, (watermark, watermark_free) in enumerate(train_loader):
        watermark = watermark.to(device)
        watermark_free = watermark_free.to(device)

        # Forward pass
        outputs = model(watermark)
        # print(outputs)
        loss, loss_psnr, loss_ssim, loss_lpips = composite_loss(outputs, watermark_free, alpha, beta, gamma)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('PSNR/train', loss_psnr, epoch * len(train_loader) + i)

        if (i+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            print(f'PSNR: {loss_psnr:.4f}, SSIM: {loss_ssim:.4f}, LPIPS: {loss_lpips:.4f}')
            print('-----------------------------------------')

    # Save the model at the end of each epoch
    model_save_path = f"2_model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

writer.close()
print('Training complete!')
