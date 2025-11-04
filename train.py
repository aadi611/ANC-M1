import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from model import UNetANC
from dataset import AudioDataset

# Configuration
DATASET_PATHS = {
    'clean_testset': r"C:\mp train\dataset\clean_testset_final",
    'noisy_dataset': r"C:\mp train\dataset\pnoisy_trainset_28spk_wav"
}

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria is met.
        
        Returns:
            bool: True if early stopping should trigger
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered!')
                return True
        else:
            if self.verbose and val_loss < self.best_loss:
                print(f'Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}')
            self.best_loss = val_loss
            self.counter = 0
        
        return False

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, train_loss: float, val_loss: float, 
                   filepath: str = 'best_model.pth') -> None:
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, filepath)
    print(f'✓ Checkpoint saved: {filepath}')

def plot_losses(train_losses: List[float], val_losses: List[float], 
                save_path: str = 'training_progress.png') -> None:
    """Plot and save training progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device, 
                epoch: int, log_interval: int = 100) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.6f}')

    return total_loss / len(loader)

def validate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module,
             device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            total_loss += criterion(output, clean).item()

    return total_loss / len(loader)

def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                model: torch.nn.Module, criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer, num_epochs: int, 
                device: torch.device, checkpoint_path: str = 'best_model.pth') -> Tuple[List[float], List[float]]:
    """
    Main training loop with validation and early stopping.
    
    Returns:
        Tuple of (train_losses, val_losses)
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*60}')
        
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss:   {val_loss:.6f}')
        print(f'  LR:         {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)

        # Plot progress
        plot_losses(train_losses, val_losses)

        # Early stopping check
        if early_stopping(val_loss):
            break

    return train_losses, val_losses

def validate_paths(paths: Dict[str, str]) -> bool:
    """Validate that all dataset paths exist."""
    all_valid = True
    for name, path in paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"✗ Error: Directory not found - {name}: {path}")
            all_valid = False
        else:
            print(f"✓ Found {name}: {path}")
    return all_valid

def main():
    """Main training pipeline."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Validate paths
    print("\nValidating dataset paths...")
    if not validate_paths(DATASET_PATHS):
        return

    try:
        # Load dataset
        print("\nInitializing dataset...")
        max_samples = 1000  # Set to None for full dataset
        dataset = AudioDataset(
            DATASET_PATHS['clean_testset'],
            DATASET_PATHS['noisy_dataset'],
            max_samples=max_samples
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducibility
        )
        print(f"Train samples: {train_size}, Val samples: {val_size}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )

        # Initialize model
        print("\nInitializing model...")
        model = UNetANC().to(device)
        print(f"✓ Model parameters: {model.count_parameters():,}")
        
        # Setup training components
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Train model
        print("\nStarting training...")
        train_losses, val_losses = train_model(
            train_loader, val_loader, model, criterion, 
            optimizer, num_epochs=50, device=device
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Best validation loss: {min(val_losses):.6f}")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

if __name__ == "__main__":
    main()
