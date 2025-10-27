import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import UNetANC
from dataset import AudioDataset
import os

DATASET_PATHS = {
    'clean_testset': r"C:\mp train\dataset\clean_testset_final",
    'noisy_dataset': r"C:\mp train\dataset\pnoisy_trainset_28spk_wav"
}

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered')
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += criterion(output, clean).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Check dataset paths
    for name, path in DATASET_PATHS.items():
        if not os.path.exists(path):
            print(f"Error: Directory not found - {name}: {path}")
            exit(1)

    try:
        print("\nInitializing dataset...")
        
        # Limit number of samples using max_samples for testing or debugging
        max_samples = 1000 # Set to None to use the entire dataset
        dataset = AudioDataset(
            DATASET_PATHS['clean_testset'],
            DATASET_PATHS['noisy_dataset'],
            max_samples=max_samples
        )
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")

        # Split dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

        # Initialize model, criterion, and optimizer
        model = UNetANC().to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, device=device)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
