import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Mamba, ModelArgs
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

class PositionalMambaModel(torch.nn.Module):
    def __init__(self, mamba_args):
        super().__init__()
        self.mamba = Mamba(mamba_args)
        
        # Define valid token ranges for each position
        self.position_ranges = {
            0: (1, 9),      # Position 1: 1-9
            1: (100, 199),  # Position 2: 100-199
            2: (200, 299)   # Position 3: 200-299
        }
        
        # Create position-specific masks
        self.position_masks = {}
        for pos, (min_val, max_val) in self.position_ranges.items():
            mask = torch.zeros(mamba_args.vocab_size)
            mask[min_val:max_val + 1] = 1
            self.position_masks[pos] = mask.bool()
        
        self.register_buffer('position_masks_tensor', torch.stack(list(self.position_masks.values())))
        
        # Feature processing layers
        self.feature_projection = torch.nn.Linear(2, mamba_args.d_model)
        
    def forward(self, x, features):
        # x shape: [batch_size, seq_length]
        # features shape: [batch_size, seq_length//3, 2]
        batch_size, seq_length = x.shape
        
        # Get input embeddings
        x_emb = self.mamba.embedding(x)  # [batch_size, seq_length, d_model]
        
        # Project features to match model dimension
        features = features.view(batch_size, seq_length//3, 2)
        projected_features = self.feature_projection(features)  # [batch_size, seq_length//3, d_model]
        
        # Repeat features for each token in the message
        expanded_features = projected_features.repeat_interleave(3, dim=1)  # [batch_size, seq_length, d_model]
        
        # Add features to embeddings
        x_emb = x_emb + expanded_features
        
        # Process through Mamba layers
        for layer in self.mamba.layers:
            x_emb = layer(x_emb)
            
        x_emb = self.mamba.norm_f(x_emb)
        logits = self.mamba.lm_head(x_emb)
        
        # Apply position-specific masks
        positions = torch.arange(seq_length, device=x.device) % 3
        position_masks = self.position_masks_tensor[positions]  # [seq_length, vocab_size]
        
        # Set logits of invalid tokens to large negative value
        masked_logits = logits.masked_fill(~position_masks.unsqueeze(0), -1e9)
        
        return masked_logits
    
    def generate_next_token(self, x, features, position_in_message):
        """Generate next token with position-specific constraints.
        
        Args:
            x: Input sequence [batch_size, seq_length]
            features: Context features [batch_size, seq_length//3, 2]
            position_in_message: Position in current message (0, 1, or 2)
        """
        # Get input embeddings
        x_emb = self.mamba.embedding(x)
        
        # Project features
        features = features.view(1, -1, 2)
        projected_features = self.feature_projection(features)
        expanded_features = projected_features.repeat_interleave(3, dim=1)
        
        # Add features to embeddings
        x_emb = x_emb + expanded_features
        
        # Process through Mamba layers
        for layer in self.mamba.layers:
            x_emb = layer(x_emb)
            
        x_emb = self.mamba.norm_f(x_emb)
        logits = self.mamba.lm_head(x_emb)
        
        last_token_logits = logits[0, -1]
        
        # Apply position-specific mask
        mask = self.position_masks[position_in_message]
        masked_logits = last_token_logits.masked_fill(~mask, -1e9)
        
        next_token = masked_logits.argmax()
        return next_token

class SequenceDataset(Dataset):
    def __init__(self, num_sequences=10000, seq_length=50):
        """Generate synthetic data with position-specific token ranges and context features.
        Args:
            num_sequences: Number of 3-token messages to generate
            seq_length: Length of flattened sequence (must be multiple of 3)
        """
        assert seq_length % 3 == 0, "seq_length must be multiple of 3"
        
        # Generate random sequences following the token range rules
        sequences = []
        features = []
        for _ in range(num_sequences):
            msg = [
                np.random.randint(1, 10),          # Token 1: 1-9
                np.random.randint(100, 200),       # Token 2: 100-199
                np.random.randint(200, 300)        # Token 3: 200-299
            ]
            # Generate random features between 0 and 1
            feature_1 = np.random.random()
            feature_2 = np.random.random()
            
            sequences.append(msg)
            features.append([feature_1, feature_2])
            
        # Convert to numpy arrays
        self.data = np.array(sequences).flatten()
        self.features = np.array(features)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        
        # Get corresponding features for each message in the sequence
        feature_idx = idx // 3
        num_messages = self.seq_length // 3
        features = self.features[feature_idx:feature_idx + num_messages]
        
        return (torch.tensor(x, dtype=torch.long), 
                torch.tensor(features, dtype=torch.float),
                torch.tensor(y, dtype=torch.long))

def main():
    # Training parameters
    batch_size = 32
    seq_length = 21  # Must be multiple of 3
    vocab_size = 300  # Large enough to contain all possible tokens
    
    # Create synthetic dataset
    train_dataset = SequenceDataset(num_sequences=1000, seq_length=seq_length)
    val_dataset = SequenceDataset(num_sequences=100, seq_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    args = ModelArgs(
        d_model=256,          # Hidden dimension
        n_layer=4,            # Number of Mamba blocks
        vocab_size=vocab_size,
        d_state=16,           # SSM state expansion factor
        d_conv=4,             # Local convolution width
        expand=2,             # Block expansion factor
    )
    
    model = PositionalMambaModel(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training parameters
    optimizer = AdamW(model.parameters(), lr=1e-3)
    num_epochs = 10
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, features, y) in enumerate(progress_bar):
            x, features, y = x.to(device), features.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x, features)
            
            # Reshape both logits and targets
            B, L, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            targets_flat = y.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, features, y in val_loader:
                x, features, y = x.to(device), features.to(device), y.to(device)
                logits = model(x, features)
                B, L, V = logits.shape
                logits_flat = logits.reshape(-1, V)
                targets_flat = y.reshape(-1)
                val_loss += F.cross_entropy(logits_flat, targets_flat).item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss/len(train_loader),
            'val_loss': val_loss,
        }, f'mamba_positional_checkpoint_epoch_{epoch+1}.pt')

    # Generate sample sequence
    model.eval()
    with torch.no_grad():
        # Start with a seed sequence and features
        seed_data = train_dataset[0]
        seed = seed_data[0][:6]  # Take first 2 messages as seed
        seed_features = seed_data[1][:2]  # Take features for first 2 messages
        
        x = seed.unsqueeze(0).to(device)  # Add batch dimension
        features = seed_features.unsqueeze(0).to(device)  # Add batch dimension
        
        print("\nStarting generation:")
        print("Seed sequence:", seed.tolist())
        print("Seed features:", seed_features.tolist())
        print("Seed as messages:", [seed.tolist()[i:i+3] for i in range(0, len(seed), 3)])
        
        # Generate next 6 tokens (2 messages)
        generated = []
        for i in range(6):
            position_in_message = i % 3  # 0, 1, or 2
            next_token = model.generate_next_token(x, features, position_in_message)
            generated.append(next_token.item())
            
            # Print each token as it's generated
            print(f"Generated token {i+1} (position {position_in_message}): {next_token.item()}")
            
            # Update input sequence
            x = torch.cat([x[:, 1:], next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        print("\nGenerated sequence:")
        print("Tokens:", generated)
        print("As messages:", [generated[i:i+3] for i in range(0, len(generated), 3)])
        
        # Verify token ranges
        for i, token in enumerate(generated):
            pos = i % 3
            min_val, max_val = model.position_ranges[pos]
            if not (min_val <= token <= max_val):
                print(f"Warning: Token {token} at position {pos} is outside valid range [{min_val}, {max_val}]")

if __name__ == "__main__":
    main() 