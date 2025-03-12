import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import HERO_MAP

# Dataset class
class Dota2DraftDataset(Dataset):
    def __init__(self, data, labels):
        self.data: list[dict] = data
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # The row represents a one hot encoded draft
        match_info = self.data.iloc[idx]

        # Construct the radiant and dire teams as a list of indices
        radiant_draft_names = match_info["radiant_draft"]
        dire_draft_names = match_info["dire_draft"]

        # Convert the hero names to indices
        radiant_draft_idx = [HERO_MAP[hero] for hero in radiant_draft_names]
        dire_draft_idx    = [HERO_MAP[hero] for hero in dire_draft_names]

        # Convert the indices to tensors
        radiant_draft_tensor = torch.tensor(radiant_draft_idx, dtype=torch.long)
        dire_draft_tensor = torch.tensor(dire_draft_idx, dtype=torch.long)

        # Return the draft tensors and the label
        return radiant_draft_tensor , dire_draft_tensor, self.labels[idx]


# Embedding model class
class DraftPredictionNN(nn.Module):
    def __init__(self, num_heroes: int, embedding_dim: int, dropout_prob: float, layers: list):
        super(DraftPredictionNN, self).__init__()
        self.embedding = nn.Embedding(num_heroes, embedding_dim, padding_idx=0)

        # verify layers is a list of 2 integers
        assert len(layers) == 2, "layers must be a list of 2 integers"
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, layers[0]),  # Increased the size of the first layer
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Dropout to reduce overfitting
            nn.Linear(layers[0], layers[1]),         # Added another hidden layer
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Another dropout layer
            nn.Linear(layers[1], 1),
            nn.Sigmoid()
        )

    def forward(self, radiant_team, dire_team):
        # Compute team embeddings by averaging hero embeddings
        radiant_embed = torch.mean(self.embedding(radiant_team), dim=1)  # Shape [batch_size, embedding_dim]
        dire_embed = torch.mean(self.embedding(dire_team), dim=1)        # Shape [batch_size, embedding_dim]
        
        # Concatenate radiant and dire embeddings along the last dimension
        combined = torch.cat([radiant_embed, dire_embed], dim=1)  # Shape [batch_size, embedding_dim * 2]
        
        # Pass through fully connected layers
        return self.fc(combined)


# Helper functions
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, num_heroes, embedding_dim, dropout_prob, layers):
    model = DraftPredictionNN(num_heroes=num_heroes, embedding_dim=embedding_dim, dropout_prob=dropout_prob, layers=layers)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

if __name__ == "__main__":

    # Test the model and dataset class
    from .config import load_data

    # Load the dataset
    data = load_data()

    # The row represents a one hot encoded draft
    match_info = data.iloc[2]

    # Construct the radiant and dire teams as a list of indices
    radiant_draft_names = match_info["radiant_draft"]
    dire_draft_names = match_info["dire_draft"]

    # Convert the hero names to indices
    radiant_draft_idx = [HERO_MAP[hero] for hero in radiant_draft_names]
    dire_draft_idx    = [HERO_MAP[hero] for hero in dire_draft_names]

    # Convert the indices to tensors
    radiant_draft_tensor = torch.tensor(radiant_draft_idx, dtype=torch.long)
    dire_draft_tensor = torch.tensor(dire_draft_idx, dtype=torch.long)

    print(f"Radiant draft: {radiant_draft_names}")
    print(f"Radiant draft: {radiant_draft_idx}")
    print(f"Radiant draft: {radiant_draft_tensor}")

