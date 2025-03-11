import torch
import torch.nn as nn
from embedding_model import DraftPredictionNN, load_model
from config import TRAINED_MODEL_PATH, HERO_MAP, EMBEDDING_DIM, LAYERS, NUM_HEROS

def get_hero_indices(hero_list):
    """Convert hero names to model input indices."""
    indices = [HERO_MAP[hero] for hero in hero_list if hero in HERO_MAP]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def predict_draft(radiant_heroes, dire_heroes):
    """Loads the trained model and predicts the draft outcome."""
    model = DraftPredictionNN(num_heroes=NUM_HEROS, embedding_dim=EMBEDDING_DIM, dropout_prob=1e-3, layers=LAYERS)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model.eval()
    
    # Convert hero names to indices
    radiant_team = get_hero_indices(radiant_heroes)
    dire_team = get_hero_indices(dire_heroes)
    
    # Pad teams for model input
    radiant_team = nn.utils.rnn.pad_sequence([radiant_team[0]], batch_first=True, padding_value=0)
    dire_team = nn.utils.rnn.pad_sequence([dire_team[0]], batch_first=True, padding_value=0)
    
    # Predict the outcome
    with torch.no_grad():
        prediction = model(radiant_team, dire_team)
        return prediction.item()

if __name__ == "__main__":
    radiant_heroes = ["Lycan", "Enchantress", "Dragon Knight", "Lion", "Ember Spirit"]
    dire_heroes = ["Gyrocopter", "Terrorblade", "Bounty Hunter", "Earth Spirit", "Tinker"]
    prediction = predict_draft(radiant_heroes, dire_heroes)
    winner = "Radiant" if prediction > 0.5 else "Dire"
    print(f"{winner} wins with confidence {prediction:.2f}")
