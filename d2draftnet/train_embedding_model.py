import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn
from pathlib import Path
from config import HEROS, MATCH_DATA_PATH, TRAINED_MODEL_DIRECTORY, MODEL_PATH, load_data
from embedding_model import Dota2DraftDataset, DraftPredictionNN
import matplotlib.pyplot as plt

@dataclass
class ModelTraining:
    embedding_dim: int
    dropout_prob: float
    batch_size: int
    learning_rate: float
    epochs: int
    layers: list 
    train_test_split: float
    trained_models_dir: Path = TRAINED_MODEL_DIRECTORY

    def __post_init__(self):
        # Check if CUDA is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Ensure the trained models directory exists
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

        # Load the dataframe
        try:
            data = load_data()
            print(f"Loaded data from {MATCH_DATA_PATH}...")
        # Load the data with the detected or fallback encoding
        except Exception as e:
            print(f"Failed to load CSV with encoding. Error: {e}")
            raise

        print(f"Training from N = {len(data):,} samples")

        # Convert Winner column to binary labels
        labels = data["winner"].apply(lambda x: 1 if x == "Radiant" else 0).values

        # Split into training and testing datasets
        self.train_data, self.test_data, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=self.train_test_split, #random_state=42
        )


        # Initialize train DataLoader
        self.train_loader = DataLoader(
            Dota2DraftDataset(self.train_data, self.y_train), # Converts [Axel, Bane, Kez, ...] to [1, 2, 3, ...]
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # Initialize test DataLoader
        self.test_loader = DataLoader(
            Dota2DraftDataset(self.test_data, self.y_test),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # Get the totol number of heroes
        self.num_heroes = len(HEROS) + 1

        # Initialize the model
        self.model = DraftPredictionNN(
            num_heroes=self.num_heroes,
            embedding_dim=self.embedding_dim,
            dropout_prob=self.dropout_prob,  # Pass dropout probability to the model
            layers=self.layers  # Pass the layers to the model
        )

        # Use binary cross-entropy loss
        self.criterion = nn.BCELoss()

        # Use Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def train_model(self, show_plot: bool = True, verbose: bool =True, return_data: bool = False) -> list:
        """Train the model."""
        accuracy_values = []
        for epoch in range(self.epochs):
            self.model.train()
            for radiant_team, dire_team, labels in self.train_loader:
                radiant_team = nn.utils.rnn.pad_sequence(radiant_team, batch_first=True, padding_value=0)
                dire_team = nn.utils.rnn.pad_sequence(dire_team, batch_first=True, padding_value=0)
                labels = torch.stack(labels)

                # Forward pass
                outputs = self.model(radiant_team, dire_team)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accuracy
                predictions = (outputs > 0.5).float()
                correct = (predictions == labels).sum().item()
                accuracy = correct / labels.size(0)

            if verbose:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")
            accuracy_values.append(accuracy)

        if show_plot:
            # smooth the accuracy values
            window = 7
            raw_accuracy = accuracy_values.copy()
            smooth_accuracy = accuracy_values.copy()
            N_smooths = 12
            for _ in range(N_smooths):
                smooth_accuracy = np.convolve(accuracy_values, np.ones(window) / window, mode='valid')
            plt.plot(raw_accuracy, label='Raw Accuracy', alpha=0.5, color='gray')
            plt.scatter(np.arange(len(raw_accuracy)), raw_accuracy, label='Raw Accuracy')
            plt.plot(np.linspace(0, len(raw_accuracy), len(smooth_accuracy)), smooth_accuracy )
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Training Results')
            plt.savefig('training_loss.png')
            plt.show()

        if return_data:
            return accuracy_values
        else:
            return None

    def evaluate_model(self, verbose: bool = False, save_bool: bool = False) -> float:
        """Evaluate the model on the test set."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for radiant_team, dire_team, labels in self.test_loader:
                radiant_team = nn.utils.rnn.pad_sequence(radiant_team, batch_first=True, padding_value=0)
                dire_team = nn.utils.rnn.pad_sequence(dire_team, batch_first=True, padding_value=0)
                labels = torch.stack(labels)

                outputs = self.model(radiant_team, dire_team)
                predictions = (outputs > 0.5).float()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        if verbose:
            print(f"Test Accuracy: {accuracy:.4f}")
        if save_bool:
            self.save_model()
        return accuracy

    def save_model(self):
        """Save the trained model to a file."""
        filepath = MODEL_PATH
        if not filepath.exists():
            torch.save(self.model.state_dict(), filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Model file already exists.")
            val = input("(y/n) Overwrite the model file? ")
            if val.lower() == "y":
                torch.save(self.model.state_dict(), filepath)
                print(f"Model saved to {filepath}")
            else:
                print("Model not saved.")

if __name__ == "__main__":

    # Define the base parameters
    embedding_dim = 3; dropout_prob = 1e-3; batch_size = 2**9
    learning_rate = 5e-4; epochs = 50; test_train_split = 0.2
    layers = [32, 16]

    dict_of_param_dicts = {
        'Base Model': {
            'embedding_dim': embedding_dim,
            'dropout_prob': dropout_prob,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'train_test_split': test_train_split,
            'layers': layers
        },
    }

    train_accuracy_list = []
    test_accuracy_list = []
    for name, params in dict_of_param_dicts.items():
        print(f'Training model with parameters: {params}')
        # Initialize the ModelTraining class
        trainer = ModelTraining(
            embedding_dim=params['embedding_dim'],
            dropout_prob=params['dropout_prob'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            layers=params['layers'],
            train_test_split=params['train_test_split']
        )

        N_train = len(trainer.train_data)
        N_test = len(trainer.test_data)

        # Train the model
        train_accuracy = trainer.train_model(show_plot=True, verbose=True, return_data=True)

        # Test the model
        test_accuracy = trainer.evaluate_model(verbose=True, save_bool=True)

        # Save the accuracy
        mean_final_train_accuracy = np.mean(train_accuracy[-4:-1])
        train_accuracy_list.append(mean_final_train_accuracy)
        test_accuracy_list.append(test_accuracy)

    from scipy.stats import binom

    # Plot the binomial distributions
    trial_names = list(dict_of_param_dicts.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    width = 0.25  # Width of the bars
    x = np.arange(len(trial_names))  # Base positions for models

    for i, (train_accuracy, test_accuracy, name) in enumerate(zip(train_accuracy_list, test_accuracy_list, trial_names)):
        # Compute the possible number of successes and corresponding likelihoods
        successes = np.arange(0, N_test + 1)  # Possible number of successes
        likelihoods = binom.pmf(successes, N_test, test_accuracy)  # Likelihood of each outcome

        # Convert successes to accuracy for plotting
        probabilities = successes / N_test

        # Bar plot for train and test accuracy
        axes[0].bar(x[i] - width / 2, train_accuracy, width, color=f"C{i}")
        axes[0].bar(x[i] + width / 2, test_accuracy, width, color=f"C{i}", alpha=0.5)

        # Print the accuracy on the bars
        axes[0].text(x[i] - width / 2, train_accuracy, f"{train_accuracy:.2%}", ha='center', va='bottom')
        axes[0].text(x[i] + width / 2, test_accuracy, f"{test_accuracy:.2%}", ha='center', va='bottom')

        # Plot the likelihood curve for this model
        axes[1].plot(probabilities, likelihoods, label=f"{name} Test Distribution", linestyle='-', color=f"C{i}")
        # Add a vertical line for train accuracy
        axes[1].axvline(train_accuracy, color=f"C{i}", linestyle=':', label=f"{name} Train Accuracy", alpha=0.8)

    # Configure bar plot (axes[0])
    axes[0].set_title("Train vs Test Accuracies")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trial_names)
    #axes[0].legend()

    # Configure likelihood plot (axes[1])
    axes[1].set_title("Likelihood of Test Accuracy")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Likelihood")
    axes[1].legend()
    axes[1].set_xlim(0.4, 0.8)
    axes[1].grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
