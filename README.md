# D2DraftNet — Embedding-based Dota 2 Draft Outcome Prediction

## Overview
D2DraftNet is a neural network model that predicts the outcome of a Dota 2 match based only on the hero draft. The model learns to represent each hero pick as a point in a continuous, low-dimensional vector space. This allows it to capture relationships between heroes and infer how different drafts interact.

Given the 10 picked heroes (5 Radiant, 5 Dire), the model outputs the probability that Radiant will win.

---

## Draft Representation

Each hero is assigned a learned embedding vector in `R^d`:

```
h_i → v_i ∈ R^d
```

where:
- `h_i` is a hero ID (categorical)
- `v_i` is the embedding vector for hero `h_i`

---

## Team Representation

A team's draft is represented as the average of its 5 hero embeddings:

```
v_radiant = (1/5) * sum of Radiant hero vectors
v_dire    = (1/5) * sum of Dire hero vectors
```

This reduces a team's draft to a single vector summarizing its overall composition in the embedding space.

---

## Predictive Model

The two team vectors are concatenated:

```
x = [v_radiant ; v_dire] ∈ R^(2d)
```

This combined draft vector is passed through a fully-connected neural network:

```
Embedding → Mean Pooling → Concatenation → Linear Layers → Sigmoid Output
```

The output is:

```
P(Radiant Win) = sigmoid( f(x) )
```

where `f(x)` is the neural network's learned transformation.

---

## Why Embeddings?

One-hot encoding treats every hero as unrelated to every other.  
Embeddings allow the model to learn:

- Similarity between heroes
- Counters and synergies
- Team identity as a point in continuous space
- Complex draft-level interactions

---

## Training Details

| Component  | Method                       |
|------------|--------------------------------|
| Loss       | Binary Cross Entropy (BCE)    |
| Optimizer  | Adam                          |
| Input      | Hero Drafts → Embeddings → Team Vectors |
| Output     | Probability of Radiant Win    |

---

## Project Structure

```
d2draftnet/
├── config.py             # Hero map, data loading
├── embedding_model.py   # Embedding model + Dataset
├── model_training.py    # Training loop, evaluation, plotting
└── data/                # Match dataset
```

---

## Example Usage

```python
from d2draftnet.model_training import ModelTraining

trainer = ModelTraining(
    embedding_dim=3,
    dropout_prob=1e-3,
    batch_size=512,
    learning_rate=5e-4,
    epochs=50,
    layers=[32, 16],
    train_test_split=0.2
)

trainer.train_model()
trainer.evaluate_model(verbose=True, save_bool=True)
```

---

# Diagram — Draft Embedding Pipeline

```
Radiant Draft   Dire Draft
 [h1 h2 h3 h4 h5]   [h6 h7 h8 h9 h10]
        ↓               ↓
  Embedding Layer  Embedding Layer
        ↓               ↓
 Radiant Mean       Dire Mean
        ↓               ↓
       Concatenation: [v_radiant ; v_dire]
                    ↓
         Fully Connected Layers
                    ↓
            Predicted Win Probability
```

---

# Diagram — Embedding Space Intuition

Imagine similar heroes cluster in embedding space:

```
+--------------------------+
|                          |
|     Tanky Heroes        X
|                        X   X
|                         X
|   Push/Healing Heroes  X
|                  X
|           X
|   Nuke/Burst Heroes    X
|                          |
+--------------------------+
```

> Draft strength arises from the *positions and combinations* of heroes in this space.

---

## Requirements

```bash
pip install torch pandas sklearn matplotlib numpy scipy
```

---

*D2DraftNet learns how drafts win — not just which heroes are strong individually, but how they interact as a team.*

