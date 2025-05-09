
# D2DraftNet — Embedding-Based Model for Dota 2 Draft Prediction

## Overview
D2DraftNet is a neural network model that predicts the outcome of a Dota 2 match using only the hero draft. The core idea is to represent each hero pick not as a one-hot vector, but as a learned point in a low-dimensional vector space. This allows the model to capture complex, non-obvious relationships between heroes — such as synergies, counters, or general playstyle tendencies.

---

## Draft Representation

### Hero Embedding
Each hero is mapped to a learned vector in $R^d$, where \(d\) is a tunable hyperparameter (typically small, e.g., \(d=3\)).

Formally:

$$
h_i \mapsto \mathbf{v}_i \in \mathbb{R}^d
$$

where:
- $h_i$ is the hero ID
- $\mathbf{v}_i$ is its embedding vector (learned during training)

---

### Team Embedding
A team's draft (5 heroes) is represented by the *average* of its hero embeddings:

$$
v_{team} = \frac{1}{5} \sum_{i=1}^{5} v_i
$$

This gives a single vector summarizing the composition and identity of the team in a continuous space.

---

## Model Architecture

The model takes the two team vectors — Radiant and Dire — and concatenates them into a single vector:

$$
x = [v_{radiant}; v_{dire}]
$$

This is passed through a simple fully-connected neural network:
```
Embedding Layer → Mean Pooling → Concatenation → Linear Layers → Sigmoid Output
```

- Linear layers perform affine transformations.
- Non-linearities (ReLU) allow for complex decision boundaries.
- The final output is a single probability: \(P(\text{Radiant Win})\)

---

## Why Embeddings?

One-hot encoding treats every hero as equally distant from every other hero — which is unrealistic in Dota 2.

Embedding the draft into a learned vector space allows the model to:
- Group similar heroes together.
- Capture interactions and matchups.
- Compress drafts into dense, informative vectors.
- Learn synergy and counterplay patterns from data — not from hand-crafted features.

---

## Training

| Component     | Method                         |
|---------------|--------------------------------|
| Loss Function | Binary Cross Entropy (BCE)    |
| Optimizer     | Adam                          |
| Input         | Radiant and Dire drafts       |
| Output        | Probability of Radiant Victory|
