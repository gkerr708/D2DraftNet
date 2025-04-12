# D2DraftNet — Embedding-based Dota 2 Draft Outcome Prediction

## Overview
D2DraftNet is a neural network model that maps discrete Dota 2 drafts (hero picks) into a continuous, low-dimensional vector space. This allows the model to capture latent structure in drafts and learn interaction effects between heroes directly from data.

Given only the set of 10 picked heroes, the model outputs the probability that Radiant will win.

---

## Mathematical Framework

### Draft Representation
Each hero pick is mapped from a categorical variable (hero name) to a learned vector in ℝᵈ via an embedding function:

\[
h_i \rightarrow \mathbf{v}_i \in \mathbb{R}^d
\]

where:
- \( h_i \) = hero index from HERO_MAP
- \( \mathbf{v}_i \) = embedding vector of dimension \(d = \text{embedding\_dim}\)

---

### Team Representation
Each team (Radiant / Dire) is represented as the mean of its 5 hero embedding vectors:

\[
\mathbf{v}_{\text{radiant}} = \frac{1}{5} \sum_{i=1}^{5} \mathbf{v}_i^{(R)}
\]

\[
\mathbf{v}_{\text{dire}} = \frac{1}{5} \sum_{i=1}^{5} \mathbf{v}_i^{(D)}
\]

This projects each draft into ℝᵈ, forming a compact, continuous representation of the team's composition.

---

### Predictive Model
The concatenated team vectors form a vector in ℝ²ᵈ:

\[
\mathbf{x} = [\mathbf{v}_{\text{radiant}} ; \mathbf{v}_{\text{dire}}] \in \mathbb{R}^{2d}
\]

This is passed through a sequence of learned affine transformations (linear layers) with non-linear activations to predict the probability of Radiant victory:

\[
P(\text{Radiant Win}) = \sigma( f(\mathbf{x}) )
\]

where:
- \( f(\cdot) \) = neural network (linear → ReLU → linear → ReLU → linear)
- \( \sigma(\cdot) \) = sigmoid function

---

## Why Embeddings?
- Heroes are categorical → one-hot encoding loses relational information.
- Embeddings learn a *metric space* where similar heroes or synergies cluster together.
- Distances and directions in this space encode draft structure.

This is standard in NLP (word embeddings) but applied here to team composition.

---

## Training Details
| Component | Method |
|-----------|--------|
| Loss | Binary Cross Entropy |
| Optimizer | Adam |
| Input | Hero drafts → Embeddings → Team Vectors |
| Output | Probability of Radiant win |

