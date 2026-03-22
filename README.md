# Objective

This repository implements a configuration-driven pipeline for large-scale CLIP selective-attention experiments.

# Features
## Modular
  - Separates datasets, models, and experiment configurations into dedicated YAML registries.
  - Organizes the workflow into distinct scripts for design-matrix generation, attention and embedding extraction, and session execution.
  
## Scalable
  - Improves efficiency through batch operations and matrix-based computation.
  - Precomputes and stores reusable embeddings, then retrieves subsets as needed to balance speed and computational resources.

## Reproducible
  - Runs scripts through command-line, with configurations managed in YAML files, making workflows easier to track and reproduce.
  - Produces automated Quarto reports to facilitate cross-experiment comparison and inspection.
  - Pre-generates design matrices to standardize trial structure across runs.
  - Fixes random seeds for consistent execution.
 
# Core techinical details

## Extract Attention Map and Rollout in ViT
reference: [Exploring Explainability for Vision Transformers](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)


# Note
`dataset/` is not included in this repository due to copyright restrictions.
