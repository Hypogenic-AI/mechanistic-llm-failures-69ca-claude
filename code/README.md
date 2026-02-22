# Code Repositories

Code repositories cloned for the research project: **Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs**

## Repositories

### TransformerLens
- **Source:** https://github.com/TransformerLensOrg/TransformerLens
- **Description:** A library for mechanistic interpretability of GPT-2 style language models. Enables researchers to reverse engineer the algorithms learned during training.
- **Key Features:**
  - Loads 50+ open source language models (GPT-2, GPT-Neo, Pythia, Llama, etc.)
  - Exposes all internal activations via hook-based interventions
  - Supports activation caching, editing, removal, and replacement
  - Foundation for path patching, activation patching, and circuit discovery
- **Local path:** `code/TransformerLens/`
- **Role in project:** Primary tool for model loading and mechanistic analysis

### SAELens
- **Source:** https://github.com/jbloomAus/SAELens
- **Description:** A library for training and analyzing Sparse Autoencoders (SAEs) for mechanistic interpretability research.
- **Key Features:**
  - SAE training on any PyTorch model
  - Deep integration with TransformerLens
  - HuggingFace Transformers support
  - Pre-trained SAEs available for common models
- **Local path:** `code/SAELens/`
- **Role in project:** Training SAEs to decompose activations during commonsense reasoning

### SAE-Reasoning
- **Source:** https://github.com/AIRI-Institute/SAE-Reasoning
- **Description:** Implementation of "Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders" (Galichin et al., 2025).
- **Key Features:**
  - SAE training on reasoning datasets (DeepSeek-R1 distilled models)
  - ReasonScore computation for identifying reasoning features
  - Feature interpretation and steering experiment code
  - Model diffing for tracking feature emergence
- **Local path:** `code/SAE-Reasoning/`
- **Role in project:** Reference implementation for adapting ReasonScore to commonsense reasoning

### ACDC (Automatic Circuit Discovery)
- **Source:** https://github.com/ArthurConmy/ACDC
- **Description:** Automated Circuit Discovery for Mechanistic Interpretability, accompanying the NeurIPS 2023 paper by Conmy et al.
- **Key Features:**
  - Identifies minimal computational subgraphs for specific behaviors
  - Edge editing in transformer computational graphs
  - Built on TransformerLens HookPoints
  - Supports circuit validation metrics
- **Local path:** `code/ACDC/`
- **Role in project:** Automated discovery of circuits responsible for commonsense reasoning

### rome (Rank-One Model Editing)
- **Source:** https://github.com/kmeng01/rome
- **Description:** Implementation of causal tracing and ROME for efficiently editing factual associations in transformers.
- **Key Features:**
  - Causal tracing for localizing knowledge storage
  - ROME editing for GPT-2 XL and GPT-J
  - CounterFact evaluation benchmark
  - Visualization tools for causal analysis
- **Local path:** `code/rome/`
- **Role in project:** Reference for causal tracing methodology applied to commonsense knowledge

## Note on .gitignore

These repositories are excluded from version control due to their size. They can be re-cloned using the URLs listed above.
