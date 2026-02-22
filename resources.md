# Resources Catalog: Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs

> A comprehensive catalog of papers, datasets, code repositories, methods, and experimental workflows
> for investigating how and why large language models fail at commonsense reasoning tasks,
> examined through the lens of mechanistic interpretability.

---

## 1. Papers

All papers are organized by category. Each entry includes the arXiv ID, title, authors (abbreviated), year, and a one-sentence description of its relevance to this research project. Local PDF copies are stored in `papers/`.

---

### Mechanistic Interpretability Foundations

1. **arXiv:2211.00593** -- "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small"
   - *Wang et al., 2022*
   - Establishes the circuit discovery methodology and path patching technique used to isolate the computational subgraph responsible for a specific behavior in a transformer, providing the foundational approach we adapt for commonsense reasoning circuits.
   - Local: `papers/elhage2022_transformer_circuits_IOI.pdf`

2. **arXiv:2202.05262** -- "Locating and Editing Factual Associations in GPT (ROME)"
   - *Meng et al., 2022*
   - Introduces causal tracing to localize where factual knowledge is stored (primarily in mid-layer feed-forward networks) and the ROME method for targeted knowledge editing, directly applicable to understanding how commonsense facts are stored and retrieved.
   - Local: `papers/meng2022_ROME_locating_editing_factual.pdf`

3. **arXiv:2309.10312** -- "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
   - *Bricken et al., 2023*
   - Demonstrates that sparse autoencoders (SAEs) can decompose neural activations into interpretable, monosemantic features, providing the technical foundation for extracting commonsense-specific features from model internals.
   - Local: `papers/bricken2023_towards_monosemanticity.pdf`

4. **arXiv:2310.17230** -- "Codebook Features: Sparse Discrete Representations for Interpretability"
   - *Tamkin et al., 2023*
   - Proposes an alternative sparse discrete representation scheme for neural network interpretability, offering a complementary approach to SAEs for identifying discrete commonsense reasoning features.
   - Local: `papers/tamkin2023_codebook_features.pdf`

5. **arXiv:2310.01405** -- "Representation Engineering: A Top-Down Approach to AI Transparency"
   - *Zou et al., 2023*
   - Presents a top-down methodology for reading and controlling model representations, enabling identification and manipulation of high-level commonsense reasoning directions in activation space.
   - Local: `papers/zou2023_representation_engineering.pdf`

6. **arXiv:2104.08696** -- "Knowledge Neurons in Pretrained Transformers"
   - *Dai et al., 2021*
   - Identifies specific neurons in feed-forward layers that store factual knowledge, providing a technique for locating neurons that may encode commonsense knowledge and diagnosing failures at the neuron level.
   - Local: `papers/dai2021_knowledge_neurons.pdf`

7. **arXiv:2403.19647** -- "Sparse Feature Circuits: Discovering Mechanisms in the Wild"
   - *Marks et al., 2024*
   - Extends circuit discovery to operate over sparse autoencoder features rather than raw neurons, enabling the discovery of interpretable causal graphs that may reveal how commonsense reasoning is mechanistically implemented.
   - Local: `papers/marks2024_sparse_feature_circuits.pdf`

8. **arXiv:2310.06824** -- "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"
   - *Marks and Tegmark, 2023*
   - Reveals that truth-value representations form linear structure in LLM activation space, suggesting that commonsense correctness may similarly be geometrically organized and detectable via linear probes.
   - Local: `papers/marks2023_geometry_of_truth.pdf`

9. **arXiv:2406.11717** -- "Refusal in Language Models Is Mediated by a Single Direction"
   - *Arditi et al., 2024*
   - Demonstrates that complex model behaviors (refusal) can be mediated by a single direction in activation space, motivating the search for similarly compact directional representations of commonsense reasoning competence or failure modes.
   - Local: `papers/arditi2024_refusal_single_direction.pdf`

10. **arXiv:2305.01610** -- "Finding Neurons in a Haystack: Case Studies with Sparse Probing"
    - *Gurnee et al., 2023*
    - Develops sparse probing techniques that identify small sets of neurons encoding specific concepts, directly applicable to finding neurons responsible for physical intuition, social reasoning, and other commonsense domains.
    - Local: `papers/gurnee2023_finding_neurons_haystack.pdf`

11. **arXiv:2304.14767** -- "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"
    - *Geva et al., 2023*
    - Identifies a three-step mechanism (subject enrichment, relation propagation, attribute extraction) for factual recall, providing a template for decomposing commonsense reasoning into analogous mechanistic stages.
    - Local: `papers/geva2023_dissecting_recall_factual.pdf`

12. **arXiv:2304.15004** -- "Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)"
    - *Conmy et al., 2023*
    - Introduces an automated approach to discovering minimal circuits responsible for specific model behaviors, enabling scalable identification of commonsense reasoning circuits without exhaustive manual search.
    - Local: `papers/conmy2023_automated_circuit_discovery_ACDC.pdf`

---

### Commonsense Reasoning Benchmarks

13. **arXiv:1911.11641** -- "PIQA: Reasoning about Physical Intuition By Choice"
    - *Bisk et al., 2020*
    - Provides the primary physical commonsense reasoning benchmark (goal + two candidate solutions), used to evaluate and probe how models represent physical intuition about everyday actions.
    - Local: `papers/bisk2020_piqa_physical_commonsense.pdf`

14. **arXiv:1811.00937** -- "CommonsenseQA: A Question Answering Challenge Targeting World Knowledge"
    - *Talmor et al., 2019*
    - Introduces a five-choice commonsense question answering benchmark grounded in ConceptNet, serving as the primary evaluation for general commonsense knowledge retrieval and reasoning.
    - Local: `papers/talmor2019_commonsenseqa.pdf`

15. **arXiv:1905.07830** -- "HellaSwag: Can a Machine Really Finish Your Sentence?"
    - *Zellers et al., 2019*
    - Presents an adversarially-filtered sentence completion benchmark requiring commonsense to select plausible continuations, testing whether models can apply commonsense to predict coherent event sequences.
    - Local: `papers/zellers2019_hellaswag.pdf`

16. **arXiv:1907.10641** -- "WinoGrande: An Adversarial Winograd Schema Challenge at Scale"
    - *Sakaguchi et al., 2020*
    - Provides a large-scale adversarial pronoun resolution benchmark requiring commonsense reasoning, useful for probing whether models mechanistically resolve coreference via commonsense or shallow heuristics.
    - Local: `papers/sakaguchi2020_winogrande.pdf`

---

### Probing Methods

17. **arXiv:1909.03368** -- "Designing and Interpreting Probes with Control Tasks"
    - *Hewitt and Manning, 2019*
    - Establishes rigorous methodology for probing neural network representations with selectivity controls, ensuring that any commonsense probes we design measure genuine knowledge rather than probe memorization.
    - Local: `papers/hewitt2019_probes_control_tasks.pdf`

18. **arXiv:1909.01066** -- "Language Models as Knowledge Bases? (LAMA)"
    - *Petroni et al., 2019*
    - Introduces the cloze-based probing paradigm for evaluating factual knowledge stored in language models, adaptable to probing commonsense knowledge via appropriately designed fill-in-the-blank templates.
    - Local: `papers/petroni2019_language_models_knowledge_bases.pdf`

19. **arXiv:2106.09346** -- "Probing Across Time: What Does RoBERTa Know and When?"
    - *Liu et al., 2021*
    - Examines how knowledge representations evolve across training time and layers, informing our analysis of when and where commonsense knowledge is acquired and potentially lost during model processing.
    - Local: `papers/liu2021_probing_across_time.pdf`

---

### Reasoning Analysis

20. **arXiv:2305.18354** -- "Faith and Fate: Limits of Transformers on Compositionality"
    - *Dziri et al., 2023*
    - Demonstrates fundamental limitations of transformers on compositional reasoning tasks, providing theoretical grounding for understanding why commonsense reasoning -- which often requires composing multiple facts -- fails in LLMs.
    - Local: `papers/dziri2023_faith_fate_compositionality.pdf`

21. **arXiv:2210.13382** -- "Measuring Causal Effects of Data Statistics on Machine Learning Model Predictions"
    - *Elazar et al., 2022*
    - Develops methods for measuring how data statistics causally affect model predictions, applicable to understanding whether commonsense failures are driven by training data distribution artifacts or architectural limitations.
    - Local: `papers/elazar2022_causal_effects_data_statistics.pdf`

---

### Sparse Autoencoders

22. **arXiv:2408.05147** -- "Scaling and Evaluating Sparse Autoencoders"
    - *Gao et al., 2024*
    - Provides systematic analysis of how SAE quality scales with model size and dictionary size, guiding hyperparameter choices for training SAEs on commonsense reasoning activations.
    - Local: `papers/gao2024_scaling_evaluating_SAEs.pdf`

23. **arXiv:2406.04093** -- "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
    - *Templeton et al., 2024*
    - Demonstrates that SAE-based feature extraction scales to production-grade models (Claude 3 Sonnet), validating the approach of using SAEs to find interpretable commonsense features in larger models.
    - Local: `papers/templeton2024_scaling_monosemanticity_claude.pdf`

---

### Directly Relevant Recent Work

24. **arXiv:2503.18878** -- "Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders"
    - *Galichin et al., 2025*
    - Directly applies SAEs to discover reasoning-specific features and introduces the ReasonScore metric, providing the most immediately applicable methodology for identifying commonsense reasoning features and understanding their role in failures.
    - Local: `papers/galichin2025_interpreting_reasoning_features_SAE.pdf`

25. **"How does Chain of Thought Think? A Mechanistic Interpretability Study of Chain-of-Thought Reasoning"**
    - *Chen et al., 2025*
    - Investigates the internal mechanisms of chain-of-thought reasoning using mechanistic interpretability, offering insight into how step-by-step reasoning processes (including commonsense steps) are implemented in transformers.
    - Local: `papers/chen2025_how_CoT_think_mechanistic_SAE.pdf`

26. **"Do VLMs Have Bad Eyes? Diagnosing Compositional Failures via Mechanistic Interpretability"**
    - *Aravindan et al., 2025*
    - Applies mechanistic interpretability to diagnose compositional reasoning failures in vision-language models, providing a parallel methodology for diagnosing commonsense compositional failures in text-only LLMs.
    - Local: `papers/aravindan2025_VLMs_bad_eyes_compositional_failures.pdf`

---

## 2. Datasets

Each dataset is used to evaluate and probe commonsense reasoning capabilities. All datasets are sourced from HuggingFace and stored locally with sample files for quick inspection.

| Dataset | Source | Train / Val / Test Size | Format | Tests | Local Path |
|---------|--------|------------------------|--------|-------|------------|
| PIQA | HuggingFace: `piqa` | 16,113 / 1,838 / 3,084 | Goal + 2 candidate solutions + label | Physical commonsense reasoning (everyday physical interactions) | `datasets/piqa/` |
| CommonsenseQA | HuggingFace: `tau/commonsense_qa` | 9,741 / 1,221 / 1,140 | Question + 5 answer choices + answer key | General commonsense knowledge (ConceptNet-grounded) | `datasets/commonsenseqa/` |
| HellaSwag | HuggingFace: `hellaswag` | 39,905 / 10,042 / 10,003 | Context + 4 candidate endings + label | Sentence completion requiring commonsense (event plausibility) | `datasets/hellaswag/` |
| WinoGrande | HuggingFace: `allenai/winogrande` (`winogrande_xl`) | 40,398 / 1,267 / 1,767 | Sentence with blank + 2 options + answer | Pronoun resolution requiring commonsense (Winograd schema) | `datasets/winogrande/` |

### Dataset Details

**PIQA** -- Physical Intuition QA. Each example consists of a goal describing an everyday physical task (e.g., "To clean a dusty keyboard") and two candidate solutions. The model must select the physically plausible solution. This is the primary benchmark for probing physical commonsense circuits.

**CommonsenseQA** -- Derived from ConceptNet subgraphs, each question targets a specific commonsense relation (e.g., causes, has-property, at-location). The five distractor choices are semantically related, requiring genuine commonsense discrimination rather than surface-level association.

**HellaSwag** -- Adversarially constructed using ActivityNet and WikiHow so that machine-generated wrong endings are difficult for models but trivial for humans. Tests the model's ability to apply commonsense to predict coherent event sequences.

**WinoGrande** -- An adversarially-filtered, large-scale Winograd schema challenge. Each example requires resolving a pronoun by applying commonsense knowledge about the described situation, testing whether models mechanistically resolve coreference via commonsense reasoning or rely on shallow heuristics.

### Sample Data Files

Quick-inspection JSON sample files are available alongside each dataset directory:

- `datasets/piqa_samples.json`
- `datasets/commonsenseqa_samples.json`
- `datasets/hellaswag_samples.json`
- `datasets/winogrande_samples.json`

---

## 3. Code Repositories

All repositories are cloned locally under the `code/` directory and provide the core tooling for mechanistic interpretability experiments.

| Repository | URL | Description | Key Features | Local Path |
|-----------|-----|-------------|--------------|------------|
| TransformerLens | [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Mechanistic interpretability library for GPT-2 style transformer models | 50+ pre-loaded models, activation caching and editing, hook-based interventions, logit lens, attention pattern visualization | `code/TransformerLens/` |
| SAELens | [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) | Library for training and analyzing sparse autoencoders on transformer activations | SAE training pipelines, HuggingFace integration for pre-trained SAEs, TransformerLens compatibility, feature dashboard generation | `code/SAELens/` |
| SAE-Reasoning | [github.com/AIRI-Institute/SAE-Reasoning](https://github.com/AIRI-Institute/SAE-Reasoning) | Reasoning feature discovery and steering via sparse autoencoders | SAE training on reasoning-specific data, ReasonScore computation for identifying reasoning features, activation steering for reasoning enhancement | `code/SAE-Reasoning/` |
| ACDC | [github.com/ArthurConmy/ACDC](https://github.com/ArthurConmy/ACDC) | Automated Circuit Discovery for Mechanistic Interpretability | Automated edge editing in computational graphs, iterative circuit pruning, circuit validation and faithfulness metrics | `code/ACDC/` |
| rome | [github.com/kmeng01/rome](https://github.com/kmeng01/rome) | Causal tracing and knowledge editing in transformers | ROME and MEMIT editing methods, causal tracing implementation, CounterFact evaluation benchmark | `code/rome/` |

### Repository Usage Notes

**TransformerLens** is the primary workhorse library. It provides `HookedTransformer` models with full access to intermediate activations at every layer, attention head, and MLP component. Key classes:
- `HookedTransformer` -- Load and run models with activation access
- `ActivationCache` -- Store and manipulate cached activations
- `patching` module -- Activation and path patching utilities
- `utils` module -- Logit lens, attention visualization, and helper functions

**SAELens** provides pre-trained SAEs for popular models (GPT-2, Pythia) and tools for training custom SAEs. It integrates directly with TransformerLens for seamless activation extraction and feature analysis.

**SAE-Reasoning** implements the methodology from Galichin et al. (2025), including the ReasonScore metric for quantifying how much a given SAE feature contributes to reasoning performance. This is directly adaptable to commonsense reasoning analysis.

**ACDC** automates the circuit discovery process described in Conmy et al. (2023). Given a model, dataset, and metric, it iteratively prunes edges in the computational graph to find the minimal circuit responsible for the target behavior.

**rome** provides the causal tracing infrastructure from Meng et al. (2022), enabling localization of where commonsense knowledge is stored by systematically corrupting and restoring activations across layers.

---

## 4. Key Methods and Tools

A summary of the primary methodologies available through the collected resources, organized by analysis type.

### Circuit Discovery Methods

- **Path Patching** *(TransformerLens; Wang et al., 2022 IOI paper)*
  Traces information flow between specific model components (attention heads, MLPs) by replacing activations along specific paths while holding others fixed. Identifies which component-to-component connections are critical for commonsense reasoning.

- **Activation Patching** *(TransformerLens)*
  Replaces activations at a specific layer/position from a clean run with those from a corrupted run (or vice versa) to measure the causal effect of that component on the output. Quantifies each component's contribution to commonsense predictions.

- **Attention Knockout** *(Geva et al., 2023)*
  Zeros out specific attention edges (e.g., all attention from position A to position B) to test whether information flow through those edges is necessary for correct predictions. Identifies critical attention-mediated information pathways for commonsense queries.

- **ACDC -- Automated Circuit Discovery** *(ACDC repo; Conmy et al., 2023)*
  Automated circuit discovery via iterative edge pruning. Starting from the full computational graph, ACDC removes edges that have minimal effect on a target metric, converging on a minimal sufficient circuit. Scales circuit discovery beyond what is feasible manually.

### Feature Decomposition Methods

- **Sparse Autoencoders (SAEs)** *(SAELens; Bricken et al., 2023; Gao et al., 2024)*
  Train sparse autoencoders on model activations to decompose polysemantic neurons into monosemantic features. Applied to commonsense reasoning activations, SAEs can reveal interpretable features corresponding to physical properties, social norms, causal relationships, and other commonsense concepts.

- **ReasonScore** *(SAE-Reasoning; Galichin et al., 2025)*
  A metric for identifying reasoning-related SAE features by measuring each feature's differential activation between correct reasoning traces and control conditions. Adaptable to score features for commonsense-reasoning specificity.

- **Codebook Features** *(Tamkin et al., 2023)*
  An alternative to SAEs that uses discrete codebook entries for sparse representations. Provides a complementary lens for identifying discrete commonsense reasoning primitives.

### Knowledge Localization Methods

- **Causal Tracing** *(rome repo; Meng et al., 2022)*
  Localizes knowledge storage by corrupting all input embeddings, then selectively restoring individual layer activations to measure which layers are critical for recovering correct outputs. Identifies the layers where commonsense knowledge is stored.

- **ROME Editing** *(rome repo; Meng et al., 2022)*
  Directly modifies feed-forward network weights to insert, update, or delete specific factual associations. Can be used to test whether commonsense failures can be repaired by targeted weight edits, confirming localization hypotheses.

- **Knowledge Neurons** *(Dai et al., 2021)*
  Identifies specific neurons in FFN layers that are activated for particular facts by measuring integrated gradients with respect to the fill-in-the-blank knowledge expression probability. Applicable to finding neurons encoding commonsense facts.

### Representation Analysis Methods

- **Vocabulary Projection / Logit Lens** *(TransformerLens; nostalgebraist, 2020)*
  Projects intermediate hidden states through the unembedding matrix to obtain a probability distribution over the vocabulary at each layer. Reveals how commonsense-relevant tokens rise or fall in probability across layers, diagnosing where reasoning goes wrong.

- **Linear Probing** *(Hewitt and Manning, 2019; Gurnee et al., 2023)*
  Trains linear classifiers on intermediate representations to detect the presence of specific commonsense properties (e.g., "is_physical_object", "can_be_heated", "is_dangerous"). Measures where and when commonsense knowledge is linearly accessible.

- **Representation Engineering** *(Zou et al., 2023)*
  Identifies and manipulates high-level concept directions in activation space using contrastive stimuli. Can be used to find a "commonsense reasoning" direction and test whether amplifying it improves performance on failure cases.

- **Geometry of Truth Analysis** *(Marks and Tegmark, 2023)*
  Analyzes the geometric structure of truth-value representations. Applied to commonsense statements, this can reveal whether correct and incorrect commonsense judgments are linearly separable and at which layers this separation emerges.

---

## 5. Recommended Workflow

A step-by-step experimental workflow for investigating mechanistic causes of commonsense reasoning failures, organized into phases.

### Phase 1: Setup and Baseline Evaluation

1. **Load target model via TransformerLens.**
   Start with GPT-2 Small (124M parameters) for rapid iteration, then scale to Pythia-70M and Pythia-160M for cross-model comparison. Use `HookedTransformer.from_pretrained()` to load models with full activation access.

2. **Evaluate on PIQA (primary) and CommonsenseQA (secondary) to establish baseline performance.**
   Record per-example predictions, confidence scores, and correct/incorrect labels. Partition the dataset into: (a) consistently correct, (b) consistently incorrect, and (c) borderline examples. PIQA is recommended as the primary benchmark because its binary-choice format simplifies mechanistic analysis compared to five-choice CommonsenseQA.

### Phase 2: Coarse Localization

3. **Apply attention knockout (Geva et al. methodology) to identify critical information flow.**
   For a sample of correct and incorrect PIQA predictions, systematically knock out attention edges between positions and measure the effect on prediction accuracy. Identify which attention patterns are critical for commonsense reasoning and whether they differ between correct and failed predictions.

4. **Use vocabulary projection (logit lens) to examine intermediate representations.**
   For both correct and incorrect examples, project hidden states to vocabulary space at each layer. Track how the probability of the correct answer evolves across layers. Identify the layer(s) where correct and incorrect examples diverge -- this pinpoints the processing stage where reasoning fails.

5. **Run causal tracing (rome methodology) to localize commonsense knowledge storage.**
   Corrupt subject/goal embeddings and selectively restore layer activations. Determine which layers are critical for commonsense knowledge recovery. Compare localization patterns between physical commonsense (PIQA) and general commonsense (CommonsenseQA).

### Phase 3: Feature-Level Analysis

6. **Train SAEs on model activations during commonsense reasoning (using SAELens).**
   Train sparse autoencoders on residual stream activations at the critical layers identified in Phase 2. Use dictionary sizes recommended by Gao et al. (2024) scaling laws. Extract features that activate differentially during commonsense reasoning.

7. **Apply ReasonScore-like methodology (from SAE-Reasoning) to identify commonsense-specific features.**
   Compute a CommonsenseScore for each SAE feature by measuring differential activation between commonsense reasoning examples and control examples (e.g., factual recall, syntactic tasks). Rank features by commonsense specificity and manually inspect the top features for interpretability.

### Phase 4: Circuit Discovery

8. **Use ACDC or manual path patching to discover circuits for goal-action coherence.**
   Define a commonsense reasoning metric (e.g., logit difference between correct and incorrect PIQA solutions). Run ACDC to automatically discover the minimal circuit responsible for this metric. Alternatively, use manual path patching guided by the localization results from Phase 2.

9. **Compare circuit activation patterns between correct and failed predictions.**
   For the discovered commonsense circuit, compare activation magnitudes, attention patterns, and feature activations between examples the model gets right versus those it gets wrong. Identify specific circuit components (attention heads, MLP layers, SAE features) that behave differently in failure cases.

### Phase 5: Failure Diagnosis

10. **Test whether failures occur at enrichment, propagation, or extraction stage.**
    Following the three-stage factual recall framework from Geva et al. (2023), test at which stage commonsense reasoning breaks down:
    - **Enrichment failure:** The model fails to enrich the goal/subject representation with relevant commonsense knowledge (e.g., physical properties, typical outcomes). Diagnose via probing intermediate representations for commonsense attributes.
    - **Propagation failure:** Commonsense knowledge is present in early layers but fails to propagate to later processing stages. Diagnose via layer-by-layer logit lens and attention pattern analysis.
    - **Extraction failure:** Commonsense knowledge reaches the final layers but is not correctly extracted into the prediction. Diagnose via unembedding analysis and final-layer feature inspection.

### Phase 6: Validation and Extension

11. **Validate findings with targeted interventions.**
    Use activation steering (adding/subtracting identified commonsense feature directions) to test whether amplifying commonsense features in failure cases rescues correct predictions. Use ROME editing to test whether inserting missing commonsense associations repairs specific failures.

12. **Cross-benchmark generalization.**
    Test whether discovered circuits and features generalize from PIQA to HellaSwag (event commonsense) and WinoGrande (social/pragmatic commonsense). Characterize which circuit components are shared across commonsense domains versus domain-specific.

13. **Scale to larger models.**
    Replicate key findings in Pythia-160M, Pythia-410M, or GPT-2 Medium to assess whether failure mechanisms are consistent across model scales or whether larger models develop qualitatively different commonsense circuits.

---

## Appendix: Quick Reference

### File Structure

```
project-root/
├── papers/                          # PDF copies of all 26 papers
├── datasets/
│   ├── piqa/                        # PIQA dataset files
│   ├── commonsenseqa/               # CommonsenseQA dataset files
│   ├── hellaswag/                   # HellaSwag dataset files
│   ├── winogrande/                  # WinoGrande dataset files
│   ├── piqa_samples.json            # PIQA quick-inspection samples
│   ├── commonsenseqa_samples.json   # CommonsenseQA quick-inspection samples
│   ├── hellaswag_samples.json       # HellaSwag quick-inspection samples
│   └── winogrande_samples.json      # WinoGrande quick-inspection samples
├── code/
│   ├── TransformerLens/             # Mechanistic interpretability library
│   ├── SAELens/                     # SAE training and analysis
│   ├── SAE-Reasoning/               # Reasoning feature discovery
│   ├── ACDC/                        # Automated circuit discovery
│   └── rome/                        # Causal tracing and knowledge editing
├── results/                         # Experimental results and artifacts
├── artifacts/                       # Generated artifacts (figures, models)
└── resources.md                     # This file
```

### Paper-to-Method Mapping

| Method | Primary Paper(s) | Implementation |
|--------|-----------------|----------------|
| Path Patching | Wang et al. 2022 (#1) | TransformerLens |
| Activation Patching | Wang et al. 2022 (#1) | TransformerLens |
| Attention Knockout | Geva et al. 2023 (#11) | Custom (TransformerLens hooks) |
| ACDC | Conmy et al. 2023 (#12) | ACDC repo |
| SAE Training | Bricken et al. 2023 (#3), Gao et al. 2024 (#22) | SAELens |
| ReasonScore | Galichin et al. 2025 (#24) | SAE-Reasoning |
| Causal Tracing | Meng et al. 2022 (#2) | rome repo |
| ROME Editing | Meng et al. 2022 (#2) | rome repo |
| Logit Lens | nostalgebraist 2020 (via TransformerLens) | TransformerLens |
| Linear Probing | Hewitt & Manning 2019 (#17), Gurnee et al. 2023 (#10) | Custom (sklearn/PyTorch) |
| Representation Engineering | Zou et al. 2023 (#5) | Custom (TransformerLens hooks) |
| Sparse Feature Circuits | Marks et al. 2024 (#7) | SAELens + TransformerLens |

### Recommended Model Progression

| Stage | Model | Parameters | Rationale |
|-------|-------|-----------|-----------|
| Prototyping | GPT-2 Small | 124M | Best TransformerLens support, extensive prior interpretability work |
| Cross-validation | Pythia-70M | 70M | Smallest Pythia model, useful for sanity checks |
| Primary analysis | Pythia-160M | 160M | Good balance of capability and tractability |
| Scale testing | GPT-2 Medium / Pythia-410M | 345M / 410M | Tests whether findings generalize to larger models |
