# Literature Review: Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs

**Research Hypothesis:** Commonsense reasoning failures in LLMs arise from specific breakdowns in goal-action coherence integration within the model's internal mechanisms. By identifying where and how these failures occur mechanistically, we can determine whether they are fundamentally architectural, representational, or training-induced.

---

## 1. Research Area Overview

The field of mechanistic interpretability seeks to reverse-engineer the internal computations of neural networks, moving beyond treating models as black boxes toward understanding the specific algorithms they implement. In parallel, commonsense reasoning has emerged as one of the most persistent and revealing failure modes of large language models (LLMs). Despite achieving human-level or near-human performance on many NLP benchmarks, LLMs continue to fail on tasks requiring everyday physical, social, and causal reasoning -- the kind of knowledge that humans acquire effortlessly through embodied experience and cultural participation. The intersection of these two fields represents a critical frontier: rather than merely documenting that LLMs fail at commonsense reasoning, mechanistic interpretability offers the tools to understand *why* they fail at the level of individual neurons, attention heads, and computational circuits.

Understanding the internal mechanisms behind commonsense failures is important for several reasons. First, it can disambiguate between fundamentally different failure hypotheses. A model might fail at commonsense reasoning because the relevant knowledge was never encoded during training (a training-induced failure), because the knowledge is encoded but stored in a representation that cannot be effectively retrieved or composed (a representational failure), or because the transformer architecture itself lacks the computational primitives needed for certain kinds of reasoning (an architectural failure). Each of these failure modes demands a different intervention -- more or better training data, improved training objectives, or architectural innovations -- and mechanistic interpretability is uniquely positioned to distinguish between them. Second, commonsense reasoning failures are particularly dangerous in deployed systems because they are unpredictable: a model that appears competent on complex tasks may fail catastrophically on simple physical or social reasoning. Understanding the mechanistic roots of these failures is therefore essential for building reliable and trustworthy AI systems.

The existing literature has made substantial progress on both mechanistic interpretability and commonsense reasoning independently, but has largely not connected the two. Circuit-level analyses have focused on syntactic tasks like indirect object identification, factual recall, and simple logical operations. Commonsense benchmarks have focused on behavioral evaluation -- measuring accuracy without probing the underlying mechanisms. This literature review surveys the key findings from both areas and identifies the specific gaps that our research aims to fill: applying the tools of mechanistic interpretability to understand the internal computations (and failures) behind commonsense reasoning, with a particular focus on goal-action coherence as tested by benchmarks like PIQA.

---

## 2. Key Papers and Findings

### 2.1 Circuit-Level Mechanistic Interpretability

#### Wang et al. (2022) -- "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small" (arXiv: 2211.00593)

This landmark paper represents the most comprehensive end-to-end reverse-engineering of a natural language behavior in a language model to date. Wang et al. identified a circuit of 26 attention heads grouped into 7 functional classes that together implement indirect object identification (IOI) in GPT-2 small. The IOI task involves sentences like "When Mary and John went to the store, John gave a drink to ___," where the model must predict "Mary" -- the indirect object that is not the repeated name. The authors discovered that the circuit implements a four-step algorithm: (1) **Duplicate Token Heads** detect all names in the input and identify which name appears more than once; (2) **Previous Token Heads** and **Induction Heads** work together to identify the duplicated name (the "S-inhibition" signal); (3) **S-Inhibition Heads** use this signal to inhibit attention to the duplicated (subject) name at the final token position; and (4) **Name Mover Heads** copy the remaining (indirect object) name to the output. The study introduced path patching as a technique for tracing information flow through specific edges (attention head to attention head connections) in the computational graph, providing causal rather than merely correlational evidence for the circuit's structure.

Critically, Wang et al. established three quantitative criteria for evaluating mechanistic explanations: **faithfulness** (the circuit's behavior matches the full model's behavior on the relevant task), **completeness** (the circuit accounts for all of the model's performance, meaning the complement of the circuit contributes negligibly), and **minimality** (no component of the circuit can be removed without degrading performance). The circuit achieved high faithfulness (recovering most of the logit difference between the correct and incorrect answers) and reasonable completeness, though some gaps remained -- suggesting that the model may use additional, as-yet-undiscovered mechanisms. This work is foundational for our research because it demonstrates that specific, interpretable circuits can be identified for natural language tasks, and because the IOI task shares structural similarities with commonsense reasoning tasks that require tracking entities and their relationships. The methodology of path patching and the faithfulness/completeness/minimality evaluation framework directly inform how we plan to study circuits for commonsense reasoning.

#### Conmy et al. (2023) -- "Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)" (arXiv: 2304.14004)

While Wang et al.'s IOI circuit discovery relied heavily on manual investigation and human intuition, Conmy et al. addressed the scalability challenge by developing ACDC (Automatic Circuit DisCovery), an automated method for identifying circuits in neural networks. ACDC works by iteratively testing whether each edge in the model's computational graph is important for a given task, using activation patching to measure the causal effect of corrupting each edge. The method starts from the full computational graph and prunes edges whose removal does not significantly affect the model's output on the task of interest. The authors validated ACDC by recovering known circuits (including the IOI circuit from Wang et al.) and discovering new circuits for tasks like greater-than comparison and docstring completion.

ACDC is particularly relevant to our research because manually discovering circuits for commonsense reasoning -- which likely involves more complex and distributed computations than IOI -- would be prohibitively labor-intensive. The automation provided by ACDC, or methods inspired by it, will be essential for scaling circuit discovery to the richer computational structures we expect to find in commonsense reasoning. However, ACDC has limitations: it assumes a clean separation between task-relevant and task-irrelevant edges, which may not hold for diffuse computations like commonsense reasoning, and its computational cost grows with model size. These limitations will need to be addressed in our experimental design.

### 2.2 Knowledge Storage and Retrieval Mechanisms

#### Dai et al. (2022) -- "Knowledge Neurons in Pretrained Transformers" (arXiv: 2104.08696)

Dai et al. introduced the concept of "knowledge neurons" -- individual neurons in the feed-forward network (FFN) layers of pretrained transformers that are responsible for expressing specific factual knowledge. Using a method based on integrated gradients (a gradient-based attribution technique), they identified neurons whose activation is causally linked to the model's ability to recall specific facts. Working primarily with BERT, they found that each factual association (e.g., "The capital of France is Paris") is stored in a small, sparse set of neurons, averaging 4.13 knowledge neurons per fact. The causal role of these neurons was validated through intervention experiments: suppressing (zeroing out) identified knowledge neurons caused an average 29% decrease in the probability of the correct factual answer, while amplifying their activations (doubling their values) caused an average 31% increase. Furthermore, the study found that knowledge neurons are concentrated in the upper layers of the FFN modules, consistent with the general finding that higher layers in transformers encode more abstract, semantic information.

This work is significant for our research because it establishes that factual knowledge has a sparse, localized representation in transformer FFN layers. A key question for our project is whether commonsense knowledge -- which is more diffuse, implicit, and relational than simple factual associations -- exhibits similar or different storage patterns. If commonsense knowledge is stored in analogous "commonsense neurons," interventions on these neurons could reveal why specific commonsense reasoning fails. If, on the other hand, commonsense knowledge is stored more distributedly, this would suggest a fundamentally different representational strategy that may be more vulnerable to certain failure modes.

#### Meng et al. (2022) -- "Locating and Editing Factual Associations in GPT (ROME)" (arXiv: 2202.05262)

Meng et al. developed a causal tracing methodology to localize where factual associations are stored in autoregressive language models, specifically GPT-2 and GPT-J. Their approach involves three steps: (1) corrupting the subject tokens in the input by adding noise to their embeddings, which degrades the model's ability to recall the associated fact; (2) restoring clean activations at specific layers and positions one at a time; and (3) measuring which restorations recover the model's ability to produce the correct fact. This "causal trace" revealed that factual associations are primarily mediated by mid-layer MLP modules at the position of the last token of the subject entity. Based on this localization, the authors developed ROME (Rank-One Model Editing), a technique for directly editing factual associations by modifying the weights of the identified MLP layers, treating them as key-value stores where the key is the subject representation and the value is the associated attribute.

ROME's causal tracing methodology is directly applicable to our research. We can use an analogous approach to localize where commonsense knowledge is processed: corrupt the goal description in a PIQA-style query, then systematically restore activations to identify which layers and positions are critical for goal-action coherence. If commonsense knowledge is localized similarly to factual associations (in mid-layer MLPs), this would suggest a shared storage mechanism; if it is localized differently (e.g., in attention heads or in different layers), this would reveal important differences between factual and commonsense knowledge representation. The ROME framework also provides a model for potential interventions: if we can identify where commonsense knowledge is stored, we may be able to edit or enhance it.

#### Geva et al. (2023) -- "Dissecting Recall of Factual Associations in Auto-Regressive Language Models" (arXiv: 2304.14767)

Building on the localization work of Meng et al., Geva et al. provided a much more detailed mechanistic account of how factual associations are retrieved during inference. Using careful interventions on attention edges in GPT-2 and LLaMA models, they identified a three-step internal mechanism for attribute extraction from factual queries (e.g., "The Eiffel Tower is located in ___"): **(A) Subject Enrichment:** In the early MLP layers, the representation at the last subject token position is enriched with many subject-related attributes. This enrichment process is driven by the early MLP sublayers and results in a representation that encodes not just the specific queried attribute but a broad set of facts about the subject. **(B) Relation Propagation:** Information about the relation (e.g., "is located in") propagates from the relation token positions to the final prediction position via attention mechanisms. This step effectively tells the model what kind of attribute to extract. **(C) Attribute Extraction:** At the prediction position, upper-layer attention heads "query" the enriched subject representation to extract the specific attribute that matches the relation. These attention heads often encode subject-attribute mappings directly in their parameters, functioning as "knowledge hub" heads that each encode hundreds of distinct factual associations.

This three-step framework -- enrichment, propagation, extraction -- is the most detailed mechanistic account of knowledge retrieval in LLMs and is central to our research design. We hypothesize that commonsense reasoning failures may occur at any of these three stages: the model may fail to adequately enrich the goal representation with relevant physical/causal knowledge (Stage A failure), the relation between goal and action may not propagate correctly (Stage B failure), or the extraction of the appropriate action may fail even when the enriched representation contains the necessary information (Stage C failure). By applying Geva et al.'s framework to PIQA-style goal-action queries, we can systematically test at which stage commonsense reasoning breaks down.

### 2.3 Sparse Autoencoders for Feature Discovery

#### Bricken et al. (2023) -- "Towards Monosemanticity" (arXiv: 2309.10312)

Bricken et al. addressed one of the fundamental challenges in mechanistic interpretability: the polysemanticity problem, where individual neurons respond to multiple, semantically unrelated concepts (a phenomenon linked to superposition, where models represent more features than they have dimensions). The authors trained sparse autoencoders (SAEs) on the activations of a one-layer transformer, decomposing the model's internal representations into a larger set of sparse, more interpretable features. Each SAE feature ideally corresponds to a single, coherent concept. The key insight is that while the model's neurons are polysemantic (each neuron participates in representing many different concepts simultaneously), the directions in activation space identified by the SAE are more monosemantic (each direction corresponds to a single interpretable concept). The authors demonstrated that SAE features are more interpretable than raw neurons, that they can be used to understand model behavior, and that intervening on individual features produces predictable effects on model outputs.

This foundational work is critical for our research because commonsense knowledge is likely represented in superposition -- distributed across many neurons rather than localized in individual ones. If we attempt to study commonsense representations using raw neurons, we will face severe polysemanticity problems. SAEs provide a principled way to decompose these distributed representations into interpretable features, potentially revealing "commonsense features" that are not visible at the neuron level. The methodology of training SAEs on intermediate activations and then analyzing the resulting features is a core component of our planned experimental pipeline.

#### Templeton et al. (2024) -- "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (arXiv: 2406.04093)

Templeton et al. scaled the sparse autoencoder approach from Bricken et al.'s one-layer transformer to Claude 3 Sonnet, a production-scale language model. This work demonstrated that SAEs can extract interpretable features even from very large models, identifying features corresponding to a remarkable range of concepts -- from specific entities (Golden Gate Bridge, Rosalind Franklin) to abstract concepts (code errors, deceptive behavior, gender bias). The authors found that features at different scales of the SAE dictionary correspond to different levels of abstraction, with larger dictionaries capturing more fine-grained concepts. Critically, the features are not just interpretable but causally meaningful: intervening on specific features (clamping them to high or low values) produces predictable changes in model behavior. For example, amplifying the "Golden Gate Bridge" feature caused the model to insert references to the bridge into unrelated conversations.

The scaling results from Templeton et al. validate SAEs as a viable approach for studying representations in models of practical interest, not just toy models. For our research, this work suggests that we should be able to find commonsense-relevant features in medium-scale models (like GPT-2 medium or Pythia models) using SAEs, and that these features will be causally meaningful -- allowing us to test whether amplifying commonsense features improves commonsense reasoning and whether suppressing them induces failures.

#### Galichin et al. (2025) -- "Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders" (arXiv: 2503.18878)

Galichin et al. applied SAEs specifically to study reasoning capabilities, training them on DeepSeek-R1-Llama-8B, a model fine-tuned for mathematical reasoning. They introduced the **ReasonScore** metric for identifying features specifically associated with reasoning behavior: features that activate significantly more during reasoning traces than during non-reasoning text. Using this metric, they identified 46 reasoning-specific features that cluster into three behavioral modes: **uncertainty** (features active when the model expresses doubt or considers alternatives), **exploration** (features active during hypothesis generation and search), and **reflection** (features active during self-evaluation and error-checking). Remarkably, steering the model by amplifying these reasoning features improved performance on the AIME mathematics benchmark by +13.4%, demonstrating that these features are causally involved in reasoning, not merely correlated with it.

A particularly important finding for our research is the "model diffing" analysis: the authors compared SAE features in the reasoning-fine-tuned model with those in the base (non-reasoning) model and found that reasoning features are largely absent in the base model, emerging specifically through reasoning-focused fine-tuning. This suggests that reasoning capabilities are not merely latent in pretrained models waiting to be elicited, but are actively created or amplified through training. For commonsense reasoning, this raises the critical question: do analogous "commonsense reasoning features" exist, and if so, are they present in base models (having been acquired from pretraining data) or do they require specific fine-tuning to emerge? The ReasonScore methodology can be directly adapted to search for commonsense-specific features, and the steering experiments provide a template for testing whether identified features are causally involved in commonsense reasoning.

### 2.4 Reasoning Limitations

#### Dziri et al. (2023) -- "Faith and Fate: Limits of Transformers on Compositionality" (arXiv: 2305.18354)

Dziri et al. provided a theoretical and empirical analysis of fundamental limitations of transformers on compositional reasoning tasks. They demonstrated that transformers approximate multi-step compositional tasks through **linearized subgraph matching** rather than performing true compositional reasoning. In this approximation, the model matches input patterns to similar patterns seen during training and interpolates between stored solutions, rather than applying compositional rules step-by-step. The critical consequence is that performance degrades systematically with **composition depth** -- the number of reasoning steps required. The authors tested this theory on three task families: multi-digit multiplication (where composition depth corresponds to the number of digits), logic puzzles (where composition depth corresponds to the number of inference steps), and dynamic programming problems (where composition depth corresponds to the number of subproblems).

Across all three task families, they found a consistent pattern: transformers achieve high accuracy at low composition depths but degrade sharply as depth increases, even when trained on examples at higher depths. This finding has profound implications for commonsense reasoning, which often requires composing multiple pieces of knowledge (e.g., understanding that a material is rigid, therefore it cannot bend, therefore it cannot be used as a flexible container). If commonsense reasoning failures are partly driven by composition depth effects, this would point to an architectural limitation rather than a knowledge gap. Our research can directly test this hypothesis by analyzing how the internal mechanisms of LLMs differ between commonsense tasks requiring shallow reasoning (single-step goal-action matching) and those requiring deeper composition (multi-step causal chains).

### 2.5 Commonsense Reasoning Benchmarks

#### Bisk et al. (2020) -- "PIQA: Reasoning about Physical Commonsense in Natural Language" (arXiv: 1911.11641)

Bisk et al. introduced PIQA (Physical Interaction: Question Answering), a benchmark specifically designed to test physical commonsense reasoning. PIQA uses a **goal-solution format**: each item consists of a goal (e.g., "To separate egg whites from the yolk using a water bottle") and two candidate solutions, one physically plausible and one implausible. This format is particularly significant for mechanistic interpretability because it explicitly separates the goal representation from the action representation, making it possible to study how the model integrates information about goals and actions -- what we term "goal-action coherence." The benchmark contains approximately 16,000 training examples and 2,000 validation examples. Human performance is 94.9%, while the best model at the time of publication (RoBERTa-large) achieved only 77.1%, revealing a substantial gap.

The authors' error analysis is particularly informative for our research. They found that model failures cluster around specific knowledge dimensions: failures on **spatial and relational words** (understanding relative positions, orientations, and physical relationships), failures on **versatile concepts** (words like "tape" or "paper" that have multiple physical uses depending on context), and failures requiring **causal reasoning** about physical interactions (understanding that heating wax makes it soft, or that rough surfaces create friction). These failure dimensions map well onto mechanistic hypotheses: spatial failures may reflect inadequate enrichment of spatial features at the subject representation stage, versatile-concept failures may reflect polysemanticity in the model's concept representations, and causal reasoning failures may reflect composition depth limitations. PIQA's goal-solution structure makes it the ideal primary benchmark for our research on goal-action coherence mechanisms.

#### Talmor et al. (2019) -- "CommonsenseQA" (arXiv: 1811.00937)

Talmor et al. created CommonsenseQA, a challenging multiple-choice question answering dataset that specifically targets commonsense knowledge. The dataset construction leverages ConceptNet, a commonsense knowledge graph: for each source concept, the authors extracted multiple target concepts sharing the same semantic relation, then asked crowdworkers to write questions that discriminate between these targets. This design ensures that questions require genuine commonsense reasoning rather than simple word association. The dataset contains 12,247 questions with 5 answer choices each, covering a wide range of commonsense knowledge types including spatial, temporal, causal, and social reasoning. At the time of publication, the best model (BERT-large) achieved only 56% accuracy compared to 89% human performance, though subsequent models have substantially narrowed this gap. CommonsenseQA's concept-based structure makes it a useful complement to PIQA for our research, as it tests more abstract commonsense knowledge.

#### Zellers et al. (2019) -- "HellaSwag" (arXiv: 1905.07830)

Zellers et al. introduced HellaSwag, a benchmark for evaluating commonsense natural language inference through sentence completion. Given a context describing an activity (drawn from WikiHow articles and ActivityNet captions), the model must select the most plausible continuation from four options. The dataset is notable for its use of **Adversarial Filtering (AF)**, a procedure that iteratively generates and filters machine-generated incorrect endings to ensure they are challenging for models while remaining easy for humans. The resulting dataset contains approximately 40,000 training examples and achieves a large human-machine gap: humans score above 95% while state-of-the-art models at publication scored below 50%. HellaSwag tests temporal and causal commonsense -- understanding what typically happens next in a sequence of actions -- making it relevant to our study of goal-action coherence, as the correct continuation must be coherent with the established goal or activity.

#### Sakaguchi et al. (2020) -- "WinoGrande" (arXiv: 1907.10641)

Sakaguchi et al. created WinoGrande, a large-scale dataset of 44,000 pronoun resolution problems inspired by the Winograd Schema Challenge (WSC). Each problem consists of a sentence with a blank that must be filled with one of two entity options, where the correct answer requires commonsense reasoning about the entities' properties and relationships. The key methodological contribution is AFLITE (Adversarial Filtering - Lite), an algorithm for reducing spurious statistical biases in the dataset by removing examples that can be solved through surface-level cues. The authors demonstrated that AFLITE significantly reduces the accuracy of models that rely on word associations, from over 90% on the original WSC to 59-79% on WinoGrande, confirming that high WSC performance was inflated by dataset artifacts. WinoGrande tests a different aspect of commonsense than PIQA -- understanding how properties and attributes of entities determine their roles in events -- and provides a complementary evaluation dimension for our research.

### 2.6 Additional Relevant Work

**Marks & Tegmark (2023) -- "The Geometry of Truth" (arXiv: 2310.06824):** This work demonstrated that LLMs develop linear representations of truth and falsehood in their internal activations. Using datasets of simple true/false statements, the authors showed that truth is linearly separable in the model's representation space, that probes trained on one truth dataset generalize to others, and that surgically intervening along the "truth direction" can flip the model's assessment of true and false statements. This is relevant to our research because commonsense reasoning failures could involve the model assigning incorrect truth values to physical propositions, and the geometric structure of commonsense truth representations may differ from factual truth representations.

**Zou et al. (2023) -- "Representation Engineering" (arXiv: 2310.01405):** Zou et al. introduced representation engineering (RepE), a framework for understanding and controlling LLM behavior by identifying and manipulating directions in the model's representation space that correspond to high-level concepts like honesty, fairness, and harmfulness. The approach uses contrastive pairs of stimuli to identify concept directions, then applies reading vectors (for monitoring) or control vectors (for steering) along these directions. RepE provides a complementary approach to SAEs for our research: while SAEs decompose representations into sparse features, RepE identifies specific directions associated with target concepts. We can use RepE-style contrastive analysis to identify directions associated with "physical plausibility" or "goal-action coherence."

**Arditi et al. (2024) -- "Refusal in Language Models Is Mediated by a Single Direction" (arXiv: 2406.11717):** Arditi et al. showed that the refusal behavior in safety-tuned chat models is mediated by a single direction in the model's residual stream. Erasing this direction prevents the model from refusing harmful requests, while adding it causes refusal on harmless requests. This finding is remarkable because it shows that complex behavioral patterns (refuse harmful requests, comply with benign ones) can be controlled by a single linear feature. For our research, this raises the question of whether commonsense reasoning competence might similarly be mediated by a small number of directions or features, and whether commonsense failures occur when these directions are insufficiently activated.

**Marks et al. (2024) -- "Sparse Feature Circuits" (arXiv: 2403.19647):** Marks et al. bridged the gap between circuit discovery and sparse autoencoders by introducing methods for discovering circuits defined over SAE features rather than raw model components. These "sparse feature circuits" consist of interpretable, monosemantic features connected by causal edges, providing a much more interpretable account of model computations than traditional circuits defined over polysemantic neurons and attention heads. The authors also introduced SHIFT, a method for improving classifier generalization by ablating features that humans judge to be task-irrelevant. This work is directly relevant to our research because it provides the methodology for discovering interpretable circuits for commonsense reasoning: we can train SAEs on our target models, then use sparse feature circuit discovery to identify the specific interpretable features and their causal connections that implement goal-action coherence.

---

## 3. Common Methodologies

### Activation Patching / Causal Tracing

Activation patching (also called causal tracing or interchange interventions) is the primary causal methodology in mechanistic interpretability. The core idea is to run the model on two inputs -- a "clean" input (where the model behaves correctly) and a "corrupted" input (where the model fails) -- and then selectively replace ("patch") activations from the clean run into the corrupted run at specific locations (layers, positions, components) to determine which activations are causally necessary for correct behavior.

**Mean ablation** is the simplest variant: replace a component's activation with its mean activation across a dataset, effectively removing that component's contribution to the specific input. If performance degrades, the component is important. **Path patching** (introduced by Wang et al., 2022) extends this by patching along specific computational paths (edges between components) rather than at individual nodes, enabling finer-grained causal analysis. For example, path patching can determine not just that attention head 9.6 is important, but specifically that information flowing from head 7.3 to head 9.6 is important. **Attention knockout** involves zeroing out specific attention weights to prevent information flow between positions, testing which position-to-position information transfer is necessary.

Causal tracing as developed by Meng et al. (2022) is a specific protocol: corrupt subject tokens by adding noise, then restore clean activations at individual (layer, position) sites, measuring which restorations recover the correct output. This produces a "causal trace" heatmap showing the causal importance of each site.

### Circuit Discovery

Circuit discovery aims to identify the minimal subnetwork (circuit) that implements a specific behavior. The general approach involves **backward tracing from outputs**: starting from the output logits, identify which components (attention heads, MLP layers) contribute most to the output, then trace backward to identify what inputs these components rely on, recursively building up the circuit. Manual circuit discovery (as in Wang et al., 2022) combines activation patching, attention pattern analysis, and hypothesis-driven investigation. **ACDC automation** (Conmy et al., 2023) replaces this manual process with systematic edge pruning: start with the full computational graph, test each edge via activation patching, and remove edges whose removal does not significantly affect the task-relevant output. The resulting pruned graph is the discovered circuit. Variants include node-based pruning (removing entire attention heads or MLP layers) and edge-based pruning (removing specific connections between components), with edge-based approaches providing finer granularity.

### Probing

Probing involves training a supervised classifier (the "probe") on a model's internal representations to predict a property of interest (e.g., part-of-speech, semantic role, factual truth). If a linear probe achieves high accuracy, this is taken as evidence that the property is linearly encoded in the representation. **Linear probes** are preferred because their limited capacity ensures that they are extracting information already present in the representation rather than learning the task themselves. **Control tasks** (Hewitt & Liang, 2019) provide a crucial methodological refinement: by defining control tasks where labels are randomly assigned to word types, the authors establish a baseline for what the probe's architecture alone can learn. A good probe should achieve high accuracy on the linguistic task but low accuracy on the control task -- a property called "selectivity." Probes that are not selective may be learning the task rather than reading it from the representation. For our research, probing will be used to test whether commonsense knowledge (e.g., physical properties of objects, causal relationships) is linearly represented in intermediate model layers, and whether the quality of these representations differs between correct and incorrect commonsense predictions.

### Sparse Autoencoders

Sparse autoencoders (SAEs) decompose a model's dense, polysemantic internal activations into a larger set of sparse, more monosemantic features. An SAE is trained to reconstruct the model's activations at a specific layer through a bottleneck that enforces sparsity: the encoder maps the activation to a high-dimensional sparse code, and the decoder maps back to the original dimension. The sparsity constraint encourages each dimension of the sparse code to correspond to a single interpretable concept. **Feature decomposition** involves analyzing the learned features by examining what inputs maximally activate each feature, producing feature-level interpretability. The **ReasonScore** metric (Galichin et al., 2025) provides a quantitative method for identifying features specifically associated with a target behavior (like reasoning): it compares the activation frequency of each feature during target behavior versus baseline behavior, identifying features that are significantly more active during the target behavior. Features with high ReasonScores are candidates for being causally involved in the behavior, which can be confirmed through steering experiments (amplifying or suppressing the feature and measuring the effect on behavior).

### Vocabulary Projection / Logit Lens

The logit lens technique (and its refinements like the tuned lens) projects intermediate model representations into vocabulary space by applying the model's unembedding matrix to hidden states at intermediate layers. This reveals what the model is "thinking" at each layer in terms of token probabilities. For a factual query like "The Eiffel Tower is in ___," the logit lens shows how the probability of "Paris" builds up across layers -- perhaps starting low in early layers and increasing as the model processes the input through successive layers. For our research, the logit lens can reveal at which layer the model begins to "commit" to a commonsense answer (correct or incorrect), providing a complementary view to activation patching about where commonsense reasoning occurs. Comparing logit lens trajectories for correct versus incorrect predictions can reveal whether failures involve the model never building the correct representation or initially building it but then overwriting it in later layers.

---

## 4. Standard Baselines and Evaluation Metrics

### Logit Difference for Circuit Analysis

The primary metric for evaluating circuit explanations is the **logit difference**: the difference in log-probability (or raw logit) between the correct and incorrect answers. For a binary choice task like PIQA, this is simply logit(correct_solution) - logit(incorrect_solution). A positive logit difference indicates that the model favors the correct answer, and the magnitude indicates the model's confidence. When evaluating circuits, the key measurement is how much of the full model's logit difference is recovered by the circuit alone. A circuit that recovers 90% of the logit difference is highly faithful. This metric is preferable to simple accuracy because it is continuous and sensitive to changes in model confidence, not just answer identity.

### Faithfulness, Completeness, and Minimality for Circuit Validation

Following Wang et al. (2022), circuit explanations should be evaluated on three criteria. **Faithfulness** measures whether the circuit, when isolated (with all other model components ablated), produces the same output distribution as the full model. High faithfulness means the circuit captures the relevant computation. **Completeness** measures whether the complement of the circuit (all components not in the circuit) contributes to the task. High completeness means there is no significant task-relevant computation happening outside the identified circuit. **Minimality** measures whether every component in the circuit is necessary -- no component can be removed without degrading the circuit's performance. A circuit that is faithful but not minimal may contain redundant components; a circuit that is minimal but not complete may miss important alternative pathways.

### Accuracy Metrics for Commonsense Benchmarks

Standard evaluation on commonsense benchmarks uses simple accuracy: the fraction of questions where the model selects the correct answer. For PIQA and WinoGrande (binary choice), chance performance is 50%. For CommonsenseQA (5-way multiple choice), chance is 20%. For HellaSwag (4-way), chance is 25%. Beyond aggregate accuracy, stratified accuracy across different knowledge dimensions (spatial, causal, temporal) and error analysis of specific failure modes are essential for connecting behavioral performance to mechanistic findings.

### ReasonScore for Reasoning Feature Identification

The **ReasonScore** (Galichin et al., 2025) quantifies the degree to which an SAE feature is specifically associated with reasoning behavior. It is computed as the difference in activation frequency between reasoning traces and non-reasoning text, normalized appropriately. Features with high ReasonScores are candidate "reasoning features." The metric can be adapted for commonsense reasoning by computing activation differences between commonsense-relevant inputs (e.g., PIQA questions) and generic text, identifying features that are specifically engaged during commonsense reasoning.

---

## 5. Datasets in the Literature

### PIQA (Physical Interaction: Question Answering)
- **Size:** ~16,000 training examples, ~2,000 validation, ~3,000 test (labels hidden)
- **Format:** Goal + two candidate solutions (binary choice)
- **Domain:** Physical commonsense reasoning
- **Key properties:** Goal-solution structure explicitly separates goal from action, enabling mechanistic study of goal-action coherence. Covers spatial reasoning, material properties, causal interactions, and everyday physical procedures.
- **Baselines:** Human 94.9%, GPT-2 (zero-shot) ~55%, RoBERTa-large 77.1%

### CommonsenseQA
- **Size:** ~10,000 training examples, ~1,200 validation, ~1,140 test
- **Format:** Question + 5 answer choices (multiple choice)
- **Domain:** General commonsense, constructed from ConceptNet relations
- **Key properties:** Concept-based design ensures questions require genuine commonsense reasoning beyond word association. Covers diverse knowledge types including spatial, temporal, causal, and social commonsense.
- **Baselines:** Human 89%, BERT-large 56% (at publication)

### HellaSwag
- **Size:** ~40,000 training examples, ~10,000 validation, ~10,000 test
- **Format:** Context + 4 candidate continuations (multiple choice)
- **Domain:** Temporal and causal commonsense (activity completion)
- **Key properties:** Adversarially filtered to be challenging for language models while easy for humans. Tests understanding of typical activity sequences and causal consequences.
- **Baselines:** Human 95.6%, BERT-large 47.3% (at publication)

### WinoGrande
- **Size:** ~40,000 training examples, ~1,200 validation, ~1,700 test
- **Format:** Sentence with blank + 2 candidate entity fillers (binary choice)
- **Domain:** Commonsense pronoun resolution / entity role understanding
- **Key properties:** AFLITE algorithm reduces spurious statistical biases. Tests understanding of how entity properties determine roles in events.
- **Baselines:** Human 94%, RoBERTa-large 79.1% (at publication)

### CounterFact
- **Size:** ~21,919 factual statements with counterfactual alternatives
- **Format:** Subject-relation-object triples with alternative objects
- **Domain:** Factual associations (used for knowledge editing evaluation)
- **Key properties:** Designed for evaluating factual editing methods like ROME. Includes paraphrase robustness tests and neighborhood consistency checks. Relevant to our research as a reference dataset for comparing factual versus commonsense knowledge mechanisms.

### PARAREL
- **Size:** 328 relation types with multiple natural language templates per relation
- **Format:** Cloze-style factual queries with multiple phrasings
- **Domain:** Factual knowledge probing across diverse relation types
- **Key properties:** Multi-template design enables testing whether knowledge representations are robust to surface-form variation. Useful for our research as a methodological template: we can construct multi-template versions of commonsense queries to test robustness of commonsense representations.

---

## 6. Gaps and Opportunities

The literature review reveals several significant gaps at the intersection of mechanistic interpretability and commonsense reasoning, each representing an opportunity for our research:

### Gap 1: No mechanistic interpretability work specifically targets commonsense reasoning failures

The circuit-level analyses to date (Wang et al., 2022; Conmy et al., 2023) have focused exclusively on syntactic and simple semantic tasks: indirect object identification, greater-than comparison, and docstring completion. Knowledge localization studies (Dai et al., 2022; Meng et al., 2022; Geva et al., 2023) have focused on simple factual recall (capital cities, birthdates, etc.). No published work has applied circuit discovery, activation patching, or causal tracing to understand how LLMs process commonsense reasoning tasks or why they fail on them. This is a foundational gap that our research directly addresses.

### Gap 2: Knowledge neuron and circuit approaches focus on factual recall, not reasoning

The knowledge neurons paradigm (Dai et al., 2022) and ROME (Meng et al., 2022) were developed for and evaluated on factual associations -- simple subject-relation-object triples. Commonsense reasoning requires not just retrieving stored facts but composing and applying them in context. Whether the same storage mechanisms (mid-layer FFN modules, knowledge neurons) are involved in commonsense reasoning, or whether commonsense engages qualitatively different mechanisms, remains unknown. Our research will test whether commonsense knowledge uses the same storage and retrieval mechanisms as factual knowledge.

### Gap 3: SAE-based reasoning features have only been studied for mathematical reasoning

Galichin et al. (2025) identified reasoning features using SAEs, but exclusively in the context of mathematical reasoning in a model fine-tuned for chain-of-thought mathematical problem solving. Whether analogous reasoning features exist for commonsense reasoning -- and whether they have similar or different characteristics (behavioral modes, layer distribution, emergence through training) -- is entirely unexplored. Commonsense reasoning is qualitatively different from mathematical reasoning: it relies on implicit, graded knowledge about the physical and social world rather than explicit, rule-based inference. Our research will apply the ReasonScore methodology to identify commonsense-specific features.

### Gap 4: The three-step retrieval mechanism has not been tested for commonsense knowledge

Geva et al.'s three-step mechanism (subject enrichment, relation propagation, attribute extraction) provides the most detailed account of knowledge retrieval in LLMs, but it was developed and validated exclusively for factual associations. Commonsense knowledge differs from factual knowledge in important ways: it is more implicit, more contextual, and more compositional. Testing whether commonsense retrieval follows the same three-step process, or uses a different mechanism, will reveal fundamental insights about how different types of knowledge are represented and processed.

### Gap 5: PIQA's goal-solution structure is ideal for studying goal-action coherence mechanistically

Among commonsense benchmarks, PIQA is uniquely suited for mechanistic study because its goal-solution format explicitly separates two components that the model must integrate: the goal (what needs to be accomplished) and the solution (how to accomplish it). This separation maps directly onto the subject-relation framework used in factual recall studies: the goal functions analogously to the subject, the implicit relation is "can be accomplished by," and the solution functions as the attribute. This structural parallel means we can adapt existing mechanistic interpretability tools (causal tracing, three-step analysis, circuit discovery) with minimal modification.

### Gap 6: Composition depth effects need to be studied in commonsense contexts

Dziri et al. (2023) demonstrated that transformers struggle with compositional tasks and that performance degrades with composition depth, but their analysis used synthetic tasks (multiplication, logic puzzles, dynamic programming). Whether the same composition-depth degradation applies to commonsense reasoning -- which often requires composing physical properties, causal chains, and contextual constraints -- has not been tested. Our research can stratify PIQA examples by the number of implicit reasoning steps required and test whether mechanistic failure patterns differ across composition depths.

---

## 7. Recommendations for Our Experiment

Based on the literature reviewed above, we make the following specific recommendations for our experimental design:

### Recommendation 1: Start with GPT-2 Small or Pythia Models

GPT-2 small (124M parameters) and the Pythia model family (70M to 6.9B parameters) are the best-supported models in the TransformerLens library, which provides the hooks, utilities, and activation caching infrastructure necessary for mechanistic interpretability work. GPT-2 small is the model on which the IOI circuit was discovered (Wang et al., 2022), providing a known baseline for circuit discovery methodology. The Pythia family offers the advantage of multiple model scales with identical architectures and training data, enabling the study of how commonsense reasoning mechanisms change with scale. We recommend starting with GPT-2 small for methodology development and then scaling to Pythia-1.4B or Pythia-2.8B for more realistic commonsense reasoning analysis.

### Recommendation 2: Use PIQA as Primary Benchmark

PIQA should be the primary benchmark for our research for three reasons: (a) its goal-solution format provides explicit separation of the two components whose integration we wish to study; (b) the error analysis from Bisk et al. (2020) provides qualitative categories of failure (spatial, versatile concepts, causal) that can guide mechanistic investigation; and (c) its binary-choice format maps cleanly to the logit-difference metric standard in circuit analysis. We recommend using CommonsenseQA and WinoGrande as secondary benchmarks to test whether findings generalize beyond physical commonsense.

### Recommendation 3: Apply the Three-Step Retrieval Framework to Commonsense Queries

Geva et al.'s three-step framework (enrichment, propagation, extraction) should be our primary analytical lens. For PIQA queries, this translates to: (A) **Goal Enrichment** -- does the model enrich the goal representation with relevant physical/causal knowledge in early MLP layers? (B) **Coherence Propagation** -- does information about the goal-action relationship propagate to the prediction position? (C) **Solution Extraction** -- do upper attention heads correctly extract the coherent solution from the enriched representations? We should compare these processes between correctly and incorrectly answered questions to identify at which stage failures occur.

### Recommendation 4: Use SAEs to Find Commonsense-Specific Features

Train sparse autoencoders on intermediate activations of our target model(s) when processing PIQA inputs. Adapt the ReasonScore methodology from Galichin et al. (2025) to identify features that are specifically active during commonsense reasoning: compute activation frequencies on PIQA inputs versus generic text, and identify features with high differential activation. Cluster identified features to determine whether commonsense reasoning involves distinct behavioral modes analogous to the uncertainty/exploration/reflection modes found for mathematical reasoning. Perform steering experiments to test causal involvement: amplifying commonsense features should improve PIQA accuracy, and suppressing them should degrade it.

### Recommendation 5: Apply Circuit Discovery to Identify Goal-Action Coherence Circuits

Use ACDC or path patching to discover the circuit implementing goal-action coherence in PIQA tasks. Start with path patching (more controlled, allows hypothesis-driven investigation) and validate with ACDC (more automated, reduces experimenter bias). Evaluate discovered circuits using the faithfulness/completeness/minimality criteria from Wang et al. (2022). Compare the discovered circuit to the IOI circuit (same model, different task) to determine how much circuit overlap exists between syntactic and commonsense tasks.

### Recommendation 6: Compare Mechanisms for Correct vs. Incorrect Predictions

The core of our research design should be a systematic comparison of internal mechanisms between correct and incorrect commonsense predictions. For each analytical tool (causal tracing, probing, SAE features, logit lens), we should compare: (a) the causal trace heatmap for correctly versus incorrectly answered questions; (b) probe accuracy at each layer for physical property knowledge in correct versus incorrect cases; (c) SAE feature activations for correct versus incorrect cases; and (d) logit lens trajectories showing how the probability of the correct answer evolves across layers in correct versus incorrect cases. Systematic differences will reveal the mechanistic origin of failures.

### Recommendation 7: Test Whether Failures Are at the Enrichment, Propagation, or Extraction Stage

This is the critical experiment for testing our hypothesis. Using the three-step framework from Recommendation 3, we should: (a) Probe for goal-relevant knowledge at the last-goal-token position after early MLP layers (testing enrichment); (b) Measure attention from prediction position to goal position in middle layers (testing propagation); (c) Analyze upper attention head behavior for solution extraction (testing extraction). If failures are predominantly at stage (A), this suggests a training-induced problem (the model never learned the relevant physical knowledge). If failures are at stage (B), this suggests a representational problem (knowledge exists but cannot be routed to where it is needed). If failures are at stage (C), this suggests an architectural or extraction problem (information arrives at the right place but the extraction mechanism fails). These distinctions are precisely what our hypothesis aims to resolve.

---

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *arXiv:2406.11717*.
- Bisk, Y., Zellers, R., Le Bras, R., Gao, J., & Choi, Y. (2020). PIQA: Reasoning about Physical Commonsense in Natural Language. *AAAI 2020*. *arXiv:1911.11641*.
- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*. *arXiv:2309.10312*.
- Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*. *arXiv:2304.14004*.
- Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers. *ACL 2022*. *arXiv:2104.08696*.
- Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., ... & Choi, Y. (2023). Faith and Fate: Limits of Transformers on Compositionality. *NeurIPS 2023*. *arXiv:2305.18354*.
- Galichin, N., Kovalev, N., Ershov, E., Goncharov, A., & Panov, M. (2025). Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders. *arXiv:2503.18878*.
- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., ... & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. *arXiv:2406.04093*.
- Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. *EMNLP 2023*. *arXiv:2304.14767*.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP 2019*. *arXiv:1909.03368*.
- Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. *arXiv:2310.06824*.
- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. *ICLR 2025*. *arXiv:2403.19647*.
- Meng, K., Bau, D., Mitchell, A., & Yun, C. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. *arXiv:2202.05262*.
- Sakaguchi, K., Le Bras, R., Bhagavatula, C., & Choi, Y. (2020). WinoGrande: An Adversarial Winograd Schema Challenge at Scale. *AAAI 2020*. *arXiv:1907.10641*.
- Talmor, A., Herzig, J., Lourie, N., & Berant, J. (2019). CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. *NAACL 2019*. *arXiv:1811.00937*.
- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., ... & Olah, C. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*. *arXiv:2406.04093*.
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. *ICLR 2023*. *arXiv:2211.00593*.
- Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? *ACL 2019*. *arXiv:1905.07830*.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.
