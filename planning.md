# Research Plan: Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Large language models fail at commonsense reasoning tasks that humans find trivial -- recommending driving to a location five minutes away on foot, or suggesting physically implausible solutions to everyday problems. These failures are dangerous in deployed systems because they are unpredictable: a model may demonstrate sophisticated reasoning on complex tasks while failing on simple physical intuitions. Understanding the *mechanistic* origin of these failures -- not just cataloguing them -- is essential for determining whether fixes require more data, better training objectives, or architectural changes.

### Gap in Existing Work
Based on the literature review, there is a clear gap at the intersection of mechanistic interpretability and commonsense reasoning:
1. **No mechanistic interpretability work targets commonsense reasoning.** Circuit discovery has focused on syntactic tasks (IOI in GPT-2), knowledge localization on simple factual recall (capital cities). Nobody has applied causal tracing, circuit discovery, or SAE feature analysis to commonsense tasks.
2. **The three-step retrieval mechanism (enrichment → propagation → extraction) from Geva et al. has never been tested for commonsense knowledge.** Factual and commonsense knowledge may use fundamentally different mechanisms.
3. **SAE-based reasoning features have only been studied for math.** Whether commonsense-specific features exist and whether they behave differently from math-reasoning features is unknown.
4. **PIQA's goal-solution structure is ideal but unexploited** for mechanistic study of goal-action coherence.

### Our Novel Contribution
We apply mechanistic interpretability tools -- specifically causal tracing, logit lens analysis, layer-wise probing, and activation patching -- to understand *where* in the transformer network commonsense reasoning breaks down. We compare internal mechanisms between correct and incorrect PIQA predictions in GPT-2 Small to diagnose whether failures arise from:
- **Enrichment failures** (relevant commonsense knowledge never activated)
- **Propagation failures** (knowledge activated but not routed to decision point)
- **Extraction failures** (knowledge arrives but wrong answer extracted)

This is the first study to apply the Geva et al. three-stage retrieval framework to commonsense reasoning.

### Experiment Justification
- **Experiment 1 (Baseline Evaluation + Logit Lens):** Needed to establish which PIQA examples GPT-2 gets right vs wrong, and to track where in the layer stack the correct answer probability diverges. This identifies the *layers* where failure occurs.
- **Experiment 2 (Causal Tracing):** Needed to localize where commonsense knowledge is stored -- in which layers and at which token positions. Direct comparison with factual knowledge localization (mid-layer MLPs) tests whether commonsense uses the same storage mechanism.
- **Experiment 3 (Activation Patching by Component):** Needed to identify which specific components (attention heads, MLP layers) are causally responsible for commonsense reasoning. Comparing component importance between correct and incorrect predictions reveals which components fail.
- **Experiment 4 (Layer-wise Probing):** Needed to test whether commonsense-relevant features (physical properties, causal relations) are linearly represented at each layer, and whether probe accuracy differs between examples the model gets right vs wrong. This distinguishes representational from extraction failures.

---

## Research Question
Where in the transformer architecture do commonsense reasoning failures originate, and are these failures primarily due to missing knowledge representation (enrichment), failed information routing (propagation), or incorrect answer extraction?

## Background and Motivation
LLMs fail at commonsense tasks not because they lack relevant facts, but because they fail to integrate goal-action coherence. PIQA provides an ideal testbed: its goal-solution format explicitly separates goals from candidate actions, enabling mechanistic study of how models integrate these components. Existing work on mechanistic interpretability has focused on syntactic and factual tasks, leaving commonsense reasoning unexplored.

## Hypothesis Decomposition

**H1 (Enrichment Hypothesis):** Commonsense failures occur because the model fails to enrich goal representations with relevant physical/causal knowledge in early-to-mid MLP layers.
- Testable via: Probing early/mid layers for commonsense properties; causal tracing showing critical layers.

**H2 (Propagation Hypothesis):** The model encodes commonsense knowledge but fails to propagate it from goal tokens to the prediction position.
- Testable via: Attention pattern analysis; logit lens showing knowledge present at goal positions but absent at prediction position.

**H3 (Extraction Hypothesis):** Commonsense knowledge reaches the prediction position but the extraction mechanism selects the wrong answer.
- Testable via: Logit lens showing correct answer briefly favored then overwritten; activation patching of upper-layer attention heads.

**H4 (Distributed Failure):** Failures are not localized to a single stage but involve degraded processing across multiple stages.
- Testable via: Comparing causal trace profiles between correct and incorrect predictions.

## Proposed Methodology

### Approach
Use GPT-2 Small (124M parameters) via TransformerLens as our primary model. GPT-2 Small is the best-understood model in the mechanistic interpretability literature, with known circuits (IOI) as calibration benchmarks. We use PIQA as the primary benchmark, processing examples in a prompt format that produces a binary choice. We apply four complementary mechanistic analysis methods.

### Experimental Steps

1. **Baseline Evaluation** (~15 min)
   - Load GPT-2 Small via TransformerLens
   - Evaluate on PIQA validation set using log-probability scoring
   - Partition into correct/incorrect predictions
   - Compute baseline accuracy and confidence distributions

2. **Logit Lens Analysis** (~20 min)
   - For correct and incorrect PIQA examples, compute logit lens at every layer
   - Track probability of correct vs incorrect solution across layers
   - Identify divergence layers (where correct examples start favoring the right answer and incorrect ones don't)
   - Visualize layer-by-layer probability trajectories

3. **Causal Tracing** (~30 min)
   - Corrupt goal token embeddings with Gaussian noise
   - Restore clean activations at each (layer, position) pair
   - Measure recovery of correct-answer logit difference
   - Produce causal trace heatmaps for correct vs incorrect predictions
   - Compare with known factual knowledge localization patterns

4. **Activation Patching by Component** (~30 min)
   - For each attention head and MLP layer: patch from corrupted to clean run
   - Measure effect on logit difference for correct answer
   - Identify most important attention heads and MLP layers
   - Compare importance rankings between correct and incorrect predictions

5. **Layer-wise Linear Probing** (~20 min)
   - Extract residual stream activations at each layer for PIQA examples
   - Train linear probes to predict correct solution from goal representation
   - Measure probe accuracy at each layer for correct vs incorrect model predictions
   - Determine at which layer the model "knows" the answer and whether this differs for failures

### Baselines
- Random baseline (50% for PIQA binary choice)
- GPT-2 Small zero-shot performance (expected ~55-60%)
- Known factual knowledge causal trace patterns from Meng et al. as reference

### Evaluation Metrics
- **Logit Difference:** logit(correct_sol) - logit(incorrect_sol) -- primary metric for all mechanistic analyses
- **Accuracy:** fraction of correctly answered PIQA questions
- **Causal Importance:** change in logit difference when restoring/patching each component
- **Probe Accuracy:** linear probe classification accuracy at each layer
- **Divergence Layer:** first layer where correct/incorrect predictions statistically separate

### Statistical Analysis Plan
- Bootstrap confidence intervals (n=1000) for accuracy and logit differences
- Paired t-tests comparing causal importance between correct/incorrect sets
- Effect sizes (Cohen's d) for correct vs incorrect differences at each layer
- Significance level: α = 0.05 with Bonferroni correction for layer-wise comparisons

## Expected Outcomes
- **If H1 (Enrichment):** Causal tracing will show failures concentrated in early/mid MLP layers; probes will show lower accuracy in early layers for incorrect predictions.
- **If H2 (Propagation):** Logit lens will show correct answer knowledge at goal token positions but not at prediction position; attention patterns will show weaker goal→prediction attention in failures.
- **If H3 (Extraction):** Logit lens will show correct answer briefly favored then overwritten in upper layers; activation patching will show critical upper attention heads.
- **If H4 (Distributed):** All stages will show moderate degradation with no single dominant failure point.

## Timeline and Milestones
1. Environment setup & data loading: 10 min
2. Baseline evaluation + logit lens: 30 min
3. Causal tracing experiments: 30 min
4. Activation patching: 30 min
5. Linear probing: 20 min
6. Analysis & visualization: 30 min
7. REPORT.md writing: 30 min
**Total: ~3 hours**

## Potential Challenges
- GPT-2 Small may perform near-chance on PIQA, limiting the signal for correct/incorrect comparison → Mitigation: even near-chance performance produces enough correct/incorrect examples for comparison
- Commonsense reasoning may be too distributed for clean localization → Mitigation: we treat this as a finding (H4) rather than a failure
- PIQA prompt format for autoregressive models needs careful design → Mitigation: use multiple prompt templates and select best-performing one

## Success Criteria
1. Clear causal trace heatmaps showing where commonsense knowledge is stored
2. Statistical comparison of internal mechanisms between correct and incorrect predictions
3. Evidence supporting or refuting each of H1-H4
4. At least one novel finding about commonsense failure mechanisms
