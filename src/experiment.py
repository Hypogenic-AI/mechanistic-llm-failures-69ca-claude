"""
Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs

Main experiment script. Runs multiple complementary analyses on GPT-2 Small using PIQA:
1. Baseline evaluation + logit lens
2. Layer-wise causal tracing (mean ablation per layer)
3. Activation patching by component (attention heads + MLPs)
4. Layer-wise linear probing
5. Attention pattern analysis
6. Logit difference trajectory analysis
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import transformer_lens
from transformer_lens import HookedTransformer

# ---------- Configuration ----------
SEED = 42
MODEL_NAME = "gpt2"
DEVICE = "cuda:0"
N_EVAL = 500
N_TRACE = 80
N_PATCH = 80
N_PROBE = 300
NOISE_STD = 3.0
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_piqa():
    ds = load_from_disk("datasets/piqa")
    return ds["validation"]


def format_piqa_prompt(goal, solution):
    return f"Goal: {goal}\nSolution: {solution}"


def load_model():
    print(f"Loading {MODEL_NAME}...", flush=True)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"{model.cfg.d_model} dims", flush=True)
    return model


def compute_sequence_logprob(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[:-1].gather(1, tokens[1:].unsqueeze(1)).squeeze(1)
    return token_log_probs.mean().item()


# ================================================================
# Experiment 1: Baseline Evaluation + Logit Lens
# ================================================================
def evaluate_piqa(model, dataset, n_examples=N_EVAL):
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 1: Baseline Evaluation + Logit Lens", flush=True)
    print(f"{'='*60}", flush=True)

    results = []
    correct_count = 0
    n_layers = model.cfg.n_layers

    logit_lens_correct = []
    logit_lens_incorrect = []

    for i in tqdm(range(min(n_examples, len(dataset))), desc="Eval PIQA"):
        example = dataset[i]
        goal, sol1, sol2, label = example["goal"], example["sol1"], example["sol2"], example["label"]

        prompt1 = format_piqa_prompt(goal, sol1)
        prompt2 = format_piqa_prompt(goal, sol2)

        with torch.no_grad():
            logits1, cache1 = model.run_with_cache(prompt1)
            logits2, cache2 = model.run_with_cache(prompt2)

            tokens1 = model.to_tokens(prompt1)[0]
            tokens2 = model.to_tokens(prompt2)[0]

            lp1 = compute_sequence_logprob(logits1[0], tokens1)
            lp2 = compute_sequence_logprob(logits2[0], tokens2)

            predicted = 0 if lp1 > lp2 else 1
            is_correct = (predicted == label)
            if is_correct:
                correct_count += 1

            # Logit lens on prompt1 (use cache1)
            if len(tokens1) > 2:
                layer_probs = []
                for layer in range(n_layers):
                    resid = cache1[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                    layer_logits = model.unembed(model.ln_final(resid.unsqueeze(0)))
                    layer_prob = F.softmax(layer_logits[0], dim=-1)
                    target_token = tokens1[-1].item()
                    prob = layer_prob[target_token].item()
                    layer_probs.append(prob)

                if is_correct:
                    logit_lens_correct.append(layer_probs)
                else:
                    logit_lens_incorrect.append(layer_probs)

            results.append({
                "index": i, "goal": goal, "label": label,
                "predicted": predicted, "correct": is_correct,
                "logprob_sol1": lp1, "logprob_sol2": lp2,
                "logit_diff": abs(lp1 - lp2),
            })

        del cache1, cache2
        if i % 100 == 0:
            torch.cuda.empty_cache()

    accuracy = correct_count / len(results)
    print(f"\nBaseline Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})", flush=True)

    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "n_examples": len(results),
                   "n_correct": correct_count, "n_incorrect": len(results) - correct_count,
                   "results": results}, f, indent=2)

    logit_lens_data = {
        "correct_mean": np.mean(logit_lens_correct, axis=0).tolist() if logit_lens_correct else [],
        "correct_std": np.std(logit_lens_correct, axis=0).tolist() if logit_lens_correct else [],
        "incorrect_mean": np.mean(logit_lens_incorrect, axis=0).tolist() if logit_lens_incorrect else [],
        "incorrect_std": np.std(logit_lens_incorrect, axis=0).tolist() if logit_lens_incorrect else [],
        "n_correct": len(logit_lens_correct),
        "n_incorrect": len(logit_lens_incorrect),
    }
    with open(RESULTS_DIR / "logit_lens_data.json", "w") as f:
        json.dump(logit_lens_data, f, indent=2)

    plot_logit_lens(logit_lens_data, n_layers)
    return results, accuracy


def plot_logit_lens(data, n_layers):
    if not data["correct_mean"] or not data["incorrect_mean"]:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    cm, cs = np.array(data["correct_mean"]), np.array(data["correct_std"])
    im, ist = np.array(data["incorrect_mean"]), np.array(data["incorrect_std"])
    ax.plot(layers, cm, 'g-o', label=f'Correct (n={data["n_correct"]})', markersize=4, linewidth=2)
    ax.fill_between(layers, cm - cs, cm + cs, alpha=0.2, color='green')
    ax.plot(layers, im, 'r-s', label=f'Incorrect (n={data["n_incorrect"]})', markersize=4, linewidth=2)
    ax.fill_between(layers, im - ist, im + ist, alpha=0.2, color='red')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("P(last token)", fontsize=12)
    ax.set_title("Logit Lens: Token Probability Across Layers", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_lens_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved logit lens plot", flush=True)


# ================================================================
# Experiment 2: Layer-wise Causal Tracing (Mean Ablation)
# ================================================================
def run_causal_tracing(model, dataset, eval_results, n_examples=N_TRACE):
    """Efficient causal tracing: ablate each layer's residual stream and measure
    the impact on logit difference. No per-position loop."""
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 2: Layer-wise Causal Tracing", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_each = min(n_examples // 2, len(correct_indices), len(incorrect_indices))

    # For each layer, measure how much ablating it hurts the logit diff
    ablation_effect_correct = np.zeros(n_layers)
    ablation_effect_incorrect = np.zeros(n_layers)
    count_c, count_i = 0, 0

    for group_name, indices in [("correct", correct_indices[:n_each]),
                                 ("incorrect", incorrect_indices[:n_each])]:
        print(f"  Tracing {group_name}...", flush=True)
        for idx in tqdm(indices, desc=f"  Causal trace ({group_name})"):
            example = dataset[idx]
            label = example["label"]
            sol_correct = example["sol1"] if label == 0 else example["sol2"]
            sol_incorrect = example["sol2"] if label == 0 else example["sol1"]

            prompt_c = format_piqa_prompt(example["goal"], sol_correct)
            prompt_i = format_piqa_prompt(example["goal"], sol_incorrect)

            tokens_c = model.to_tokens(prompt_c)
            tokens_i = model.to_tokens(prompt_i)
            if tokens_c.shape[1] > 100 or tokens_i.shape[1] > 100:
                continue

            with torch.no_grad():
                # Clean logit difference
                logits_c = model(tokens_c)
                logits_i = model(tokens_i)
                clean_lp_c = F.log_softmax(logits_c[0, -1], dim=-1)[tokens_c[0, -1]].item()
                clean_lp_i = F.log_softmax(logits_i[0, -1], dim=-1)[tokens_i[0, -1]].item()
                clean_diff = clean_lp_c - clean_lp_i

                # For each layer: zero-ablate its residual stream contribution
                for layer in range(n_layers):
                    def ablate_hook(value, hook, l=layer):
                        # Replace this layer's output with zeros (mean ablation = zero for centered acts)
                        value[:, :, :] = 0.0
                        return value

                    ablated_logits_c = model.run_with_hooks(
                        tokens_c,
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", ablate_hook)]
                    )
                    ablated_logits_i = model.run_with_hooks(
                        tokens_i,
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", ablate_hook)]
                    )
                    abl_lp_c = F.log_softmax(ablated_logits_c[0, -1], dim=-1)[tokens_c[0, -1]].item()
                    abl_lp_i = F.log_softmax(ablated_logits_i[0, -1], dim=-1)[tokens_i[0, -1]].item()
                    ablated_diff = abl_lp_c - abl_lp_i

                    effect = clean_diff - ablated_diff  # positive = this layer helps correct prediction

                    if group_name == "correct":
                        ablation_effect_correct[layer] += effect
                    else:
                        ablation_effect_incorrect[layer] += effect

                if group_name == "correct":
                    count_c += 1
                else:
                    count_i += 1

                torch.cuda.empty_cache()

    if count_c > 0: ablation_effect_correct /= count_c
    if count_i > 0: ablation_effect_incorrect /= count_i

    trace_data = {
        "n_correct": count_c, "n_incorrect": count_i,
        "ablation_effect_correct": ablation_effect_correct.tolist(),
        "ablation_effect_incorrect": ablation_effect_incorrect.tolist(),
    }
    with open(RESULTS_DIR / "causal_trace_data.json", "w") as f:
        json.dump(trace_data, f, indent=2)

    plot_causal_traces(trace_data, n_layers)
    return trace_data


def plot_causal_traces(data, n_layers):
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    correct = np.array(data["ablation_effect_correct"])
    incorrect = np.array(data["ablation_effect_incorrect"])

    ax.bar(np.array(layers) - 0.15, correct, width=0.3, label=f"Correct (n={data['n_correct']})",
           color="green", alpha=0.7)
    ax.bar(np.array(layers) + 0.15, incorrect, width=0.3, label=f"Incorrect (n={data['n_incorrect']})",
           color="red", alpha=0.7)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Causal Effect on Logit Difference", fontsize=12)
    ax.set_title("Layer-wise Causal Importance for Commonsense Reasoning", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "causal_traces.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Difference plot
    fig, ax = plt.subplots(figsize=(10, 6))
    diff = correct - incorrect
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(layers, diff, color=colors, alpha=0.7)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Causal Effect Difference (Correct - Incorrect)", fontsize=12)
    ax.set_title("Where Does Commonsense Reasoning Succeed vs Fail?", fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "causal_trace_diff.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved causal trace plots", flush=True)


# ================================================================
# Experiment 3: Activation Patching (Attention Heads + MLPs)
# ================================================================
def run_activation_patching(model, dataset, eval_results, n_examples=N_PATCH):
    """Patch individual attention heads and MLPs to measure their causal importance."""
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 3: Activation Patching by Component", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_each = min(n_examples // 2, len(correct_indices), len(incorrect_indices))

    attn_imp_correct = np.zeros((n_layers, n_heads))
    attn_imp_incorrect = np.zeros((n_layers, n_heads))
    mlp_imp_correct = np.zeros(n_layers)
    mlp_imp_incorrect = np.zeros(n_layers)

    for group_name, indices, attn_imp, mlp_imp in [
        ("correct", correct_indices[:n_each], attn_imp_correct, mlp_imp_correct),
        ("incorrect", incorrect_indices[:n_each], attn_imp_incorrect, mlp_imp_incorrect)
    ]:
        count = 0
        print(f"  Patching {group_name}...", flush=True)
        for idx in tqdm(indices, desc=f"  Patching ({group_name})"):
            example = dataset[idx]
            label = example["label"]
            sol_correct = example["sol1"] if label == 0 else example["sol2"]
            prompt = format_piqa_prompt(example["goal"], sol_correct)
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 100:
                continue

            with torch.no_grad():
                # Clean run
                clean_logits, clean_cache = model.run_with_cache(prompt)
                clean_lp = F.log_softmax(clean_logits[0, -1], dim=-1)[tokens[0, -1]].item()

                # For each attention head: zero-ablate and measure effect
                for layer in range(n_layers):
                    for head in range(n_heads):
                        def ablate_head(value, hook, h=head):
                            value[:, :, h, :] = 0.0
                            return value

                        abl_logits = model.run_with_hooks(
                            tokens,
                            fwd_hooks=[(f"blocks.{layer}.attn.hook_result", ablate_head)]
                        )
                        abl_lp = F.log_softmax(abl_logits[0, -1], dim=-1)[tokens[0, -1]].item()
                        attn_imp[layer, head] += (clean_lp - abl_lp)

                    # MLP ablation
                    def ablate_mlp(value, hook):
                        value[:, :, :] = 0.0
                        return value

                    abl_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", ablate_mlp)]
                    )
                    abl_lp = F.log_softmax(abl_logits[0, -1], dim=-1)[tokens[0, -1]].item()
                    mlp_imp[layer] += (clean_lp - abl_lp)

                count += 1
                del clean_cache
                torch.cuda.empty_cache()

        if count > 0:
            attn_imp /= count
            mlp_imp /= count

    patch_data = {
        "attn_importance_correct": attn_imp_correct.tolist(),
        "attn_importance_incorrect": attn_imp_incorrect.tolist(),
        "mlp_importance_correct": mlp_imp_correct.tolist(),
        "mlp_importance_incorrect": mlp_imp_incorrect.tolist(),
    }
    with open(RESULTS_DIR / "activation_patching_data.json", "w") as f:
        json.dump(patch_data, f, indent=2)

    plot_activation_patching(patch_data, n_layers, n_heads)
    return patch_data


def plot_activation_patching(data, n_layers, n_heads):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ac = np.array(data["attn_importance_correct"])
    ai = np.array(data["attn_importance_incorrect"])
    mc = np.array(data["mlp_importance_correct"])
    mi = np.array(data["mlp_importance_incorrect"])

    vmax_a = max(abs(ac).max(), abs(ai).max(), 1e-8)
    im = axes[0, 0].imshow(ac, aspect='auto', cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a)
    axes[0, 0].set_title("Attn Head Importance (Correct)", fontsize=12)
    axes[0, 0].set_xlabel("Head"); axes[0, 0].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(ai, aspect='auto', cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a)
    axes[0, 1].set_title("Attn Head Importance (Incorrect)", fontsize=12)
    axes[0, 1].set_xlabel("Head"); axes[0, 1].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[0, 1])

    layers = list(range(n_layers))
    axes[1, 0].bar(np.array(layers) - 0.15, mc, width=0.3, label="Correct", color="green", alpha=0.7)
    axes[1, 0].bar(np.array(layers) + 0.15, mi, width=0.3, label="Incorrect", color="red", alpha=0.7)
    axes[1, 0].set_title("MLP Importance", fontsize=12)
    axes[1, 0].set_xlabel("Layer"); axes[1, 0].set_ylabel("Importance")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    diff = ac - ai
    vabs = max(abs(diff).max(), 1e-8)
    im = axes[1, 1].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    axes[1, 1].set_title("Attn Importance Diff (Correct-Incorrect)", fontsize=12)
    axes[1, 1].set_xlabel("Head"); axes[1, 1].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "activation_patching.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved activation patching plots", flush=True)


# ================================================================
# Experiment 4: Layer-wise Linear Probing
# ================================================================
def run_probing(model, dataset, eval_results, n_examples=N_PROBE):
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 4: Layer-wise Linear Probing", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_correct = min(n_examples // 2, len(correct_indices))
    n_incorrect = min(n_examples // 2, len(incorrect_indices))
    selected = correct_indices[:n_correct] + incorrect_indices[:n_incorrect]
    random.shuffle(selected)

    all_activations = {layer: [] for layer in range(n_layers)}
    all_labels = []
    model_correct_flags = []

    print(f"  Collecting activations for {len(selected)} examples...", flush=True)
    for idx in tqdm(selected, desc="  Probing"):
        example = dataset[idx]
        label = example["label"]
        is_model_correct = any(r["index"] == idx and r["correct"] for r in eval_results)

        goal_tokens = model.to_tokens(f"Goal: {example['goal']}")
        if goal_tokens.shape[1] > 100:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(goal_tokens)
            for layer in range(n_layers):
                act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                all_activations[layer].append(act)
            all_labels.append(label)
            model_correct_flags.append(is_model_correct)
            del cache
            if len(all_labels) % 100 == 0:
                torch.cuda.empty_cache()

    all_labels = np.array(all_labels)
    model_correct_flags = np.array(model_correct_flags)
    print(f"  Collected {len(all_labels)} examples "
          f"(correct={model_correct_flags.sum()}, incorrect={(~model_correct_flags).sum()})", flush=True)

    probe_results = {"overall": [], "model_correct_subset": [], "model_incorrect_subset": []}

    for layer in tqdm(range(n_layers), desc="  Training probes"):
        X = np.array(all_activations[layer])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        scores = cross_val_score(clf, X, all_labels, cv=5, scoring='accuracy')
        probe_results["overall"].append({
            "layer": layer, "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
        })

        for subset_name, mask in [("model_correct_subset", model_correct_flags),
                                   ("model_incorrect_subset", ~model_correct_flags)]:
            if mask.sum() >= 20 and len(np.unique(all_labels[mask])) > 1:
                X_sub, y_sub = X[mask], all_labels[mask]
                clf_sub = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                n_cv = min(5, min(np.bincount(y_sub)))
                if n_cv >= 2:
                    scores_sub = cross_val_score(clf_sub, X_sub, y_sub, cv=n_cv, scoring='accuracy')
                    probe_results[subset_name].append({
                        "layer": layer, "accuracy_mean": float(scores_sub.mean()),
                        "accuracy_std": float(scores_sub.std()),
                    })

    with open(RESULTS_DIR / "probing_results.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    plot_probing(probe_results, n_layers)
    return probe_results


def plot_probing(results, n_layers):
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))

    overall_acc = [r["accuracy_mean"] for r in results["overall"]]
    overall_std = [r["accuracy_std"] for r in results["overall"]]
    ax.plot(layers, overall_acc, 'b-o', label="All examples", linewidth=2, markersize=5)
    ax.fill_between(layers, np.array(overall_acc) - np.array(overall_std),
                    np.array(overall_acc) + np.array(overall_std), alpha=0.2, color='blue')

    if results["model_correct_subset"]:
        cl = [r["layer"] for r in results["model_correct_subset"]]
        ca = [r["accuracy_mean"] for r in results["model_correct_subset"]]
        cs = [r["accuracy_std"] for r in results["model_correct_subset"]]
        ax.plot(cl, ca, 'g-s', label="Model-correct", linewidth=2, markersize=5)
        ax.fill_between(cl, np.array(ca) - np.array(cs), np.array(ca) + np.array(cs), alpha=0.2, color='green')

    if results["model_incorrect_subset"]:
        il = [r["layer"] for r in results["model_incorrect_subset"]]
        ia = [r["accuracy_mean"] for r in results["model_incorrect_subset"]]
        ist = [r["accuracy_std"] for r in results["model_incorrect_subset"]]
        ax.plot(il, ia, 'r-^', label="Model-incorrect", linewidth=2, markersize=5)
        ax.fill_between(il, np.array(ia) - np.array(ist), np.array(ia) + np.array(ist), alpha=0.2, color='red')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Layer-wise Probing: Can a Probe Predict the Correct Answer from Goal?", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.35, 0.85)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probing_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved probing plot", flush=True)


# ================================================================
# Experiment 5: Attention Pattern Analysis
# ================================================================
def run_attention_analysis(model, dataset, eval_results, n_examples=60):
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 5: Attention Pattern Analysis", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_each = min(n_examples // 2, len(correct_indices), len(incorrect_indices))

    goal_attn_correct = np.zeros((n_layers, n_heads))
    goal_attn_incorrect = np.zeros((n_layers, n_heads))
    count_c, count_i = 0, 0

    for group_name, indices in [("correct", correct_indices[:n_each]),
                                 ("incorrect", incorrect_indices[:n_each])]:
        for idx in tqdm(indices, desc=f"  Attention ({group_name})"):
            example = dataset[idx]
            label = example["label"]
            sol = example["sol1"] if label == 0 else example["sol2"]
            prompt = format_piqa_prompt(example["goal"], sol)
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 100:
                continue

            goal_tok_len = model.to_tokens(f"Goal: {example['goal']}").shape[1]

            with torch.no_grad():
                _, cache = model.run_with_cache(prompt)
                for layer in range(n_layers):
                    attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [heads, seq, seq]
                    # Attention from last token to goal region
                    last_to_goal = attn[:, -1, :goal_tok_len].mean(dim=-1).cpu().numpy()
                    if group_name == "correct":
                        goal_attn_correct[layer] += last_to_goal
                    else:
                        goal_attn_incorrect[layer] += last_to_goal
                if group_name == "correct":
                    count_c += 1
                else:
                    count_i += 1
                del cache
                torch.cuda.empty_cache()

    if count_c > 0: goal_attn_correct /= count_c
    if count_i > 0: goal_attn_incorrect /= count_i

    attn_data = {"n_correct": count_c, "n_incorrect": count_i,
                 "goal_attn_correct": goal_attn_correct.tolist(),
                 "goal_attn_incorrect": goal_attn_incorrect.tolist()}
    with open(RESULTS_DIR / "attention_analysis_data.json", "w") as f:
        json.dump(attn_data, f, indent=2)

    plot_attention_analysis(attn_data, n_layers, n_heads)
    return attn_data


def plot_attention_analysis(data, n_layers, n_heads):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gc = np.array(data["goal_attn_correct"])
    gi = np.array(data["goal_attn_incorrect"])
    vmax = max(gc.max(), gi.max(), 1e-8)

    im = axes[0].imshow(gc, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[0].set_title("Last->Goal Attn (Correct)", fontsize=12)
    axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer"); plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(gi, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[1].set_title("Last->Goal Attn (Incorrect)", fontsize=12)
    axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer"); plt.colorbar(im, ax=axes[1])

    diff = gc - gi
    vabs = max(abs(diff).max(), 1e-8)
    im = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    axes[2].set_title("Attention Diff (Correct-Incorrect)", fontsize=12)
    axes[2].set_xlabel("Head"); axes[2].set_ylabel("Layer"); plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved attention pattern plots", flush=True)


# ================================================================
# Experiment 6: Logit Difference Trajectory Analysis
# ================================================================
def run_logit_diff_analysis(model, dataset, eval_results, n_examples=150):
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 6: Layer-wise Logit Difference Trajectory", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_each = min(n_examples // 2, len(correct_indices), len(incorrect_indices))

    logit_diffs_correct = []
    logit_diffs_incorrect = []

    for group_name, indices, diff_list in [
        ("correct", correct_indices[:n_each], logit_diffs_correct),
        ("incorrect", incorrect_indices[:n_each], logit_diffs_incorrect)
    ]:
        for idx in tqdm(indices, desc=f"  Logit diff ({group_name})"):
            example = dataset[idx]
            label = example["label"]
            sol_c = example["sol1"] if label == 0 else example["sol2"]
            sol_i = example["sol2"] if label == 0 else example["sol1"]

            prompt_c = format_piqa_prompt(example["goal"], sol_c)
            prompt_i = format_piqa_prompt(example["goal"], sol_i)
            tokens_c = model.to_tokens(prompt_c)
            tokens_i = model.to_tokens(prompt_i)
            if tokens_c.shape[1] > 100 or tokens_i.shape[1] > 100:
                continue

            with torch.no_grad():
                _, cache_c = model.run_with_cache(prompt_c)
                _, cache_i = model.run_with_cache(prompt_i)

                layer_diffs = []
                for layer in range(n_layers):
                    rc = cache_c[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                    ri = cache_i[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                    lc = model.unembed(model.ln_final(rc.unsqueeze(0)))[0]
                    li = model.unembed(model.ln_final(ri.unsqueeze(0)))[0]

                    # Mean logit over all tokens for correct vs incorrect prompt
                    diff = lc.mean().item() - li.mean().item()
                    layer_diffs.append(diff)

                diff_list.append(layer_diffs)
                del cache_c, cache_i
                torch.cuda.empty_cache()

    correct_mean = np.mean(logit_diffs_correct, axis=0) if logit_diffs_correct else np.zeros(n_layers)
    correct_std = np.std(logit_diffs_correct, axis=0) if logit_diffs_correct else np.zeros(n_layers)
    incorrect_mean = np.mean(logit_diffs_incorrect, axis=0) if logit_diffs_incorrect else np.zeros(n_layers)
    incorrect_std = np.std(logit_diffs_incorrect, axis=0) if logit_diffs_incorrect else np.zeros(n_layers)

    ld_data = {
        "correct_mean": correct_mean.tolist(), "correct_std": correct_std.tolist(),
        "incorrect_mean": incorrect_mean.tolist(), "incorrect_std": incorrect_std.tolist(),
        "n_correct": len(logit_diffs_correct), "n_incorrect": len(logit_diffs_incorrect),
    }
    with open(RESULTS_DIR / "logit_diff_analysis.json", "w") as f:
        json.dump(ld_data, f, indent=2)

    plot_logit_diff(ld_data, n_layers)
    return ld_data


def plot_logit_diff(data, n_layers):
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    cm, cs = np.array(data["correct_mean"]), np.array(data["correct_std"])
    im, ist = np.array(data["incorrect_mean"]), np.array(data["incorrect_std"])

    ax.plot(layers, cm, 'g-o', label=f'Model-correct (n={data["n_correct"]})', linewidth=2, markersize=5)
    ax.fill_between(layers, cm - cs, cm + cs, alpha=0.2, color='green')
    ax.plot(layers, im, 'r-s', label=f'Model-incorrect (n={data["n_incorrect"]})', linewidth=2, markersize=5)
    ax.fill_between(layers, im - ist, im + ist, alpha=0.2, color='red')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Logit Difference", fontsize=12)
    ax.set_title("Logit Difference Trajectory: Correct vs Incorrect", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_diff_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved logit diff plot", flush=True)


# ================================================================
# Experiment 7: Noise-based Causal Tracing (Meng et al. style)
# ================================================================
def run_noise_causal_tracing(model, dataset, eval_results, n_examples=40):
    """Meng et al. style causal tracing: corrupt embeddings, restore at each layer."""
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment 7: Noise-based Causal Tracing (Meng-style)", flush=True)
    print(f"{'='*60}", flush=True)

    n_layers = model.cfg.n_layers
    correct_indices = [r["index"] for r in eval_results if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_results if not r["correct"]]
    n_each = min(n_examples // 2, len(correct_indices), len(incorrect_indices))

    # Recovery at each layer for correct vs incorrect
    recovery_correct = np.zeros(n_layers)
    recovery_incorrect = np.zeros(n_layers)
    count_c, count_i = 0, 0

    for group_name, indices in [("correct", correct_indices[:n_each]),
                                 ("incorrect", incorrect_indices[:n_each])]:
        print(f"  Noise tracing {group_name}...", flush=True)
        for idx in tqdm(indices, desc=f"  Noise trace ({group_name})"):
            example = dataset[idx]
            label = example["label"]
            sol = example["sol1"] if label == 0 else example["sol2"]
            prompt = format_piqa_prompt(example["goal"], sol)
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 80:
                continue

            with torch.no_grad():
                # Clean run
                clean_logits, clean_cache = model.run_with_cache(tokens)
                clean_lp = F.log_softmax(clean_logits[0, -1], dim=-1)[tokens[0, -1]].item()

                # Fully corrupted run
                def corrupt_embed(value, hook):
                    noise = torch.randn_like(value) * NOISE_STD
                    return value + noise

                corrupt_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[("hook_embed", corrupt_embed)]
                )
                corrupt_lp = F.log_softmax(corrupt_logits[0, -1], dim=-1)[tokens[0, -1]].item()
                corruption_effect = clean_lp - corrupt_lp

                if abs(corruption_effect) < 0.01:
                    continue

                # For each layer: corrupt embed + restore this layer's residual
                for layer in range(n_layers):
                    def restore_layer(value, hook, l=layer):
                        value[:, :, :] = clean_cache[hook.name][:, :value.shape[1], :]
                        return value

                    restored_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[
                            ("hook_embed", corrupt_embed),
                            (f"blocks.{layer}.hook_resid_post", restore_layer),
                        ]
                    )
                    restored_lp = F.log_softmax(restored_logits[0, -1], dim=-1)[tokens[0, -1]].item()
                    # Recovery fraction
                    recovery = (restored_lp - corrupt_lp) / (corruption_effect + 1e-10)

                    if group_name == "correct":
                        recovery_correct[layer] += recovery
                    else:
                        recovery_incorrect[layer] += recovery

                if group_name == "correct":
                    count_c += 1
                else:
                    count_i += 1

                del clean_cache
                torch.cuda.empty_cache()

    if count_c > 0: recovery_correct /= count_c
    if count_i > 0: recovery_incorrect /= count_i

    noise_data = {
        "n_correct": count_c, "n_incorrect": count_i,
        "recovery_correct": recovery_correct.tolist(),
        "recovery_incorrect": recovery_incorrect.tolist(),
    }
    with open(RESULTS_DIR / "noise_causal_trace.json", "w") as f:
        json.dump(noise_data, f, indent=2)

    plot_noise_trace(noise_data, n_layers)
    return noise_data


def plot_noise_trace(data, n_layers):
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    rc = np.array(data["recovery_correct"])
    ri = np.array(data["recovery_incorrect"])

    ax.plot(layers, rc, 'g-o', label=f'Correct (n={data["n_correct"]})', linewidth=2, markersize=6)
    ax.plot(layers, ri, 'r-s', label=f'Incorrect (n={data["n_incorrect"]})', linewidth=2, markersize=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Recovery Fraction", fontsize=12)
    ax.set_title("Noise-based Causal Tracing:\nWhich Layers Store Commonsense Knowledge?", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "noise_causal_trace.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved noise causal trace plot", flush=True)


# ================================================================
# Main
# ================================================================
def main():
    start_time = time.time()
    print("="*60, flush=True)
    print("MECHANISTIC INTERPRETABILITY OF COMMONSENSE REASONING FAILURES", flush=True)
    print("="*60, flush=True)

    dataset = load_piqa()
    model = load_model()

    env_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model": MODEL_NAME, "seed": SEED,
    }
    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Experiment 1
    eval_results, accuracy = evaluate_piqa(model, dataset, N_EVAL)

    # Experiment 2
    trace_data = run_causal_tracing(model, dataset, eval_results, N_TRACE)

    # Experiment 3
    patch_data = run_activation_patching(model, dataset, eval_results, N_PATCH)

    # Experiment 4
    probe_data = run_probing(model, dataset, eval_results, N_PROBE)

    # Experiment 5
    attn_data = run_attention_analysis(model, dataset, eval_results, 60)

    # Experiment 6
    logit_diff_data = run_logit_diff_analysis(model, dataset, eval_results, 150)

    # Experiment 7
    noise_data = run_noise_causal_tracing(model, dataset, eval_results, 40)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)

    with open(RESULTS_DIR / "timing.json", "w") as f:
        json.dump({"total_seconds": elapsed, "total_minutes": elapsed/60}, f, indent=2)


if __name__ == "__main__":
    main()
