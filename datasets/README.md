# Datasets

Datasets collected for the research project: **Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs**

## Datasets Overview

### PIQA (Physical Interaction: Question Answering)
- **Source:** `datasets.load_dataset("piqa", revision="refs/convert/parquet")`
- **Paper:** Bisk et al. (2020), arXiv:1911.11641
- **Size:** 16,113 train / 1,838 validation / 3,084 test
- **Format:** Goal + 2 solutions + label (0 or 1)
- **Tests:** Physical commonsense reasoning (object affordances, spatial relations, material properties)
- **Local path:** `datasets/piqa/`
- **Primary benchmark** for this project due to goal-solution structure matching goal-action coherence hypothesis

### CommonsenseQA
- **Source:** `datasets.load_dataset("tau/commonsense_qa")`
- **Paper:** Talmor et al. (2019), arXiv:1811.00937
- **Size:** 9,741 train / 1,221 validation / 1,140 test
- **Format:** Question + 5 multiple-choice answers + answer key
- **Tests:** General commonsense knowledge using ConceptNet relations
- **Local path:** `datasets/commonsenseqa/`

### HellaSwag
- **Source:** `datasets.load_dataset("hellaswag")`
- **Paper:** Zellers et al. (2019), arXiv:1905.07830
- **Size:** 39,905 train / 10,042 validation / 10,003 test
- **Format:** Context (activity label + context) + 4 candidate endings + label
- **Tests:** Commonsense sentence completion / physical situation understanding
- **Local path:** `datasets/hellaswag/`

### WinoGrande
- **Source:** `datasets.load_dataset("allenai/winogrande", name="winogrande_xl")`
- **Paper:** Sakaguchi et al. (2020), arXiv:1907.10641
- **Size:** 40,398 train / 1,267 validation / 1,767 test
- **Format:** Sentence with blank + 2 options + answer (1 or 2)
- **Tests:** Commonsense pronoun resolution (Winograd schema)
- **Local path:** `datasets/winogrande/`

## Download Instructions

All datasets were downloaded via the HuggingFace `datasets` library. To re-download:

```python
from datasets import load_dataset

# PIQA
piqa = load_dataset("piqa", revision="refs/convert/parquet")

# CommonsenseQA
csqa = load_dataset("tau/commonsense_qa")

# HellaSwag
hellaswag = load_dataset("hellaswag")

# WinoGrande
winogrande = load_dataset("allenai/winogrande", name="winogrande_xl")
```

## Sample Files

Each dataset has a `*_samples.json` file containing 5 example entries for quick inspection:
- `piqa_samples.json`
- `commonsenseqa_samples.json`
- `hellaswag_samples.json`
- `winogrande_samples.json`

## Note on .gitignore

Large data files are excluded from version control. Only README files, sample JSON files, and download scripts are tracked. See `.gitignore` for details.
