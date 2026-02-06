# Beyond Standardized Benchmarks: Orthographic Variation Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accepted at AbjadNLP 2026](https://img.shields.io/badge/AbjadNLP-2026--Oral-green)](https://wp.lancs.ac.uk/abjad/)

This repository contains the official implementation for the paper **"Beyond Standardized Benchmarks: Quantifying LLM Degradation Under Realistic Orthographic Variation in Abjad Languages"** — accepted to AbjadNLP 2026 as an *oral* presentation (≈20% acceptance rate). Conference: https://wp.lancs.ac.uk/abjad/

## 📄 Paper Abstract

Large language models (LLMs) perform well on multilingual natural language inference benchmarks, but real-world low-resource text contains orthographic variation absent from curated evaluations. We present the first systematic study of LLM robustness to artificially generated orthographic variation across Arabic, Urdu, Swahili, and English using modified XNLI benchmarks. Evaluating Llama 3.3 70B, Llama 3.1 8B, Qwen 2.5 32B, and GPT-OSS models across 80 language-condition pairs, we observe substantial performance degradation, with accuracy drops of up to 41% under romanization and up to 61% under code-switching. Smaller models fail catastrophically, with Llama 8B achieving 13% accuracy on fully romanized Urdu. Error analysis identifies label bias, out-of-vocabulary issues, and script asymmetries as key failure modes.

## 🎯 Key Findings

- **Orthographic fragility is severe**: Accuracy drops 8-24% under romanization and 15-41% under code-switching
- **Model size matters**: 70B model degrades gracefully, while 8B model collapses catastrophically
- **Universal limitation**: Even high-resource English degrades 41% under code-switching
- **Arabic shows robustness**: Diacritics removal causes minimal degradation (~2%)

## 📊 Evaluated Models

- **Llama 3.3 70B Versatile** (Meta)
- **Llama 3.1 8B Instant** (Meta)
- **Qwen 2.5 32B Instruct** (Alibaba)
- **GPT-OSS 20B** (OpenAI-style)
- **GPT-OSS 120B MoE** (Mixture-of-Experts)

## 🌍 Languages & Conditions

### Arabic (ar)
- **Clean**: Original XNLI text
- **No diacritics**: All vowel diacritics removed
- **Partial diacritics**: Only final-position case markers retained

### Urdu (ur)
- **Clean**: Original Perso-Arabic script
- **R25/R50/R100**: Romanization at 25%, 50%, 100% word-level
- **M25/M50**: Mixed-script code-switching with English at 25%, 50%

### English (en)
- **Clean**: Original Latin script
- **M25/M50**: Reverse code-switching with Urdu tokens at 25%, 50%

### Swahili (sw)
- **Clean**: Original Latin script
- **Romanized**: Maintained Latin script (control)
- **M25/M50**: Code-switching with English at 25%, 50%

**Total**: 16 language-condition pairs × 5 models = 80 model-language-condition combinations

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Groq API key(s) for model inference

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Soumedhik/abjad-orthographic-variation-benchmark.git
   cd abjad-orthographic-variation-benchmark
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env    # Linux/Mac
   ```
   
   Edit `.env` and set your Groq API keys:
   ```
   GROQ_API_KEYS=your_key_1,your_key_2,your_key_3
   ```

### Dataset Preparation

The benchmark expects XNLI data in CSV format. Each file should be named `{language}_{split}.csv` and contain columns: `premise`, `hypothesis`, `label`.

```
data/
├── ar_test.csv
├── ur_test.csv
├── en_test.csv
└── sw_test.csv
```

You can download XNLI from the [official repository](https://github.com/facebookresearch/XNLI) or [Hugging Face](https://huggingface.co/datasets/xnli).

## 📦 Usage

### Run Full Benchmark

```bash
python scripts/run_benchmark.py \
    --dataset-dir ./data \
    --results-dir ./results \
    --max-examples 40
```

### Compute Performance Deltas

After running the benchmark, calculate degradation metrics:

```bash
python scripts/compute_deltas.py \
    --benchmark ./results/benchmark.csv \
    --out-dir ./results
```

### Configuration Options

All settings can be configured via CLI flags or environment variables (`.env` file):

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `--dataset-dir` | `DATASET_DIR` | `../input/xnli-multilingual-nli-dataset` | Path to XNLI CSV files |
| `--results-dir` | `RESULTS_DIR` | `./results` | Output directory |
| `--eval-split` | `EVAL_SPLIT` | `test` | Dataset split (train/validation/test) |
| `--languages` | `LANGUAGES` | `ar,ur,en,sw` | Comma-separated language codes |
| `--max-examples` | `MAX_EXAMPLES_PER_CONDITION` | `40` | Examples per condition |
| `--requests-per-minute` | `REQUESTS_PER_MINUTE` | `60` | API rate limit |
| `--write-traces` | `WRITE_TRACES` | `0` | Write per-example traces (0/1) |

## 📂 Project Structure

```
.
├── scripts/
│   ├── run_benchmark.py       # Main evaluation script
│   └── compute_deltas.py      # Calculate performance degradation
├── src/
│   └── orthographic_nli/
│       ├── __init__.py        # Package initialization
│       ├── config.py          # Configuration management
│       ├── data.py            # XNLI data loading
│       ├── variants.py        # Orthographic variant generation
│       ├── groq_client.py     # Groq API interface
│       ├── evaluate.py        # Model evaluation logic
│       ├── metrics.py         # Performance metrics
│       └── traces.py          # Detailed trace logging
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🔬 Technical Details

### Orthographic Transformations

1. **Diacritics Manipulation** (Arabic):
   - Strip all Unicode combining marks (U+064B-U+0652)
   - Retain only final-position case markers

2. **Romanization** (Urdu):
   - Character-level transliteration using Urdu romanization conventions
   - Dose-response design: 25%, 50%, 100% word-level application

3. **Code-Switching** (All languages):
   - Random token replacement from donor language
   - Maintains semantic plausibility by sampling from parallel XNLI splits

### Evaluation Metrics

- **Strict Exact-Match Accuracy**: Normalized prediction must match gold label exactly
- **Macro F1 Score**: Averaged across entailment, neutral, and contradiction classes
- **Confusion Matrices**: Per model-language-condition for error analysis

## 📈 Results

Key results from the paper (summary across all models):

| Perturbation Type | Avg. Accuracy Drop |
|-------------------|-------------------|
| Arabic Diacritic Removal | -2.4% |
| Urdu Romanization (Full) | -28.6% |
| Code-Switching (50% Mix) | -41.2% |

See the paper for detailed results, confusion matrices, and error analysis.

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{mandal2026orthographic,
  title={Beyond Standardized Benchmarks: Quantifying LLM Degradation Under Realistic Orthographic Variation in Abjad Languages},
  author={Mandal, Shibam and Bharati, Soumedhik and Ghosh, Swarup Kr and Mondal, Sayani},
  journal={arXiv preprint},
  year={2026},
  institution={Sister Nivedita University}
}
```

## 👥 Authors

- **Shibam Mandal** - Sister Nivedita University - [shibammandal603@gmail.com](mailto:shibammandal603@gmail.com)
- **Soumedhik Bharati** - Sister Nivedita University - [soumedhikbharati@gmail.com](mailto:soumedhikbharati@gmail.com)
- **Swarup Kr Ghosh** - Sister Nivedita University - [swarupg1@gmail.com](mailto:swarupg1@gmail.com)
- **Sayani Mondal** - Sister Nivedita University - [sayani.mondal9@gmail.com](mailto:sayani.mondal9@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- XNLI dataset creators for providing multilingual NLI benchmarks
- Groq for providing API access to LLMs
- Meta, Alibaba, and OpenAI for open-source model releases

## 🔗 Links

- **Paper (AbjadNLP 2026)**: https://wp.lancs.ac.uk/abjad/ (accepted — oral; ≈20% acceptance)
- **Dataset**: [XNLI](https://github.com/facebookresearch/XNLI)
- **Models**: [Groq Cloud](https://groq.com/)

## ⚠️ Limitations

- Evaluation limited to 40 examples per condition due to API costs
- Programmatic orthographic variants approximate but don't perfectly match natural variation
- Results specific to Groq API inference; may vary with self-hosted deployments
- Focus on NLI; generative task robustness may differ

## 🐛 Issues & Contributions

Found a bug or have a suggestion? Please [open an issue](https://github.com/Soumedhik/abjad-orthographic-variation-benchmark/issues).

Contributions are welcome! Please ensure code follows existing style and includes appropriate tests/documentation.

---

**Made with ❤️ for multilingual NLP research**
