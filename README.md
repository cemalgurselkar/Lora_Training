# Qwen2.5-Coder LoRA Fine-Tuning Project

This repository contains the resources, configuration, and results for fine-tuning the **Qwen2.5-Coder-1.5B-Instruct** model using **Low-Rank Adaptation (LoRA)**. The project aims to enhance the model's Python coding capabilities by training it on two distinct datasets: **DEEP** (Logic-intensive) and **DIVERSE** (Variety-focused).

## Repository Structure

The project is organized as follows:

```text
LoRA-CodeGen-Project/
│
├── configs/                     # Configuration files (JSON)
│   └── final_config.json
│
├── images/                      # Training analysis graphs
│   ├── loss_graph_deep.png
│   └── loss_graph_diverse.png
│
├── logs/                        # Raw training logs
│   ├── training_logs_DEEP.txt
│   └── training_logs_DIVERSE.txt
│
├── train.py                     # Training script with argument parsing
├── eval.py                      # Evaluation and benchmarking script
├── requirements.txt             # Project dependencies
└── README.md                    # Project report and results
```
# Benchmark Result(LiveCodeBench/AtCoder)
## Base Model (Qwen2.5-Coder-1.5B-Instructor)
* **With DEEP Dataset: 11/41**
* **With DIVERSE Dataset: 11/41**

| Model | Best Checkpoint | Pass@1 Score | Total Question (41) |
| :--- | :--- | :--- | :--- 
| **Deep_instruction** | **step-800** | **26.8%** | **11/41** |
| **Diverse_instruction** | **step-600** | **36.6%** | **15/41** |

# Training Configuration

The models were trained using the following key hyperparameters:

    Base Model: Qwen/Qwen2.5-Coder-1.5B-Instruct

    LoRA Rank (r): 32

    LoRA Alpha: 64

    Target Modules: q_proj, k_proj, v_proj, o_proj

    DoRA: Enabled

    NEFTune Noise Alpha: 5

    Batch Size: 2 (Gradient Accumulation: 8)

    Learning Rate: 1e-4

## Installation and Usage
To replicate this project, follow the steps below.

### 1.Enviroment Setup
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/LoRA-CodeGen-Project.git](https://github.com/YOUR_USERNAME/LoRA-CodeGen-Project.git)
cd LoRA-CodeGen-Project
pip install -r requirements.txt
```

### Training
You can train the model on either dataset using the train.py script. The script supports command-line arguments to select the dataset.

* Train with DEEP dataset:
```bash
python train.py --dataset DEEP
```

* Train with DIVERSE dataset:
```bash
python train.py --dataset DIVERSE
```

#### Optional Arguments:
* use_drive: Enable Google Drive saving (default).
* output_dir: Specify a local directory for outputs if Drive is not used.
* epochs: Override the default number of epochs.

### Evaluation
To run the benchmarks and reproduce the results, use the eval.py script. This script automatically handles folder structures and runs the LiveCodeBench evaluation.

* Evaluate DEEP model:
```bash
python eval.py --model_type deep_instruction
```

* Evalute DIVERSE model:
```bash
python eval.py --model_type diverse_instruction
```