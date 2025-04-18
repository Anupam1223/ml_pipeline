# 🔧 Modular & Scalable ML Pipeline

A clean, extensible, and fully modular machine learning pipeline built with **PyTorch**, **Scikit-learn**, **Optuna**, and modern Python tooling. Designed for fast experimentation, reproducibility, and ease of deployment, this framework supports both classical and deep learning workflows.

---

## 🚀 Highlights

- **Config-driven workflow** with YAML-based experiment setup
- **Unified model wrapper** supporting both Scikit-learn & PyTorch models
- **End-to-end pipeline**: data parsing → training → evaluation → visualization
- **Built-in cross-validation**, metric tracking, and custom analysis modules
- **Hyperparameter optimization** using Optuna
- **PyTorch utility layer** with reusable components (losses, metrics, schedulers, etc.)
- **Designed with Software Engineering principles** (SOLID, DRY, modularity)

---

## Tech Stack

| Tool        | Purpose                           |
|-------------|-----------------------------------|
| PyTorch     | Deep learning models              |
| Scikit-learn| Classical ML algorithms           |
| Optuna      | Hyperparameter optimization       |
| Pandas      | Data manipulation                 |
| NumPy       | Numerical operations              |
| Matplotlib  | Visualization                     |
| YAML/JSON   | Configuration management          |

---

## Project Structure

```bash
📦 your-project-root/
├── cfg/                      # Configuration files (YAML, JSON)
│
├── core/                     # Core utilities or central logic (optional/expandable)
│
├── drivers/                  # Entry-point scripts to run training/pipeline
│   └── driver.py             # Main launcher script
│
├── input_data/               # Input data loaders or CSV/ROOT parser logic
│
├── model_wrapper/            # Wrappers to interface between Sklearn, PyTorch, etc.
│
├── models/                   # Custom model architectures and definitions
│
├── performance_metrics/      # Metric computation (accuracy, precision, etc.)
│
├── pipeline/                 # Training, validation, and testing orchestration
│
├── post_training_analysis/   # Confusion matrix, ROC, and performance plots
│
├── utils/                    # General-purpose tools, custom PyTorch helpers
│
├── visualization_module/     # Visualization logic (loss curves, metrics, etc.)
│
├── .env                      # Environment variables (user-defined config)
├── .gitignore                # Git ignored files
├── data.csv                  # Sample or test dataset
└── README.md                 # Project documentation



## To run
setup the .env file
'''
ANUPAM_DIR=
INPUT_DIR=
OUTPUT_DIR=
TRAINING_FILE=

FEATURES=feature names in comma, e.g. trk_e_fom,trk_m_fom 
'''
python drivers/driver.py