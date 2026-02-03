# RoBERTa-OTA: Multiclass Hate Speech Detection with Ontology-Guided Transformer Attention

Official implementation of **RoBERTa-OTA** for 5-class hate speech detection, accepted at CSCI'25.

**Paper:** Multiclass Hate Speech Detection with RoBERTa-OTA: Integrating Transformer Attention and Graph Convolutional Networks  
**Conference:** [12th Annual Conference on Computational Science & Computational Intelligence (CSCI'25)](https://www.american-cse.org/csci2025/)

## Authors
- **Mahmoud Abusaqer** - Missouri State University
- **Jamil Saquer** - Missouri State University

## Abstract
This repository contains the implementation of RoBERTa-OTA, which combines RoBERTa's robust language understanding with ontology-guided graph neural networks to achieve state-of-the-art performance in multiclass hate speech detection across five demographic categories: age, ethnicity, gender, religion, and other cyberbullying. Our method integrates pre-trained RoBERTa embeddings with ontology-based knowledge graphs and enhanced graph convolutional networks to capture nuanced patterns in fine-grained demographic targeting.

## Key Features
- 🎯 **Enhanced 3-Layer GCN**: Processes structured demographic relationships through systematic progression
- 🔥 **Hybrid Architecture**: Combines RoBERTa embeddings with Graph Neural Networks (GNN)
- 📊 **High Performance**: Achieves 96.06% F1-weighted score (±0.19%) on 5-fold cross-validation
- 🛠️ **Complete Reproducibility**: Deterministic training with comprehensive seed control

## Repository Structure
```
RoBERTa-OTA/
├── .venv/                   # Virtual environment (library root)
├── RoBERTa-OTA.py           # Main RoBERTa-OTA implementation
├── RoBERTa_baseline.py      # RoBERTa baseline
├── .gitignore
├── README.md
├── requirements.txt
└── [dataset files]          # Place dataset here (see Dataset section)
```

## Dataset

Download the Cyberbullying Classification dataset from Kaggle:

**Dataset Link:** [Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

After downloading, place the dataset file in the **root directory** of the repository (same location as `README.md` and `requirements.txt`):
```
RoBERTa-OTA/
├── RoBERTa-OTA.py
├── RoBERTa_baseline.py
├── .gitignore
├── README.md
├── requirements.txt
└── cyberbullying_tweets.csv       # Place dataset file here
```

### Dataset Information
- **Total Samples**: 47,692 tweets across 6 categories
- **Preprocessed Samples**: 39,747 tweets across 5 hate speech categories (after removing "not_cyberbullying" class)
- **Classes**: Age (20.1%), Ethnicity (20.0%), Gender (20.0%), Religion (20.1%), Other Cyberbullying (19.8%)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/roberta-ota-multiclass-hate-speech.git
cd roberta-ota-multiclass-hate-speech
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) and place it in the **root directory** (same location as `README.md`).

## Usage

### Training RoBERTa-OTA
```bash
python RoBERTa-OTA.py
```

### Training RoBERTa Baseline
```bash
python RoBERTa_baseline.py
```

## Model Architecture

RoBERTa-OTA integrates:
1. **Pre-trained RoBERTa** for contextual word embeddings
2. **Ontology-based Knowledge Graphs** for domain knowledge (5-class cyberbullying ontology)
3. **Enhanced 3-Layer Graph Convolutional Networks (GCN)** for relationship modeling
4. **Deep 6-Layer Classifier** with advanced regularization for fine-grained classification

### Architecture Components
- **RoBERTa-Base**: 124.6M parameters
- **Enhanced GCN**: 3-layer graph network (6→64→64→32 dimensions)
- **Deep Classifier**: 6-layer network (800→400→200→5) with batch normalization and progressive dropout
- **Total Parameters**: 125,056,981 (~477.1 MB)

## Citation

If you use this code or our work in your research, please cite our paper:

```bibtex
@inproceedings{abusaqer2025roberta-ota,
  title={Multiclass Hate Speech Detection with RoBERTa-OTA: Integrating Transformer Attention and Graph Convolutional Networks},
  author={Abusaqer, Mahmoud and Saquer, Jamil},
  booktitle={12th Annual Conference on Computational Science \& Computational Intelligence (CSCI)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**⚠️ Citation Requirement:** If you use this code in your research, you **must** cite our paper (see above).

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems or have questions, please open an issue on GitHub.

## Contact

For questions or collaborations, please contact:
- Mahmoud Abusaqer - [LinkedIn](https://www.linkedin.com/in/mahmoud-abusaqer/)

---

**Accepted at CSCI'25** | [12th Annual Conference on Computational Science & Computational Intelligence](https://www.american-cse.org/csci2025/)