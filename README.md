# A Twin-Aware Multimodal Deep Learning Framework with Optimized Late Fusion

[![Paper](https://img.shields.io/badge/Paper-PLOS%20ONE-blue)](https://journals.plos.org/plosone/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

> **A twin-aware multimodal deep learning framework with optimized late fusion for early prediction of adolescent anxiety disorder**

## 📋 Abstract

Mental health problems in adolescents are frequently under-identified because current evaluation methods fail to integrate biological, behavioral, and demographic information simultaneously. We propose a **twin-aware multimodal deep learning framework** applied to the Queensland Twin Adolescent Brain (QTAB) dataset for early prediction of incident adolescent anxiety disorder.

### Key Highlights
- 🧠 **Multimodal Integration**: Combines MRI neuroimaging, behavioral questionnaires, and phenotypic data
- 👥 **Twin-Aware Evaluation**: Prevents genetic information leakage through family-level data splitting
- 🎯 **High Performance**: AUC of 0.8935 (95% CI: 0.792–0.969) with 85.7% sensitivity and 87.3% specificity
- 🔍 **Interpretable**: SHAP analysis for feature importance and clinical insights
- ⚖️ **Class-Imbalance Robust**: Prototype-based learning for handling imbalanced data

---

## 🏗️ Framework Architecture

![Framework Overview](Fig1_Overall_Framework.png)

Our framework consists of three independently trained modules integrated through optimized weighted late fusion:

### Module 1: MRI-Based Neuroimaging Analysis
- 3D CNN with residual blocks for structural brain pattern extraction
- Processes T1-weighted MRI scans
- Captures neuroanatomical vulnerability markers

### Module 2: Questionnaire-Based Behavioral Modeling
- Prototypical network with self-attention mechanism
- Processes SCAS, SMFQ, and SDQ questionnaires
- Leakage-aware feature selection

### Module 3: Phenotypic Feature Learning
- Demographic, socioeconomic, and genetic attributes
- Twin zygosity and family-level variables
- Captures environmental and genetic influences

### Fusion Strategy
- **Calibrated Late Fusion** with optimized weights:
  - 63% Questionnaire (dominant predictive role)
  - 23% MRI (neurobiological markers)
  - 14% Phenotypic (demographic context)

---

## 📊 Results

### Performance Metrics (Test Set, n=62)
| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-ROC | 0.8935 | 0.792–0.969 |
| Sensitivity | 85.7% | - |
| Specificity | 87.3% | - |
| F1-Score | 0.60 | 0.417–0.824 |

### Ablation Study
| Model Configuration | AUC-ROC |
|---------------------|---------|
| MRI only | 0.745 |
| Questionnaire only | 0.777 |
| Phenotypic only | 0.680 |
| MRI + Questionnaire | 0.800 |
| MRI + Phenotypic | 0.760 |
| Questionnaire + Phenotypic | 0.790 |
| **All Three (Ours)** | **0.894** |

**Synergistic Effect**: Three-way fusion shows disproportionate gain (+0.094) compared to pairwise combinations (+0.013–0.023), demonstrating complementary information across modalities.

---

## 📁 Repository Structure

```
.
├── CODE/                                          # Source code
│   ├── advanced_anxiety_questionnaire_prediction.ipynb   # Module 2: Questionnaire
│   ├── advanced_static_prediction.ipynb                  # Module 3: Phenotypic
│   ├── cnn4_q1_final.ipynb                               # Module 1: MRI/CNN
│   ├── run_multimodal_final.ipynb                        # Multimodal fusion
│   ├── shap_analysis.py                                  # SHAP interpretability
│   └── Standalone_MRI_Processing.ipynb                   # MRI preprocessing
│
├── figures/                                        # Publication figures
│   ├── Fig1_Overall_Framework.png
│   ├── Fig2_MRI_Preprocessing.png
│   ├── Fig3_Questionnaire_Pipeline.png
│   ├── Fig4_Prototypical_Network.png
│   ├── Fig5_Confusion_Matrix.png
│   ├── Fig6_Ablation_Study.png
│   ├── Fig7_Bootstrap_CI.png
│   ├── Fig9_Module2_SHAP_Summary.png
│   ├── Fig10_Module2_Feature_Importance.png
│   ├── Fig11_Module3_SHAP_Summary.png
│   └── Fig12_Module3_Feature_Importance.png
│
├── manuscript/                                     # LaTeX source
│   ├── plos_withoutH.tex                          # Clean manuscript
│   ├── plos_latex_HIGHLIGHT.tex                   # Tracked changes version
│   ├── plos_bibtex_sample.bib                     # References
│   └── plos2025.bst                               # Bibliography style
│
├── README.md                                       # This file
├── LICENSE                                         # MIT License
└── requirements.txt                                # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 1.9
CUDA >= 11.0 (for GPU support)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/taosiff/A-twin-aware-multimodal-deep-learning-framework-with-optimized-late-fusion.git
cd A-twin-aware-multimodal-deep-learning-framework-with-optimized-late-fusion
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download QTAB dataset** (access required)
- Request access to Queensland Twin Adolescent Brain (QTAB) dataset
- Follow data use agreement terms

### Usage

#### 1. Train Individual Modules

**Module 1: MRI-Based Prediction**
```bash
jupyter notebook CODE/cnn4_q1_final.ipynb
```

**Module 2: Questionnaire-Based Prediction**
```bash
jupyter notebook CODE/advanced_anxiety_questionnaire_prediction.ipynb
```

**Module 3: Phenotypic Prediction**
```bash
jupyter notebook CODE/advanced_static_prediction.ipynb
```

#### 2. Multimodal Fusion
```bash
jupyter notebook CODE/run_multimodal_final.ipynb
```

#### 3. Interpretability Analysis
```bash
python CODE/shap_analysis.py
```

---

## 🔬 Methodology

### Twin-Aware Data Splitting
- **Family-level splitting** prevents co-twin genetic information leakage
- Training: 60% of families
- Validation: 20% of families
- Test: 20% of families
- Ensures no twin pairs span across splits

### Leakage-Aware Preprocessing
- Removes aggregate scores from questionnaires
- Correlation-based feature selection on training data only
- Prevents information leakage from validation/test sets

### Class-Imbalance Handling
- Prototypical networks for few-shot learning
- Multi-loss objective functions
- Calibrated probability outputs (isotonic regression)

---

## 📈 Key Findings

### 1. Synergistic Multimodal Gain
Three-way fusion substantially outperforms pairwise combinations, demonstrating:
- **Error complementarity**: Each modality captures different at-risk cases
- **Adaptive weighting**: Sample-specific contribution balancing
- **Calibration regularization**: Third modality as "tie-breaker"

### 2. Feature Importance (SHAP Analysis)

**Top Questionnaire Features (Module 2):**
- SCAS19 (anxiety item)
- pSDQ13 (parent-report behavioral item)
- SCAS02, SCAS28 (anxiety items)

**Top Phenotypic Features (Module 3):**
- Resting pulse (physiological marker)
- Flanker task performance (cognitive control)
- Processing speed
- Biological sex
- Systolic blood pressure

### 3. Clinical Implications
- Early identification enables timely preventive interventions
- Multimodal integration aligns with bio-psycho-social model of anxiety
- Interpretable predictions support clinical decision-making

---

## 📊 Figures

<details>
<summary><b>Framework and Architecture</b></summary>

### Overall Framework
![Framework](Fig1_Overall_Framework.png)

### MRI Preprocessing Pipeline
![MRI](Fig2_MRI_Preprocessing.png)

### Questionnaire Pipeline
![Questionnaire](Fig3_Questionnaire_Pipeline.png)

### Prototypical Network Architecture
![Prototypical](Fig4_Prototypical_Network.png)

</details>

<details>
<summary><b>Results</b></summary>

### Confusion Matrix
![Confusion](Fig5_Confusion_Matrix.png)

### Ablation Study
![Ablation](Fig6_Ablation_Study.png)

### Bootstrap Confidence Intervals
![CI](Fig7_Bootstrap_CI.png)

</details>

<details>
<summary><b>Interpretability (SHAP)</b></summary>

### Module 2: Questionnaire Features
![SHAP Module 2](Fig9_Module2_SHAP_Summary.png)
![Feature Importance Module 2](Fig10_Module2_Feature_Importance.png)

### Module 3: Phenotypic Features
![SHAP Module 3](Fig11_Module3_SHAP_Summary.png)
![Feature Importance Module 3](Fig12_Module3_Feature_Importance.png)

</details>

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{taosif2026twin,
  title={A twin-aware multimodal deep learning framework with optimized late fusion for early prediction of adolescent anxiety disorder},
  author={Taosif, Md. and Chaman, Ummay Maimona and Prova, Nazifa Anjum and Taher, Sidrat Moon and Alam, Md Golam Rabiul and Rahman, Rafeed},
  journal={PLOS ONE},
  year={2026},
  publisher={Public Library of Science}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Md. Taosif** - *Corresponding Author* - md.taosif@g.bracu.ac.bd
- **Ummay Maimona Chaman**
- **Nazifa Anjum Prova**
- **Sidrat Moon Taher**
- **Md Golam Rabiul Alam** - *Corresponding Author* - rabiul.alam@bracu.ac.bd
- **Rafeed Rahman**

**Affiliation:** Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh

---

## 🙏 Acknowledgments

- Queensland Twin Adolescent Brain (QTAB) study team for providing the dataset
- PLOS ONE reviewers for valuable feedback
- BRAC University for institutional support

---

## 📧 Contact

For questions or collaborations:
- **Email**: md.taosif@g.bracu.ac.bd, rabiul.alam@bracu.ac.bd
- **Institution**: BRAC University, Dhaka, Bangladesh

---

## ⚠️ Data Availability

The QTAB dataset used in this study is subject to data use agreements. Researchers interested in accessing the data should contact the QTAB study team directly.

---

## 🔗 Links

- [PLOS ONE Journal](https://journals.plos.org/plosone/)
- [QTAB Study](https://qimrberghofer.edu.au/)
- [BRAC University CSE](https://www.bracu.ac.bd/academics/departments/computer-science-and-engineering)

---

<p align="center">
  <b>⭐ If you find this work useful, please consider starring the repository! ⭐</b>
</p>
