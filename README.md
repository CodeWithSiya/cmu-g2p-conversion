# CMU Grapheme-to-Phoneme Conversion with LSTM Encoder-Decoder Models

A PyTorch implementation of LSTM encoder-decoder models for G2P conversion using the CMU Pronouncing Dictionary. Explores three architectural variants: no context, fixed context, and attention-based dynamic context.

**Key Features:**

- Custom LSTM cell implementation with gating mechanisms
- Three architectural variants to compare context passing strategies
- Cross-attention mechanism for dynamic context generation
- Hyperparameter optimization via grid search
- Attention visualization and error analysis

## Dataset

CMU Pronouncing Dictionary (~135K words) split into training (92,426), validation (11,553), and test (11,554) sets. CSV format with `word` (graphemes) and `phonemes` (ARPAbet).

## Setup Instructions

### Requirements

- Python 3.10+
- CUDA 11.8+ (optional)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib torch editdistance
```

### Running

```bash
jupyter notebook src/main.ipynb
```

Run cells sequentially to reproduce results.

## File Structure

```
cmu-g2p-conversion/
├── README.md                # This file
├── data/
│   ├── g2p_train.csv        # Training set (92,426 words)
│   ├── g2p_val.csv          # Validation set (11,553 words)
│   └── g2p_test.csv         # Test set (11,554 words)
└── src/
    └── main.ipynb           # Complete implementation notebook
```

### Notebook Structure

1. Setup - Environment configuration, imports
2. Dataset Processing - Data loading, vocabulary building, DataLoaders
3. LSTM Cell - Custom LSTMCell implementation with gating
4. Encoder-Decoder Architecture - Encoder, CrossAttention, Decoder, Seq2Seq
5. Training & Validation - Loss computation and evaluation metrics
6. Hyperparameter Tuning - Grid search
7. Architecture Comparison - Train three model variants
8. Evaluation - Test set performance
9. Attention Visualization - Attention weight heatmaps
10. Error Analysis - Failure case comparison

## Model Variants

1. **No Context** - Encoder hidden state initializes decoder only
2. **Fixed Context** - Final encoder state projected into decoder gates at each step
3. **Attention Context** - Dynamic context via cross-attention at each step

## Results

Best hyperparameters: LR=10^-3, Embedding=64, Hidden=256

| Model             | Test PER | Word Accuracy |
| ----------------- | -------- | ------------- |
| No Context        | 0.1300   | 0.5628        |
| Fixed Context     | 0.1278   | 0.5689        |
| Attention Context | 0.1154   | 0.5936        |

Attention achieves ~12% relative improvement in word accuracy over baseline by addressing the hidden state bottleneck.

## Dependencies

- PyTorch - Deep learning framework
- NumPy - Numerical computations
- Matplotlib - Visualization
- EditDistance - PER calculation

## Author

Siyabonga Madondo (MDNSIY014)  
Date: 18 May 2026
