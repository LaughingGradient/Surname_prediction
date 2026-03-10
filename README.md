# Transformer-based Surname Prediction

A **character-level Transformer Encoder–Decoder model implemented from scratch in PyTorch** to generate an Indian surname given a first name.

The model learns the mapping:

First Name → Surname


Example:



Input : rahul
Output : sharma


The surname is generated **character-by-character using autoregressive decoding**.

---

# Project Motivation

Transformers are the foundation of modern language models such as GPT, BERT, and T5.  
This project demonstrates how to build a **complete Transformer architecture from scratch** without using high-level libraries.

The goal is to understand:

- Multi-head attention
- Positional encoding
- Encoder–decoder architecture
- Autoregressive text generation
- Sequence-to-sequence learning

---

# Dataset

The dataset consists of **Indian names** with the format:



name | gender | race


Example rows:



rahul sharma
sachin tendulkar
virat kohli
amit verma


From this dataset we extract **first name and surname pairs**:



rahul → sharma
sachin → tendulkar
virat → kohli


### Dataset Processing

1. Load dataset
2. Remove malformed rows
3. Keep only names with **two or more tokens**
4. Extract:
   - `first_name`
   - `last_name`
5. Sample **5,000 examples for training**

---

# Model Architecture

The model follows the original **Transformer architecture** introduced in:

**"Attention Is All You Need" (Vaswani et al., 2017)**.

## Encoder

Processes the input first name.

Components:

- Character Embedding
- Positional Encoding
- Multi-head Self Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization

## Decoder

Generates the surname.

Components:

- Masked Self Attention
- Cross Attention (attends to encoder output)
- Feed Forward Network
- Residual Connections
- Layer Normalization

## Output Layer

A linear layer predicts the **next character token**.

---

# Tokenization

The model uses **character-level tokenization**.

### Vocabulary



a-z
<pad>
<sos>
<eos>
<unk>


### Special Tokens

| Token | Purpose |
|------|------|
| `<sos>` | Start of sequence |
| `<eos>` | End of sequence |
| `<pad>` | Padding token |
| `<unk>` | Unknown character |

---

# Training

The model is trained using **teacher forcing**.

### Loss Function



CrossEntropyLoss(ignore_index=<pad>)


### Optimizer



Adam


### Training Progress

Example training output:



Epoch 1 | Avg Loss: 2.83
Epoch 5 | Avg Loss: 1.92
Epoch 10 | Avg Loss: 1.41


---

# Inference

During inference, the model generates surnames **autoregressively**.

Process:



<sos> → predict char1
<sos> char1 → predict char2
<sos> char1 char2 → predict char3
...
until <eos>


### Example

```python
generate_surname(model, "sachin")
```

Output:

tendulkar

Project Structure
surname-prediction/
│
├── surname_pred.ipynb
├── README.md
├── requirements.txt

Installation

Clone the repository:

git clone https://github.com/your-username/surname-prediction.git
cd surname-prediction


Install dependencies:

pip install -r requirements.txt


Launch Jupyter:

jupyter notebook


Open:

surname_pred.ipynb
