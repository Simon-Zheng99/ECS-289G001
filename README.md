# ECS-289G001
# Layer-wise Probing and Analysis for the Explainability of Bidirectional Auto-Regressive Transformers for Abstractive Summarization

**Authors:** Devavrat Singh Bisht, Simon Zheng, Daniel Zavorotny
UC Davis — ECS 289G Final Project

## Overview

This project investigates how individual decoder blocks in a BART-based sequence-to-sequence model contribute to abstractive summarization. Rather than treating the decoder as a black box that only produces meaningful output at its final layer, we probe the hidden states of *every* decoder block to understand what each layer is individually responsible for during generation.

**Key Finding:** Each decoder block specializes in a distinct aspect of text generation — moving from lexical diversity in early layers, to punctuation/phrasing in middle layers, to logical coherence and refinement in the final layers.

## Motivation

Existing interpretability techniques for transformers (e.g., attention maps, Token-based Importance Attribution/TiBA) reveal *token-level dependencies* but fail to explain the *functional role* of individual decoder blocks. It's difficult to attribute specific behaviors — like syntax, fluency, or coherence — to any particular layer using these methods, and they cannot explain the combined effect of multiple layers acting together.

This project fills that gap by directly analyzing the output at each decoder block to determine what each one contributes to the final summary.

## Approach

We use `facebook/bart-base` and intercept the hidden states from each of its decoder blocks (0–6), projecting them through the language modeling head (`lm_head`) to see what text each layer would generate on its own.

The analysis is split into two phases:

### Phase 1 — Zero-shot Layer Probing
The pretrained `lm_head` (trained only for the final decoder layer) is applied directly to the hidden states of earlier decoder blocks, with no additional training. This reveals what information is "naturally" present at each layer without any adaptation.

- Corresponds to `generate_summary()` in the code, run with different `layer_number` values.
- Result: early/middle blocks produce degenerate, repetitive output (e.g., `"low low low low..."`, `"long long long long..."`) since the pretrained head was never trained to interpret their representations. Only the final block(s) produce fluent output.

### Phase 2 — Per-Layer Prediction Heads (Fine-Tuning)
A **separate lm_head is fine-tuned for each decoder block**, so that each layer's hidden states are decoded through a head trained specifically to interpret them.

- Corresponds to `fine_tune_lm_head()` in the code.
- All BART parameters are frozen except a fresh copy of `lm_head` per layer, trained with cross-entropy loss against the reference summaries.
- Result: a much clearer progression of capability across layers — from empty/degenerate output, to token diversity, to punctuation handling, to coherent summaries — showing what each block *can* meaningfully contribute once properly decoded.

## Code Structure

`ModifiedBartModelV1` — wraps a pretrained BART model and exposes:

- **`get_decoder_layer_outputs`** — runs a forward pass with `output_hidden_states=True` and returns hidden states from every decoder layer.
- **`generate_summary(text, layer_number, max_length)`** — autoregressively generates a summary using the specified decoder layer's hidden states routed through `lm_head` (Phase 1 probing).
- **`fine_tune_lm_head(train_dataset, layer_number, epochs, batch_size, learning_rate)`** — freezes the base model and fine-tunes a dedicated `lm_head` for a specific decoder layer (Phase 2), saving the resulting weights to `lm_head_{layer_number}.pth`.

## Experimental Setup

| Setting | Value |
|---|---|
| Base model | `facebook/bart-base` |
| Dataset | CNN/DailyMail |
| Full dataset split | 287K train / 13K val / 11K test |
| Used split | 287K train / 100 test |
| Max summary length | 512 tokens |
| Epochs | 4 (per HuggingFace recommendation) |
| Loss function | Cross-Entropy Loss |

## Results

- **Phase 1 (frozen head):** Punctuation and coherent structure only emerge at the last 2 decoder blocks (5–6); earlier blocks collapse into repeated single-token loops.
- **Phase 2 (fine-tuned heads):** Punctuation usage appears much earlier (from block 0 onward) and is more evenly distributed across blocks, and a larger number of "insightful" summaries are produced at earlier prediction heads compared to Phase 1.
- Across both phases, a consistent pattern emerges:
  - **Early blocks** → lexical/word diversity
  - **Middle blocks** → punctuation and phrasing
  - **Late blocks** → logical coherence and final refinement

## Conclusion

Attention maps show token-level dependencies but not what individual decoder blocks are functionally responsible for. This project's layer-wise prediction-head approach fills that gap, offering a clearer, more decomposable picture of transformer decoder behavior — useful for debugging, optimizing, and fine-tuning summarization models.

## Future Work

- More extensive fine-tuning of per-layer heads.
- Extending the analysis to other NLP generation tasks beyond summarization.
- Refining the techniques used to attribute responsibility to individual layers.

## Requirements

- `torch`
- `transformers`
- `tqdm`
- `datasets` (for loading CNN/DailyMail)

## Usage

```python
model = ModifiedBartModelV1(model_name="facebook/bart-base", device="cuda:0")

# Phase 1: probe a specific decoder layer with the pretrained head
summary, inputs = model.generate_summary(text, layer_number=4)

# Phase 2: fine-tune a dedicated head for a specific decoder layer
model.fine_tune_lm_head(train_dataset, layer_number=4, epochs=50)
```
![Insightful Summaries Graph](combined_summaries.jpg)