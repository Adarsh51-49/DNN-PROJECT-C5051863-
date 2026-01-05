# Visual Storytelling with Temporal-Aware Cross-Modal Attention

## Quick Links
- **[Experiments Notebook](experiments.ipynb)** - Full experimental workflow
- **[Baseline Results](results/baseline/)** - Original model performance
- **[Improved Results](results/improved/)** - Results with my innovation

## Innovation Summary
**I modified the traditional attention mechanism to incorporate temporal priors and cross-modal interactions, expecting to improve coherence in generated stories by 15-20%.**

## Project Overview
This project implements a multimodal sequence model for visual storytelling. Given a sequence of K images and corresponding text descriptions, the model predicts the next multimodal element (K+1) in the sequence. The innovation involves a novel **Temporal-Aware Cross-Modal Attention** mechanism that dynamically weights information based on temporal position and modality relevance.

## Key Results
| Metric | Baseline | Improved | Change |
|---|---|---|---|
| BLEU-4 | 0.45 | 0.52 | **+15.5%** |
| Perplexity | 25.3 | 21.1 | **-16.6%** |
| Human Evaluation | 3.2/5 | 3.8/5 | **+18.7%** |
| Story Coherence Score | 0.61 | 0.73 | **+19.7%** |
| Repetition Rate | 12.3% | 7.4% | **-39.8%** |

## Most Important Finding
> The innovation reduced repetitive phrases by 40% in long sequences and improved narrative coherence by 19.7%, as shown in [this visualization](results/comparative_analysis/repetition_analysis.png).

## Architectural Modifications
1. **Temporal-Aware Attention**: Added positional temporal encoding to attention weights
2. **Cross-Modal Fusion Layer**: Implemented gated cross-attention between visual and textual features
3. **Multi-Task Learning**: Combined story prediction with sequence reconstruction task
4. **Curriculum Learning**: Progressive training from short to long sequences

## Repository Structure