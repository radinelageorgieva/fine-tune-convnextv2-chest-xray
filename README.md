# fine-tuning-convnextv2-chest-xray
Multi-label classification of 14 lung pathologies using a ConvNextV2 on the NIH Chest X-ray 14 dataset. 

# Multi-Label Chest X-Ray Classification using ConvNeXt-V2 and LoRA

This repository contains a deep learning project for multi-label thoracic pathology classification from chest X-ray images. The project utilizes a fine-tuned **ConvNeXt-V2** architecture combined with **LoRA (Low-Rank Adaptation)** to efficiently detect 14 distinct lung conditions while addressing severe dataset class imbalance.

## Key features
* State-of-the-Art Backbone: `ConvNeXt-V2` for advanced image feature extraction.
* Efficient Fine-Tuning: Implements Parameter-Efficient Fine-Tuning (PEFT) via LoRA to train large vision model with limited computational resources.
* Multi-Label Detection: Independent evaluations for all 14 clinical pathologies simultaneously using BCEWithLogitsLoss.
* Imbalance-Aware Loss Function: Applied pathologically-weighted **pos_weight** inside BCEWithLogitsLoss to dynamically handle severe class imbalance during training.
* Explainable AI (XAI):
  - Multi-label Saliency Maps to track exactly where the model focuses its pixel-level attention
  - Qualitative visual evaluation to compare true labels with model predictions
  - Identification of errors caused by overlapping white shadows on chest X-rays


## Evaluation and results summary
* Macro AUC-ROC: **0.8409** - demonstrates moderate ability to correctly rank and separate healthy vs. sick lung structures.
* Macro F1-score: **31%** - heavily limited by applying a static 0.50 decision threshold on weighted probabilities, which hides the true patterns learned via pos_weight.
* Specificity: Consistently above **90-95%** across most active classes, showing an accuracy in identifying true negative (healthy) lung cases.

## Main libraries
The project is built using Python and the following core data science and deep learning libraries:
* Deep Learning: torch (PyTorch), torchvision, timm (Pretrained Vision Models), peft (LoRA configurations)
* Data Processing: pandas, numpy, PIL
* Evaluation Metrics: scikit-learn: roc_auc_score, classification_report, confusion_matrix
* Visualization & Tracking: matplotlib, seaborn, mlflow, tqdm

## Future work
To transition this architecture into real-world hospital deployment, future updates will focus on:
1. Dynamic threshold tuning: moving away from a rigid 0.50 decision threshold to class-specific, optimized cuts to minimize critical False Negatives.
2. Advanced loss functions: integrating Focal Loss to dynamically penalize the model for rare diseases that are difficult to classify.

## References
1. *NIH Chest X-ray Dataset:* [Wang X., et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." IEEE CVPR, 2017.](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)
2. ConvNeXt-V2: Woo S., et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." arXiv preprint arXiv:2301.00808, 2023.
3. LoRA (Low-Rank Adaptation): Hu E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv preprint arXiv:2106.09685, 2021.
4. PyTorch BCEWithLogitsLoss: PyTorch Core Team. Binary Cross Entropy with Logits Loss documentation. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
5. Fine-Tuning Framework (PEFT): Mangrulkar S., et al. "PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models." Hugging Face Blog, 2023. https://huggingface.co/blog/peft


