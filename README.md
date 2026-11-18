# Retinal Artery-Vein Segmentation with Attention U-Net

Segmenting the blood vessels in a retinal photograph and labelling each one as an **artery** or a **vein**, including the points where they cross. Artery and vein calibre is used to study hypertension, diabetes and cardiovascular risk, but labelling it by hand is slow, which is why it is worth automating.

This work was done during my Machine Learning Research Internship at IIT Delhi (June to August 2026), on the **GAVE-2** preliminary training set (50 fundus images).

## Results

Final model: `9_attention_UNET_multilabel.ipynb`. Dice is computed per image and averaged.

| Class | 5-fold cross-validation | Held-out test (8 images) |
|---|---|---|
| Vessel (any) | 0.764 ± 0.009 | **0.752** |
| Vein | 0.755 ± 0.007 | **0.736** |
| Artery | 0.686 ± 0.016 | **0.661** |
| Crossing (artery over vein) | 0.382 ± 0.027 | 0.333 |

The 8 test images are kept out of all training and model selection and scored once, after every setting had been chosen on the cross-validation folds. Each test score is the mean of the five fold models, with a spread of at most ± 0.009.

## Data and protocol

- 50 images: **8 held out for testing**, 42 used for 5-fold cross-validation (33-34 train / 8-9 validation per fold). The split is fixed in `folds.json`, which notebooks 8 and 9 read, so their results compare on the same images.
- Model selection uses the mean Dice over artery, vein and crossing on the validation folds only.
- A change counts as real only if it exceeds twice the fold standard deviation. The notebooks print that threshold under every result.

## The final pipeline

1. **Preprocessing:** field-of-view (ROI) masking, then CLAHE on the green and red channels plus a green-minus-red channel. Green gives vessel contrast; red is where arteries and veins differ, since oxygenated blood absorbs less red light.
2. **Ground truth:** RGB annotations converted into three overlapping labels: artery, vein and any vessel. A crossing is positive in both the artery and the vein channel.
3. **Augmentation and sampling:** offline frame-level augmentation of full 768 × 768 images, then 256 × 256 patches (stride 128) kept only when they contain retina and vessel.
4. **Model:** Attention U-Net with an ImageNet-pretrained ResNet-34 encoder (at a 0.1× learning rate) and a full-resolution side branch for thin vessels.
5. **Loss:** Focal Tversky (α = β = 0.5) plus clDice, a topology-aware term that rewards continuous vessel centrelines, ramped in over the first 15 epochs.
6. **Training:** AdamW, learning rate 1e-4, batch size 8, up to 100 epochs, an EMA of the weights (decay 0.999), and early stopping with patience 25.
7. **Decision:** a separate threshold per channel, tuned on validation, instead of `argmax`.

## What the experiments showed

**Most errors were vessels drawn too thick, not vessels missed.** An error analysis on the 4-class model (`8_attention_UNET_kernel_cldice.ipynb`) split its false pixels by type. A thickness halo around real vessels came to 37% of the true vessel area. Invented vessels came to only 2%, and missed thin vessels to 18%. On the test images, removing the halo alone would have raised vessel Dice from 0.745 to 0.894, and clDice was 0.771, so the vessel tree was largely right but its widths were wrong. The cause was the loss: with β > α, Tversky penalises a missed pixel more than an invented one, which Dice does not do. The final model therefore uses α = β.

**One label per pixel cannot represent a crossing.** The 4-class softmax made artery, vein and crossing compete for the same pixel, which squeezed the crossing class and made it the source of 79% of the fold variance. Three independent sigmoid outputs let a crossing be both artery and vein:

| 5-fold Dice | 4-class softmax | 3-channel sigmoid (final) |
|---|---|---|
| Artery | 0.655 ± 0.021 | **0.686 ± 0.016** |
| Vein | 0.724 ± 0.009 | **0.755 ± 0.007** |
| Crossing | 0.375 ± 0.054 | 0.382 ± 0.027 |

Artery and vein each gained about 0.03 and the crossing estimate became half as noisy. The crossing class itself is still weak and remains the main open problem.

**Test-time augmentation was not a reliable gain.** Averaging predictions over flipped and rotated copies added about 0.01 Dice on four folds and lost 0.03 to 0.04 on the fifth (`8_attention_UNET_kernel_cldice.ipynb`), so the final model does not use it.

## Repository map

The notebooks are in two series. The numbered `00`-`11` series is the June exploration and the `attention_UNET` series is the final work, despite the earlier names containing "final".

| Notebooks | Period | What they cover |
|---|---|---|
| `00`-`11` | June | Data split, raw baseline, preprocessing and augmentation studies, loss experiments, clDice tuning, RRWNet and ViT trials |
| `SwinUnet.ipynb` | July | Swin-UNet comparison |
| `attention_UNET`, `2`-`4.attention_UNET` | July | Attention U-Net pipelines |
| `5.attention_UNET_TTA`, `5.baseline_attention_UNET_TTA` | July-August | Fixed-split baseline and test-time augmentation |
| `6`, `7_attention_UNET_kernel_cldice` | August | Kernel schedule and clDice tuning |
| `8_attention_UNET_kernel_cldice` | August | 4-class research pipeline; writes `cv_results.json`; error analysis |
| **`9_attention_UNET_multilabel`** | **September** | **Final 3-channel model and held-out test** |

`cv_results.json` holds the 4-class cross-validation run from notebook 8, not the final model. The final numbers are printed in notebook 9.

## Running it

The dataset is not included. Download GAVE-2 from its challenge page, then set `DATASET_PATH` in the configuration cell of `9_attention_UNET_multilabel.ipynb` to the training folder.

```bash
pip install torch torchvision albumentations opencv-python scikit-learn matplotlib numpy tqdm
jupyter notebook 9_attention_UNET_multilabel.ipynb
```

A GPU is needed in practice: one fold took about an hour in the original runs.

## Limitations

- 50 images from one dataset, with 8 for testing. Cross-dataset performance is untested.
- The crossing class (0.33 test Dice) is not yet usable on its own.
- Next experiment: a consistency constraint so that the vessel output equals the union of the artery and vein outputs. It was held back so its effect can be measured separately.
