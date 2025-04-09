# Comparative Study of GAN Loss Functions on MedMNIST

## Overview
This research investigates the performance impact of different loss functions in Generative Adversarial Networks (GANs) on the **MedMNIST PathMNIST** medical imaging dataset. The study focuses on evaluating and comparing:
- **LS-GAN (Least Squares GAN)**
- **WGAN (Wasserstein GAN)**
- **WGAN-GP (Wasserstein GAN with Gradient Penalty)**

## Dataset
- **Dataset:** [MedMNIST - PathMNIST](https://medmnist.com/)
- **Input Shape:** 3 × 28 × 28 RGB histopathological images
- **Application:** Synthetic data generation for medical diagnostics research

## GAN Variants Developed
Each GAN variant was implemented independently with tailored architecture, loss functions, and training strategies:

| GAN Variant  | Loss Function                  | Architecture Features              | Optimizer Used      |
|--------------|--------------------------------|------------------------------------|---------------------|
| `LSGAN`      | Mean Squared Error (MSE)       | Custom G & D with BatchNorm        | Adam (0.0002)       |
| `WGAN`       | Wasserstein Loss               | Weight clipping, RMSProp optimizer | RMSProp (0.00005)   |
| `WGAN-GP`    | Wasserstein + Gradient Penalty | Gradient penalty implementation    | Adam (0.0001)       |

## Experimental Configuration
- **Epochs:** 50 per model
- **Latent Space Dimension:** 100
- **Batch Size:** 64
- **Framework:** PyTorch
- **Environment:** GPU and CPU compatible

## Output Visualizations
Each GAN model produces:
- Individual images saved as `sample_*.png`
- A visual summary `visual_inspection.png`

## Evaluation Metrics
The following metrics were used to assess and compare the quality of generated images:
- **Inception Score (IS)** – evaluates diversity and quality
- **Fréchet Inception Distance (FID)** – compares distributions of real and generated images
- **Visual Inspection** – human-readable insight into realism

IS and FID were estimated using approximations due to dataset constraints.

## TensorBoard Integration
Launch TensorBoard to visualize losses and generated outputs:
```bash
cd path_of_the_folder
tensorboard --logdir=runs
```
Click the link and open your browser.

## Summary
This research demonstrates how different GAN loss functions influence training stability and output quality. The study concludes with a visual and quantitative comparison of image generation capabilities using MedMNIST.
