# flow_matching_tutorial

This repository implements a minimal & clean from-scratch version of the modern image generation pipeline: Flow Matching (FM) + Diffusion Transformer (DiT) on the classic MNIST handwritten digit dataset.
All core logic is hand-coded from 0: no diffusers/torchvision.models high-level wrappers, no pre-built Flow Matching modules. You can see the complete mathematical derivation and engineering implementation of Flow Matching, DiT architecture, transformer blocks, attention mechanism, and image generation pipeline in pure PyTorch.

✅ Core Features:

- Pure scratch implementation of Flow Matching (continuous normalizing flow) core algorithm
- Minimal Diffusion Transformer (DiT) backbone for image modeling (no ViT pre-trained weights)
- End-to-end MNIST image generation pipeline: train → sample → visualize
- Full PyTorch implementation with clean, commented code (easy to read & modify)
- Lightweight: MNIST is 28x28 grayscale, fast training even on CPU/low-end GPU

