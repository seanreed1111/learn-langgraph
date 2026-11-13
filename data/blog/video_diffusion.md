---
title: Diffusion Models for Video Generation
---
# Diffusion Models for Video Generation

Video generation with diffusion models extends image diffusion by adding a time dimension. The core difficulty is temporal consistency: frames must agree with each other about objects, lighting and motion, and video data is far larger and scarcer than image data.

## Approaches

One approach is to build on a pre-trained image diffusion model and insert temporal layers, such as temporal attention or 3D convolutions, then fine-tune on video. Another is to generate sparse keyframes first and then interpolate the frames in between. Latent diffusion reduces cost by running the diffusion process in a compressed latent space rather than in pixel space.

## Training and sampling

Training mixes image and video data so the model keeps strong spatial quality. Sampling uses classifier-free guidance to trade diversity for fidelity. Long videos are produced by autoregressively extending clips, which risks drift, so many systems condition each new clip on the last frames of the previous one.
