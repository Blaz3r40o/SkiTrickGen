# SkiTrickGen: Generative Modeling of Freestyle Aerial Ski Tricks

## A Deep Learning Investigation Using Biomechanically-Grounded Synthetic Kinematics and FIS Degree-of-Difficulty Data


---

## Project Overview

Freestyle aerial skiing is a discipline in which athletes execute precisely codified combinations of flips, twists, and body positions during a brief airborne phase, judged in part using a published Degree of Difficulty (DD) scale maintained by the International Ski and Snowboard Federation (FIS).

Motion data for this sport is scarce in the public domain. Real motion-capture of athletes performing aerial tricks requires specialized equipment, consenting elite athletes, and is rarely released for research or product use. This project investigates whether a generative deep learning model, trained on a physically-grounded but synthetically generated set of trick trajectories, can learn a structured "trick space" capable of reconstructing known maneuvers and producing new, plausible ones it was never explicitly trained on.

The project combines:

- Sports biomechanics
- Data engineering and synthetic data generation
- Generative deep learning
- Statistical and physical evaluation
- Latent space explainability

---

## Research Question

> To what extent can a generative model, trained on biomechanically-grounded synthetic trick kinematics conditioned on real FIS difficulty codes, learn a latent representation that supports reconstruction of known aerial tricks and generation of novel, physically plausible ones?

Specifically, the project investigates:

- Whether flip count, twist count, and body position are sufficient conditioning signals to reconstruct a trick's rotational trajectory.
- Whether the model generalizes to trick codes withheld entirely from training.
- Whether interpolating between two real tricks in latent space produces a physically coherent "hybrid" maneuver.
- Whether the learned latent space recovers the known flip/twist/position taxonomy without being told to.
- Whether a Conditional VAE outperforms a Transformer-based autoregressive baseline and a naive interpolation baseline on reconstruction and plausibility.

---
