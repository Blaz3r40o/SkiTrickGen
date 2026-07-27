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

## Domain: The Aerial Trick Space

Freestyle aerial maneuvers are built from a small set of composable elements:

- Flips (somersaults): 0–3 per jump
- Twists: distributed across the flips, varying per discipline
- Body position: tuck, pike, or layout
- Rotation direction: forward or backward, and takeoff/landing orientation (forward, switch)

This compositional structure — a small set of discrete building blocks combined into a large space of named tricks, each with an official difficulty score — makes aerial skiing well suited to conditional generative modeling: the conditioning code is low-dimensional and interpretable, while the resulting kinematic trajectory is continuous and physically constrained.

---

## Datasets

### FIS Degree-of-Difficulty Table

Provider:

International Ski and Snowboard Federation (FIS), Freestyle Skiing Judging Handbook

Variables used:

- Trick name (`trick_name`)
- Number of flips (`n_flips`)
- Number of twists (`n_twists`)
- Body position (`position`)
- Rotation direction (`rotation_direction`)
- Official degree-of-difficulty score (`dd_score`)

Edition:

- Judging Handbook, 2019 and 2025 editions

Source:

https://www.fis-ski.com/ (Freestyle Skiing Judging Handbook, published DD tables)

---

### Synthetic Kinematic Trajectories

Provider:

Generated within this project using a simplified rigid-body flight-dynamics model, parameterized by each row of the FIS DD table above.

Variables:

- Time-normalized orientation angles (yaw, pitch, roll)
- Tuck factor (proxy for moment-of-inertia change during flight)
- Reduced limb configuration (simplified skeleton, ~8–10 points)

Period:

- Generated per training run; not time-indexed real-world observations

Source:

- Derived from flight-dynamics equations (see Mathematical Formulation), grounded in published sports-biomechanics parameter ranges (see References)

---

## Mathematical Formulation

### Flight Dynamics

Given vertical takeoff velocity `v_y0` and gravitational acceleration `g`, time of flight is:

t_flight = 2 · v_y0 / g

Once airborne, angular momentum `L` about the rotation axis is conserved in the absence of external torque:

L = I(t) · ω(t) = constant

Tucking reduces the moment of inertia `I(t)`, increasing angular velocity `ω(t)` — the same principle that lets a figure skater spin faster by pulling in their arms. The cumulative rotation angle is:

θ(t) = ∫₀ᵗ ω(τ) dτ

constrained so that `θ(t_flight)` matches the flip/twist count specified by the trick's FIS code.

### Conditional Variational Autoencoder

Given a trick sequence `x` and its conditioning code `c`, the model defines an encoder `q_φ(z | x, c)`, a decoder `p_θ(x | z, c)`, and a standard normal prior `p(z)`. Training maximizes the conditional evidence lower bound:

L(θ, φ; x, c) = E[log p_θ(x | z, c)] − KL(q_φ(z | x, c) ‖ p(z))

with the reconstruction term implemented as mean-squared error over the joint-angle time series, and the KL term annealed from zero during early training to avoid posterior collapse.

### Latent Interpolation

Given two trick codes with encoded latent means `z1` and `z2`, a novel hybrid trick is generated at interpolation weight `α ∈ [0, 1]`:

z(α) = (1 − α) · z1 + α · z2

and decoded through `p_θ(x | z(α), c(α))` to produce a smooth morph between two real maneuvers.

### Transformer Baseline

As a comparison model, pose vectors are quantized into a discrete vocabulary via k-means clustering, and an autoregressive Transformer decoder models the sequence as a product of conditionals, trained with cross-entropy loss over the quantized pose tokens.

---

## Methodology

1. Digitize the FIS DD table into a structured dataset.
2. Generate synthetic kinematic trajectories per trick code using the flight-dynamics model, with per-instance noise to emulate inter-athlete variability.
3. Preprocess: normalize the time axis, standardize continuous features, encode conditioning vectors.
4. Train the Conditional VAE (primary model) and the Transformer baseline on an 80/20 split, with entire trick classes held out to test generalization.
5. Generate: sample the prior for seen and unseen trick codes; interpolate between trick pairs in latent space.
6. Evaluate and visualize results.

---

## Evaluation

- Reconstruction quality on held-out trick sequences (MSE).
- Generalization quality for trick codes withheld entirely from training.
- Physical plausibility of generated angular velocities, checked against reported human biomechanical limits.
- Comparison of the CVAE against the Transformer baseline and a naive nearest-neighbor/noise-perturbation baseline.
- Whether unsupervised structure in the latent space recovers the known flip/twist/position taxonomy.

---

## Repository Structure

ski-trick-gen/
- README.md
- requirements.txt
- .gitignore
- data/ — FIS DD table
- notebooks/ — data generation, model training, evaluation and visualization
- src/ — physics-based data generator, model architectures, training and visualization utilities
- results/ — figures and model checkpoints

---

## Installation

Clone the repository, create a virtual environment, and install dependencies from `requirements.txt`.

---

## Usage

Run the notebooks in order: data generation, model training, then evaluation and visualization.

---

## Limitations

- Kinematic sequences are synthetic, derived from a simplified rigid-body model rather than real motion capture; generated tricks describe plausible orientation/rotation profiles, not full biomechanically accurate limb dynamics.
- The reduced-order skeleton does not capture fine-grained limb or grab dynamics.
- Physical plausibility checks validate rotation-rate ranges against literature values but do not guarantee that a real athlete could safely execute a generated maneuver.
- This project is intended as a research and visualization aid, not a substitute for real coaching or biomechanical safety assessment.

---

## References

- FIS Freestyle Skiing Judging Handbook — official Degree of Difficulty tables for aerial jump codes.
- Sports biomechanics literature on aerial skiing take-off dynamics and rotational velocity limits.
- Kingma, D. P., & Welling, M. — Auto-Encoding Variational Bayes (VAE formulation).
- Sohn, K., Lee, H., & Yan, X. — Learning Structured Output Representation Using Deep Conditional Generative Models (CVAE).
- Vaswani, A. et al. — Attention Is All You Need (Transformer architecture used for the comparison model).

*(Full citation details, including specific editions and DOIs, to be finalized in the notebook's reference section.)*
