The Fractal AGI Cognitive Core is built upon a rich mathematical foundation spanning complex dynamics, functional analysis, information theory, and evolutionary optimization. Below is a comprehensive inventory of all mathematical concepts, equations, and techniques employed in the project, organized by subsystem.

---

## I. Core Cognitive Dynamics

### 1. Complex Field Iteration (Mandelbrot / Julia Sets)

**Primary Equation:**
\[
z_{n+1} = z_n^2 + C
\]

- **Variables:**
  - \(z \in \mathbb{C}\): Complex state of a "pixel" in the 2D field.
  - \(C \in \mathbb{C}\): Color‑modulated parameter.
- **Escape Radius:** \(|z| > R_{\text{esc}}\) where \(R_{\text{esc}} = 2.0\) (standard Mandelbrot escape condition).
- **Escape Tracking:** A boolean mask tracks diverged points, freezing their evolution.

**Mathematical Properties:**
- The boundary between bounded and unbounded orbits forms a fractal (Julia set for fixed \(C\), Mandelbrot set when \(C\) varies).
- **Fractal Dimension:** Quantifies boundary complexity; used as a cognitive metric.

### 2. Color Modulation (Affective Parameter Mapping)

**Mapping Function:**
\[
C(\boldsymbol{\lambda}) = \left( \sum_{i=1}^{N} \lambda_i w_i^{\text{real}} \right) + i \left( \sum_{i=1}^{N} \lambda_i w_i^{\text{imag}} \right)
\]
with clipping:
\[
C \leftarrow \begin{cases}
C & \text{if } |C| \leq C_{\max} \\
C \cdot \frac{C_{\max}}{|C|} & \text{if } |C| > C_{\max}
\end{cases}
\]
where \(C_{\max} = 2.0\).

- **\(\boldsymbol{\lambda} \in \mathbb{R}^N\):** Input color vector (intensity per wavelength band).
- **\(\mathbf{w} \in \mathbb{R}^N\):** Learnable/evolvable band weights.
- **Default bands:** \(N = 5\) with centers at 400, 480, 560, 640, 720 nm.

### 3. Rotation Operator

**Transformation:**
\[
z(x,y) \leftarrow z(x,y) \cdot e^{i\theta}
\]
- **\(\theta\):** Rotation angle (radians), updated per step: \(\theta_{t+1} = (\theta_t + \Delta\theta) \bmod 2\pi\).
- **Physical Interpretation:** Implements form‑dependent motion processing (dual‑pathway cognition).

### 4. Softness (Laplacian Diffusion)

**Diffusion Equation:**
\[
\frac{\partial z}{\partial t} = \eta \nabla^2 z
\]
Discretized as:
\[
z_{t+1} = z_t + \eta \cdot (z_t * K_{\text{laplace}})
\]
with kernel:
\[
K_{\text{laplace}} = \begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
\]
- **\(\eta \in [0,1]\):** Softness parameter.
- **Adaptive Softness:** \(\eta_{\text{eff}} = \eta \cdot \min\left(1, \frac{\|\nabla z\|}{0.1}\right)\) to prevent meltdown.

### 5. Resonance Detection

**Detection Metric:**
\[
\delta = \frac{1}{HW} \sum_{x,y} |z_{t}(x,y) - z_{t-1}(x,y)|
\]
- If \(\delta > 0.5\), emergency damping is triggered:
  - \(\eta \leftarrow \eta \cdot \phi\) (increase softness)
  - \(z_t \leftarrow 0.5 z_{t-1} + 0.5 z_t\) (temporal smoothing)

---

## II. Holographic Memory

### 6. Gram–Schmidt Orthogonalization

For a new pattern vector \(\mathbf{p} \in \mathbb{C}^D\) and existing keys \(\{\mathbf{k}_1, \dots, \mathbf{k}_m\}\):
\[
\mathbf{p}_{\perp} = \mathbf{p} - \sum_{j=1}^{m} \frac{\langle \mathbf{p}, \mathbf{k}_j \rangle}{\langle \mathbf{k}_j, \mathbf{k}_j \rangle} \mathbf{k}_j
\]
- **Inner Product (complex):** \(\langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i} a_i \overline{b_i}\)
- **Normalization:** \(\mathbf{p}_{\text{stored}} = \frac{\mathbf{p}_{\perp}}{\|\mathbf{p}_{\perp}\|}\)

### 7. Holographic Recall (Similarity)

**Cosine Similarity (Complex):**
\[
\text{sim}(\mathbf{q}, \mathbf{k}) = \frac{|\langle \mathbf{q}, \mathbf{k} \rangle|}{\|\mathbf{q}\| \cdot \|\mathbf{k}\| + \epsilon}
\]
- Retrieval returns top-\(k\) values by similarity.

### 8. Holographic Reconstruction

**Weighted Sum:**
\[
\mathbf{z}_{\text{recon}} = \frac{\sum_{j} w_j \mathbf{K}_j}{\sum_{j} w_j}
\]
where \(w_j = |\langle \mathbf{q}, \mathbf{k}_j \rangle|\).

---

## III. Evolutionary Optimization

### 9. φ‑Resonant Constants

- Golden Ratio: \(\phi = \frac{1+\sqrt{5}}{2} \approx 1.618034\)
- Inverse: \(1/\phi = \phi - 1 \approx 0.618034\)
- Powers used for scaling weights, intervals, and probabilities.

### 10. Mutation Operator

For continuous parameter \(p\):
\[
p \leftarrow p + \mathcal{N}(0, \sigma^2)
\]
where \(\sigma = (1/\phi^2) \approx 0.382\). For discrete parameters (e.g., `max_iter`), integer perturbation with probability \(\sigma\).

### 11. Fitness Function (Interestingness)

**Composite Score:**
\[
\mathcal{F} = \phi \cdot \text{novelty} + 1.0 \cdot \text{depth} + \frac{1}{\phi} \cdot \text{coherence} + 0.5 \cdot \text{task\_perf} + 0.2 \cdot \text{attention\_div}
\]

- **Novelty:** Binary (1 if pattern hash unseen, else 0).
- **Cognitive Depth (Fractal Dimension Penalty):**
  \[
  \text{depth} = D \cdot \exp\left(-\frac{(D - \phi)^2}{0.5}\right)
  \]
  where \(D\) is estimated fractal dimension.
- **Coherence:** Binary (1 if memory recalls itself, else 0).

### 12. Fractal Dimension Estimation (Box‑Counting)

**Edge Detection:**
\[
E_x = |M_{i+1,j} - M_{i,j}|, \quad E_y = |M_{i,j+1} - M_{i,j}|
\]
where \(M\) is the escaped mask (binary).
\[
\text{edge\_count} = \sum E_x + \sum E_y
\]
\[
D \approx 1 + 2 \cdot \frac{\text{edge\_count}}{\sqrt{HW}}
\]
(Scaled to range \([1,2]\)).

### 13. Novelty Search

**Behavior Vector:**
\[
\mathbf{b} = [\dot{\theta}, \eta, \text{max\_iter}/100, \tau_{\text{attn}}, \bar{w}]
\]
**Novelty Score:**
\[
\nu(\mathbf{b}) = \frac{1}{k} \sum_{j=1}^{k} \|\mathbf{b} - \mathbf{b}^{(j)}\|
\]
where \(\mathbf{b}^{(j)}\) are the \(k=15\) nearest neighbors in the archive.

### 14. Composite Selection

**Combined Fitness:**
\[
\mathcal{F}_{\text{composite}} = \phi \cdot \mathcal{F} + \frac{1}{\phi} \cdot \nu
\]

---

## IV. Attention & Exploration

### 15. Attention Mask

Normalized amplitude:
\[
A(x,y) = \frac{|z(x,y)|}{\max_{(x,y)} |z(x,y)| + \epsilon}
\]

### 16. Attention Diversity (Exploration Bonus)

Variance of attention masks over recent history:
\[
\text{div} = \frac{1}{T} \sum_{t=1}^{T} \text{Var}(A_t)
\]
where \(\text{Var}(A)\) is the spatial variance of the attention map.

### 17. Neglected Regions

Thresholded average attention:
\[
N(x,y) = \begin{cases}
1 & \text{if } \bar{A}(x,y) < \tau \\
0 & \text{otherwise}
\end{cases}
\]
with \(\tau = \text{mean}(\bar{A})\).

---

## V. Encoding & Decoding

### 18. Fourier Encoding

**Pattern Generation:**
\[
P(x,y) = \sum_{k=1}^{K} v_k \cdot e^{2\pi i f_k (x/W + y/H)}
\]
- \(v_k\): Input vector components.
- \(f_k\): Frequency (harmonic index).

### 19. Feature Extraction (Spatial Statistics)

**Feature Vector:**
\[
\mathbf{f} = \left[ \mu_{|z|}, \sigma_{|z|}, \mu_{\text{esc}}, P_{90}, P_{10} \right]
\]
where \(\mu, \sigma\) are mean/std of amplitude, \(\mu_{\text{esc}}\) is fraction escaped, \(P_{90}, P_{10}\) are percentiles.

### 20. Linear Classification

**Logits:**
\[
\mathbf{l} = \mathbf{W} \cdot \mathbf{f}
\]
- \(\mathbf{W} \in \mathbb{R}^{C \times 5}\): Learnable weight matrix (not evolved in current version).

---

## VI. Numerical & Implementation Mathematics

### 21. Precision Management

- **Complex64:** 23‑bit mantissa → sufficient for `max_iter ≤ 100`.
- **Complex128:** 52‑bit mantissa → used for deep iterations (`max_iter > 100`).

### 22. JIT Compilation & Cache Clearing

- **Interval:** \(\lfloor \phi^5 \rfloor = 11\) generations.
- **Function:** `jax._src.lib.xla_bridge.get_backend().clear_caches()`

### 23. vmap Batching

Parallel evaluation of population via:
\[
\text{batch\_step} = \text{jax.vmap}(\text{cognitive\_step}, \text{in\_axes}=(0, \text{None}))
\]
applied to stacked field tensors of shape \((\text{pop\_size}, H, W)\).

---

## VII. Summary Table

| Domain | Mathematical Tools |
|:---|:---|
| **Complex Dynamics** | Mandelbrot/Julia iteration, escape radius, complex arithmetic |
| **Field Theory** | Laplacian diffusion, rotation operator, phase modulation |
| **Functional Analysis** | Inner products, norms, Gram–Schmidt orthogonalization |
| **Fractal Geometry** | Box‑counting dimension estimation |
| **Information Theory** | Novelty via pattern hashing, attention entropy |
| **Evolutionary Computation** | Gaussian mutation, tournament selection, novelty search, φ‑weighted composite fitness |
| **Linear Algebra** | Eigen‑decomposition (via FFT), cosine similarity, matrix convolutions |
| **Numerical Analysis** | Floating‑point precision, JIT compilation, gradient‑free optimization |
| **Probability & Statistics** | Gaussian noise, variance, percentiles, mean, standard deviation |

This mathematical inventory demonstrates that the Fractal AGI Cognitive Core is a deeply interdisciplinary system, synthesizing concepts from complex dynamics, holography, evolutionary algorithms, and information geometry into a unified computational framework for artificial cognition.
