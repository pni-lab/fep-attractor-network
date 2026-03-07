---
title: "Rebuttal letter"
exports:
  - format: docx
    output: autoconvert-rebuttal-letter.docx
  - format: pdf
    output: autoconvert-rebuttal-letter.pdf
  - format: tex
    template: arxiv_nips
    output: latex/rebuttal_letter

abbreviations:
  FEP: Free Energy Principle
  VFE: Variational Free Energy
  NESS: Non-Equilibrium Steady-State
  RNN: Recurrent Neural Network

---


## *Manuscript NEUCOM-D-25-06426*

Dear Editor and Reviewers,

We sincerely thank you for the constructive and insightful feedback on our manuscript, *Self-orthogonalizing attractor neural networks emerging from the free energy principle*. We carefully revised the manuscript and appendix to improve biological grounding, accessibility, empirical context, and discussion of scalability and limitations. We also prepared additional simulations to address comments on scaling,computational efficiency and more realistic data.
Below we provide a point-by-point responses to all reviewer comments.

We shall look forward to hearing from you.

Yours sincerely,

Tamas Spisak and Karl Friston

------

## Reviewer #2

### Comment 1
**The framework claims biological plausibility; further discussion is needed on how the derived learning rules (e.g. Hebbian/anti-Hebbian) map to known neurobiological mechanisms.**

**Response**  
We thank the reviewer for this important suggestion. We now provide a more detailed discussion of the neurobiological correspondence of the derived plasticity rule, quoted below for your convenience:

> The neurobiological mechanisms of synaptic plasticity are best understood and experimentally most thoroughly studied at the level of single neurons, although efforts exist to scale up these findings to the level of populations of neurons (e.g., {cite:t}`10.1016/j.jtbi.2011.06.023; https://doi.org/10.1016/j.compbiomed.2023.107213`). We believe that our framework holds promise for a better understanding of plasticity mechanisms independent of scale, as it mathematically survives arbitrary coarse-graining under the deep particular partition formalism.
>At the single-neuron limit, the rule reduces to a discrete-time binary Hebbian/anti-Hebbian update (formally recovered when precision → ∞), closely resembling spike-timing–dependent plasticity (STDP)  {cite:p}`10.3389/fncom.2010.00156; https://doi.org/10.1038/s42003-024-06203-8`, where correlated activity produces long-term potentiation (LTP ~ Hebbian term) and predicted activity leads to long-term depression (LTD ~ anti-Hebbian term; see e.g. {cite:t}`10.1146/annurev.neuro.31.060407.125639`). Thereby, our framework connects STDP to predictive coding {cite:p}`10.1038/s41467-023-40651-w`, in a sense that presynaptic activity that is reliably predicted by postsynaptic firing is eventually depressed. 
> Moreover, the subtractive predictive error term, together with the bounded nature of the continuous Bernoulli distribution, prevent runaway potentiation, functioning analogously to homeostatic and metaplasticity mechanisms {cite:p}`10.1101/cshperspect.a005736; https://doi.org/10.1016/j.conb.2017.03.015`.
> In sum, rather than fitting neuron-level data post hoc, our framework predicts that Hebbian/anti-Hebbian-style updates should appear at all descriptive scales, with differences only in implementation, not mathematical structure. The biological literature at synaptic, circuit, and network levels is largely consistent with this multiscale interpretation.

---

### Comment 2
**The simulations focus on small-scale networks. Scaling to larger networks or real-world datasets would strengthen practical relevance.**

**Response**  
We fully agree that scaling evidence strengthens practical relevance. The original reference implementation prioritized transparency of the local update rules over speed; we retain it for all core analyses (Simulations 1–4). To address scalability, we now provide three new contributions: (i) a parallelized JAX implementation for runtime profiling, (ii) systematic scaling experiments (up to 2.5B parameters), and (iii) a new simulation on a more realistic dataset. We also added a discussion paragraph that identifies the sequential update bottleneck and outlines paths to further scaling (dedicated hardware and weight sparsity). The analytical complexity comparison (per-step cost, memory capacity bounds) is detailed in response to Reviewer #3 Comment 2 ([](#appendix-8)); here we focus on the empirical evidence.

**New methods paragraph (lines `XX–XX`):**

> In addition to the reference Python implementation used for Simulations 1–4, we provide a vectorized JAX implementation that applies the same inference and learning rules (eq. [](#inference-rule), [](#learning-rule)) in a parallelized, full-network update schedule ([](07-simulation-scaling-jax.ipynb)). This parallel schedule is computationally more efficient but not strictly equivalent to the sequential node-by-node updates of the reference implementation; we validate that both implementations produce qualitatively similar coupling matrices and retention behavior on shared test cases. The JAX implementation is used exclusively for runtime profiling (Simulation 5) and for the larger-scale face recognition experiment (Simulation 6). All core qualitative results (orthogonalization regimes, sequence learning, catastrophic forgetting) are reported using the reference implementation.

**Revised Simulation 5 paragraph (lines `XX–XX`):**

> **Simulation 5: scalability profile and memory capacity**
>
> We profile the computational scaling of the JAX implementation across network sizes ranging from $N = 64$ to $N = 50000$ nodes, i.e up to 2.5B total parameters ([](07-simulation-scaling-jax.ipynb)). Training and inference runtimes scale as expected from the analytical $O(N^2)$ per-step complexity derived in [](#appendix-8) ([](#fig-scaling)A). Throughput remains high for moderate network sizes and drops off as expected for very large $N$ ([](#fig-scaling)B). We further probe memory capacity by varying the number of stored patterns $K$ at fixed network size, measuring both retention (deterministic attractor recovery, [](#fig-scaling)C) and noisy reconstruction quality ([](#fig-scaling)D). Consistent with the analytical prediction that emergent orthogonalization approaches the projector-network capacity limit ([](#appendix-8)), we observe that the network maintains high-fidelity attractors and effective Bayesian retrieval for pattern counts substantially exceeding the classical Hopfield bound of $\sim 0.14N$.

**New Simulation 6 paragraph (lines `XX–XX`):**

> **Simulation 6: face recognition with the Olivetti faces dataset**
>
> To move beyond the small-scale handwritten digit experiments, we test the framework on the Olivetti faces dataset — a benchmark consisting of 400 grayscale face images (40 subjects, 10 images each) at full resolution $64 \times 64$ pixels ($N = 4096$ nodes, more than 16 million parameters), using the full set of 400 patterns ([](08-simulation-faces-jax.ipynb)). Training confirms that the key phenomena observed on handwritten digits — emergent orthogonalization and attractor formation — transfer to this substantially higher-dimensional and more naturalistic stimulus domain ([](#fig-scaling)E). We further evaluate stochastic reconstruction from heavily corrupted input. As shown on [](#fig-scaling)F, the network performs Bayesian inference by integrating the noisy sensory evidence (likelihood) with its learned prior beliefs (attractors), with the balance controlled by the precision (inverse temperature) parameter. Notably, the input noise level used here is severe enough that the corrupted faces approach the limit of human recognizability, yet the network reliably recovers the identity of the original face. The combined scaling, capacity, and face recognition results are shown in [](#fig-scaling).

::: {figure} fig/scaling.png
:name: fig-scaling
:width: 100%
**Scaling, memory capacity, and face recognition.** 
**A–B**: Runtime (A) and throughput (B) of the JAX implementation as a function of network size $N$, confirming the expected $O(N^2)$ per-step scaling ([](07-simulation-scaling-jax.ipynb)).
**C**: Deterministic retrieval quality (retention correlation, blue) as a function of the number of stored patterns $K$, compared with the maximum inter-pattern self-correlation baseline (red). The network maintains high-fidelity attractors well beyond the classical Hopfield capacity bound.
**D**: Stochastic (Bayesian) reconstruction: correlation of the reconstructed output with the original pattern (blue) versus correlation of the noisy input with the original (red), as a function of $K$. The network consistently improves upon the noisy input, demonstrating effective Bayesian retrieval.
**E**: Training the network on the Olivetti faces dataset ($64 \times 64$, $N = 4096$, 400 patterns; [](08-simulation-faces-jax.ipynb)). Random examples of training faces (left) and the corresponding learned attractors (right), confirming attractor formation and orthogonalization in this naturalistic domain.
**F**: Bayesian face reconstruction from noisy input. Columns show reconstructions at increasing likelihood precision (left to right, top row) and increasing prior precision (top to bottom). Even when the input is degraded to a level approaching the limit of human face recognition, the network recovers the original identity by combining sensory evidence with its learned attractor-based priors.
:::

**New discussion paragraph on scaling prospects (lines `XX–XX`):**

> A key computational bottleneck is the sequential dependency of the Gibbs-like node-by-node inference and learning: each node's new state depends on the current states of all others, preventing straightforward parallelization. Our synchronous (full-network) JAX implementation sidesteps this by applying all updates simultaneously, yielding favorable empirical scaling up to $N = 4096$ (Simulation 5); however, this parallel schedule is an approximation whose fidelity at very large $N$ warrants further study. To efficiently scale the framework towards billions of neurons two complementary directions appear most promising: (i) dedicated hardware — in particular thermodynamic computers {cite:p}`https://doi.org/10.1038/s41467-025-59011-x` and memristive substrates {cite:p}`10.3390/math11061369; 10.1109/JIOT.2024.3409373` — which can natively implement the local, stochastic update rules without the sequential bottleneck; and (ii) structured weight sparsity, which would reduce the per-step cost from $O(N^2)$ to $O(Ns)$ (where $s \ll N$ is the average number of non-zero connections per node) and simultaneously lower the memory footprint.

---

### Comment 3
**The manuscript is well-written but occasionally dense. It is suggested to add a high-level schematic and simplify derivations where possible.**

**Response**  
We appreciate this feedback and agree. We made three changes to improve accessibility. First, we added an intuition-first high-level pipeline schematic ([](#fig-framework-overview)). Second, we substantially simplified the derivation of the inference and learning rules. The inference section (lines `XX–XX`) now uses a compact local VFE argument — three equations from the local evidence $\theta_i$ to the inference rule — rather than the longer accuracy–complexity decomposition, which is retained in the appendix ([](#appendix-6)) for interested readers. The learning rule derivation (lines `XX–XX`) is similarly streamlined via the same local VFE. Third, we added a new intuition-focused subsection ([](#self-solenoidization), lines `XX–XX`) that explains the emergence of solenoidal flows from asymmetric couplings — using a dissipative-plus-rotational decomposition — without requiring a formal proof of the divergence-free condition.

::: {figure} fig/schematics.png
:name: fig-framework-overview
:width: 90%
**Overview of the framework.** Starting from universal and parsimonious structural assumptions (deep particular partition), variational free energy minimization under the Free Energy Principle (FEP) gives rise to emergent dynamics at multiple scales: local stochastic inference and learning rules that guide the update of internal states and the couplings between them, self-organizing attractor dynamics at the macroscale, and — at the computational level — Bayesian inference, approximately orthogonal attractor representations of latent external causes, and sequence learning capabilities.
:::


## Reviewer #3

### Comment 1
**Discuss how orthogonalized attractor representations and free-energy minimization affect robustness to adversarial perturbations, noise, or data poisoning.**

**Response**  
We thank the reviewer for raising this stimulating point — thinking about the framework through the lens of adversarial robustness is a genuinely interesting perspective. We now address it in a new discussion subsection (lines `XX–XX`) and a supporting mathematical appendix ([](#appendix-7)).

**New discussion subsection (lines `XX–XX`):**

> A natural question is whether the structural properties of FEP-based self-orthogonalizing attractor networks confer robustness to adversarial perturbations, noise corruption, or data poisoning — topics of growing importance in secure and trustworthy AI {cite:p}`https://doi.org/10.48550/arXiv.1412.6572`. Within the Bayesian framing of the network (eq. [](#posterior-distribution)), adversarial perturbations and data poisoning correspond to attacks at two distinct levels: sensory perturbations (including adversarial examples) enter as erroneous bias shifts $\delta\mathbf{s}$ during inference (corrupted likelihood), whereas data poisoning distorts the learned prior $p(\boldsymbol{\sigma})$, i.e. the attractor landscape itself. The inverse-temperature parameter $iT$ mediates the balance between these levels: high precision deepens attractor basins relative to any bias perturbation, so the magnitude of $\delta\mathbf{s}$ becomes small compared to the basin depth; low precision, conversely, flattens the prior landscape and lets the actual sensory evidence dominate, mitigating the effect of distorted or poisoned attractors. This precision-mediated trade-off is a direct consequence of the Bayesian posterior structure and has no counterpart in deterministic Hopfield-type retrieval (see [](#appendix-7) for formal analysis).
> Furthermore, our framework naturally distinguishes between two types of redundancy. Self-orthogonalization minimizes *representational redundancy* (attractor overlap) which maximizes inter-attractor distances and thereby the "adversarial budget", i.e. the minimum perturbation energy needed to push the network across a basin boundary ([](#appendix-7)). On the other hand, the network maintains high *structural redundancy*: the $M<N$ attractors are supported by $O(N^2)$ distributed weights, so bounded weight corruption or node failure has limited impact on the energy landscape.

The formal analysis is provided in a new appendix ([](#appendix-7)), which derives how attractor orthogonality maximizes the adversarial budget (minimum perturbation norm to cross a basin boundary), how the precision parameter $iT$ mediates the trade-off between resilience to sensory attacks and data poisoning, and how stochastic MCMC averaging and the anti-Hebbian learning term provide additional implicit regularization.

---

### Comment 2
**Include clearer empirical or analytical comparisons with established models (e.g., Hopfield networks, Boltzmann Machines, predictive coding models).**

**Response**  
We agree that a structured comparison strengthens the contribution. We added a short discussion paragraph (lines `XX–XX`) pointing to a detailed analytical appendix ([](#appendix-8)) that includes a comparison table and formal complexity and mixing-time analyses.

**New discussion paragraph (lines `XX–XX`):**

> Relative to classical single-layer Hopfield/Boltzmann formulations, the present framework preserves attractor-based inference while extending the model class along several dimensions that emerge from the FEP derivation rather than being imposed by design. These extensions — and their analytical consequences — are summarized in [](#tab-comparison) and elaborated in [](#appendix-8). Two points deserve emphasis here. First, the learning rule (eq. [](#learning-rule)) eliminates the costly free-running phase of contrastive divergence, reducing per-step learning complexity from $O(N^2 k)$ to $O(N^2)$ ([](#appendix-8)). Second, in the presence of asymmetric couplings the solenoidal component of the dynamics induces non-reversible probability currents that accelerate mixing relative to symmetric Boltzmann machines — a mechanism formally analogous to continuous normalizing flows {cite:p}`https://doi.org/10.48550/arXiv.2302.00482` and known to reduce mixing times {cite:p}`10.1088/0305-4470/37/3/L01; https://doi.org/10.1088/1751-8113/43/37/375003` ([](#appendix-8)). The relationship to hierarchical predictive coding — which minimizes the same objective in a bidirectional hierarchy rather than a fully recurrent topology — is discussed in [](#appendix-8) and constitutes a promising direction for future work.

The detailed analytical comparison is provided in a new appendix ([](#appendix-8)), covering learning-rule complexity (elimination of the free-running phase, reducing per-step cost from $O(N^2 k)$ to $O(N^2)$), solenoidal flows and mixing-time acceleration, progressive orthogonalization and memory capacity approaching the projector-network limit, and the relationship to hierarchical predictive coding. The summary comparison table is reproduced below for convenience:

> :::{table} Comparison of the FEP-ANN with canonical formulations of classical models.
> :name: tab-comparison
>
> | Feature | Hopfield | Boltzmann machine | Projector network | Predictive coding | **FEP-ANN (ours)** |
> |---|---|---|---|---|---|
> | **Derivation** | Energy-based | Statistical mechanics | Optimal storage | Hierarchical Bayesian | First principles (FEP) |
> | **State space** | Binary $\{-1,+1\}$ | Binary $\{0,1\}$ | Binary | Continuous (Gaussian) | Continuous $[-1,+1]$ (CB) |
> | **Activation** | Sign | Logistic sigmoid | Sign | Linear / nonlinear | Langevin (emergent) |
> | **Learning** | Hebbian (batch) | Contrastive divergence | Pseudo-inverse (batch) | Prediction-error min. | Hebbian/anti-Hebbian (online) |
> | **Learning phases** | One-shot | Clamped + free | One-shot | Layerwise | Single phase |
> | **Coupling** | Symmetric | Symmetric | Symmetric | Directed (hierarchy) | Symmetric or asymmetric |
> | **Sequence dynamics** | No | No | No | Temporal hierarchy | NESS solenoidal flow |
> | **Orthogonalization** | No | No | By construction | No | Emergent (VFE) |
> | **Memory capacity** | $\sim 0.14N$ | $\sim 0.14N$ | $N$ (optimal) | N/A (generative) | Approaches $N$ |
> | **Inference** | Deterministic | Gibbs MCMC | Deterministic | Message passing | MCMC + solenoidal flow |
> | **Precision control** | None | Temperature (fixed) | None | Precision-weighting | $iT$ (Bayesian) |
> | **Continual learning** | Catastrophic forgetting | Catastrophic forgetting | Catastrophic forgetting | Possible (with replay) | Built-in (spontaneous replay) |
> | **Learning complexity** | $O(KN^2)$ one-shot | $O(N^2 k)$ per step | $O(KN^2 + N^3)$ one-shot | $O(N_l)$ per layer per step | $O(N^2)$ per step |
> | **FEP / active inference** | No formal link | Variational (post hoc) | No formal link | Compatible | Derived from FEP |
>
> :::

---

### Comment 3
**Address computational scalability and potential limitations for high-dimensional or real-world data streams.**

**Response**  
We now address scalability both analytically and empirically. The analytical treatment — per-step complexity, memory capacity bounds, and mixing-time considerations — is provided in [](#appendix-8) (see also our response to Reviewer #3 Comment 2). The empirical scaling evidence — runtime profiling across network sizes, memory capacity sweeps, and a new face recognition experiment — is detailed in our response to Reviewer #2 Comment 2. We also added a new discussion paragraph (quoted in Reviewer #2 Comment 2) that identifies the sequential update dependency as the principal bottleneck and outlines two concrete paths to scale.

**Revised discussion paragraph (lines `XX–XX`):**

> The per-step computational cost of the FEP-based attractor network is $O(N^2)$ for a full-network update — dominated by one matrix-vector product for inference and one outer-product update for learning — with no free-running phase required (unlike contrastive divergence in Boltzmann machines; see [](#appendix-8)). This complexity is confirmed empirically: runtime scales quadratically with $N$ across network sizes from 64 to 4096 nodes ([](07-simulation-scaling-jax.ipynb), [](#appendix-9)). Memory capacity benefits from emergent orthogonalization, which progressively approaches the projector-network limit of $K_{\max} = N$, substantially exceeding the classical Hopfield bound of $\sim 0.14N$ ([](07-simulation-scaling-jax.ipynb)). We further validated the framework on the Olivetti faces dataset at full resolution ($N = 4096$, 400 patterns; [](08-simulation-faces-jax.ipynb)), confirming that orthogonalization, Bayesian retrieval, and generalization transfer to more naturalistic stimuli.
>
> Several limitations warrant explicit acknowledgment. First, the $O(N^2)$ memory footprint for the full weight matrix becomes prohibitive for very large $N$; sparse or factored weight parametrizations are a natural extension but remain unexplored. Second, while solenoidal flows from asymmetric couplings are expected to accelerate mixing ([](#appendix-8)), a rigorous characterization of mixing times as a function of network size, pattern count, and coupling asymmetry is needed. Third, the current demonstrations use stationary or slowly varying input statistics; performance under strongly non-stationary, streaming real-world data — where the input distribution shifts faster than the learning rate can track — remains an open question. Fourth, all simulations use a single-layer architecture; scaling to hierarchical (multi-layer) deep particular partitions, which could support richer generative models, is an important theoretical and practical frontier.

A new appendix ([](#appendix-9)) reports runtime benchmarks (see [](#tab-runtimes)) and memory capacity results for the JAX implementation across network sizes from $N = 100$ to $N = 50\,000$. The combined scaling, capacity, and face recognition results are presented in [](#fig-scaling).

---

### Comment 4
**Discuss newer neural-network types such as multiwing hyperchaotic memristive neural networks and memristive CNNs with multi-butterfly attractors.**

**Response**  
We thank the reviewer for this suggestion. We have now extended the existing discussion paragraph on thermodynamic and neuromorphic computing to include memristive and hyperchaotic attractor-network families. These systems demonstrate that recurrent, state-dependent couplings can generate rich attractor repertoires (multistability, hyperchaos, coexisting attractors) in hardware substrates whose physics naturally supplies the stochastic fluctuations that our framework requires. Our contribution is orthogonal in emphasis: we derive the architecture from first principles (FEP and deep particular partitions), rather than prescribing a phenomenological attractor family a priori.
Below we quote the revised passage (lines `XX–XX`).

**Revised passage (lines `XX–XX`):**

> Stochasticity - a key property of our network - is also very relevant from the perspective of artificial intelligence research. In our framework, noise is not an enemy; it implements the precision of inference, allowing it to strike a balance between stability and flexibility. This inherent stochasticity yields an exceptional fit with energy-efficient neuromorphic architectures {cite:p}`https://doi.org/10.1038/s43588-021-00184-y`, particularly within the emerging field of thermodynamic computing {cite:p}`https://doi.org/10.1038/s41467-025-59011-x` and memristive technologies, where attractor networks — including memristive Hopfield networks, multi-wing and multi-butterfly hyperchaotic constructions, and memristive cellular-neural-network-like circuits {cite:p}`10.3390/math11061369; 10.1140/epjp/s13360-023-04772-x; 10.1109/JIOT.2024.3409373; 10.1007/s11071-025-10982-y; 10.1016/j.chaos.2024.115526` have recently been studied. A promising direction is to ask whether local VFE-minimizing updates can be instantiated in thermodynamic and memristive substrates, combining the principled inferential interpretation of the FEP framework with the hardware efficiency and rich non-equilibrium dynamics in these emerging paradigms.

**Revised passage (lines `XX–XX`):**
>To efficiently scale the framework towards billions of neurons two complementary directions appear most promising: (i) dedicated hardware — in particular thermodynamic computers {cite:p}`https://doi.org/10.1038/s41467-025-59011-x` and memristive substrates {cite:p}`10.3390/math11061369; 10.1109/JIOT.2024.3409373` — which can natively implement the local, stochastic update rules without the sequential bottleneck; and (ii) structured weight sparsity, which would reduce the per-step cost from $O(N^2)$ to $O(Ns)$ (where $s \ll N$ is the average number of non-zero connections per node) and simultaneously lower the memory footprint.

---

### Comment 5
**While mathematically elegant, some concepts (e.g., universal partitioning of random dynamical systems) would benefit from additional intuitive explanation.**

**Response**  
We appreciate this recommendation and agree. We made several targeted changes to improve intuitive accessibility without sacrificing mathematical precision:

1. **High-level schematic** ([](#fig-framework-overview)): A new pipeline figure shows the logical flow from structural assumptions (deep particular partition) through local FEP dynamics to emergent computational properties (see also Reviewer #2 Comment 3).

2. **Universal partitioning of random dynamical systems:** We added a short intuition sentence in the introduction (lines `XX–XX`): universal partitioning means that any persistent random dynamical system can be read as performing inference at its boundaries — i.e., nested interfaces separate what is inferred from what is inferred about, so the system has a natural interpretation as an inference engine without ad hoc modeling devices.

3. **Simplified derivations** (lines `XX–XX`): We restructured the main results to emphasize the local evidence form $\theta_i = b_i + \sum_{j \neq i} J_{ij} \sigma_j$, from which both the inference rule and the learning rule follow in a few lines via local VFE minimization. The longer accuracy–complexity decomposition is now in [](#appendix-6). Crucially, the local derivation no longer requires the global Boltzmann joint as an intermediate step — it proceeds directly from the deep particular partition structure to the update rules.

4. **Self-solenoidization intuition** ([](#self-solenoidization), lines `XX–XX`): We added a new subsection that explains how asymmetric couplings induce solenoidal flows using an intuitive dissipative-plus-rotational decomposition. The key insight — that the antisymmetric component drives probability currents along iso-energy contours, like a ball circling within a bowl rather than rolling to the bottom — is presented before the formal conditions, making the concept accessible without full technical detail.

5. **Reframed global joint** (lines `XX–XX`): The Boltzmann-like stationary distribution is now introduced as a *consequence* of the local pairwise structure ("As a global consequence of the local pairwise structure..."), making clear that it is a derived property rather than an assumption.

We believe these changes substantially lower the entry barrier while preserving the full mathematical content in the appendix for readers who want it.
