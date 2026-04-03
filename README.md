## *Manuscript files and analysis source code for the paper entitled:*
# Self-orthogonalizing attractor neural networks emerging from the free energy principle
*by Tamas Spisak and Karl Friston*

### 🌐 Webpage: https://pni-lab.github.io/fep-attractor-network
### 📄 Paper: https://doi.org/10.1016/j.neucom.2026.133472
*Spisak T, Friston K. Self-orthogonalizing attractor neural networks emerging from the free energy principle. Neurocomputing. 2026 Mar 28:133472.*
### <img src="https://github.com/user-attachments/assets/a5fcb946-8c28-488f-bdfc-551ad0221a49" alt="ArXiv" width="25"/> Preprint: [arXiv:2505.22749v1 ](https://arxiv.org/abs/2505.22749)

### Highlights
- Attractor networks are derived from the Free Energy Principle (FEP)
- Giving rise to a biologically plausible, scale-independnet predictive coding-based local plasticity rule.
- Learning results in approximately orthogonal attractor representations, balancing predictive accuracy and model complexity.
- Sequential data leads to asymmetric couplings, generalizing Boltzmann Machines to non-equilibrium steady-state dynamics.
- Simulations demonstrate self-orthogonalization, sequence learning, scalability and resistance to catastrophic forgetting.

### 📄 Abstract:
Attractor dynamics are a hallmark of many complex systems, including the brain. Understanding how such self-organizing dynamics emerge from first principles is crucial for advancing our understanding of neuronal computations and the design of artificial intelligence systems. Here we formalise how attractor networks emerge from the free energy principle applied to a universal partitioning of random dynamical systems. Our approach obviates the need for explicitly imposed learning and inference rules and identifies emergent, but efficient and biologically plausible inference and learning dynamics for such self-organizing systems. These result in a collective, multi-level Bayesian active inference process. Attractors on the free energy landscape encode prior beliefs; inference integrates sensory data into posterior beliefs; and learning fine-tunes couplings to minimize long-term surprise. Analytically and via simulations, we establish that the proposed networks favor approximately orthogonalized attractor representations, a consequence of simultaneously optimizing predictive accuracy and model complexity. These attractors efficiently span the input subspace, enhancing generalization and the mutual information between hidden causes and observable effects. Furthermore, while random data presentation leads to symmetric and sparse couplings, sequential data fosters asymmetric couplings and non-equilibrium steady-state dynamics, offering a natural generalization of conventional Boltzmann Machines. Our findings offer a unifying theory of self-organizing attractor networks, providing novel insights for AI and neuroscience.

### ℹ️ Repository Info:

- Manuscript text is written in [Myst Markdown](https://mystmd.org/), see rendered webpage [here](https://pni-lab.github.io/fep-attractor-network).
- Analysis code is written in Python (v3.12.5).
- The repository contains two different implementations for FEP-based self-orthogonalizing attarctor networks:
    - [simulation/network.py](simulation/network.py): favors clarity over efficiency (it implements both σ and boundary states as separate classes), sequential update, not optimized for performance.
    - [simulation/network_jax.py](simulation/network_jax.py): a fast JAX-based implementation with parallel updates.
- See [requirements.txt](requirements.txt) for requirements.

### <img width="25" alt="image" src="https://github.com/user-attachments/assets/7405602a-2342-4838-83b7-553f520c59ba" /> Cite
```bibtex
@article{SPISAK2026133472,
title = {Self-orthogonalizing attractor neural networks emerging from the free energy principle},
journal = {Neurocomputing},
volume = {682},
pages = {133472},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.133472},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226008696},
author = {Tamas Spisak and Karl Friston}
}
```
