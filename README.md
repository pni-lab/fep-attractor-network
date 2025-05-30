## *Manuscript files and analysis source code for the paper entitled:*
# Self-orthogonalizing attractor neural networks emerging from the free energy principle
*by Tamas Spisak and Karl Friston*

### 🌐 Webpage: https://pni-lab.github.io/fep-attractor-network
### <img src="https://github.com/user-attachments/assets/a5fcb946-8c28-488f-bdfc-551ad0221a49" alt="ArXiv" width="25"/> Preprint: [arXiv:2505.22749v1 ](https://arxiv.org/abs/2505.22749)

### 📄 Abstract:
Attractor dynamics are a hallmark of many complex systems, including the brain. Understanding how such self-organizing dynamics emerge from first principles is crucial for advancing our understanding of neuronal computations and the design of artificial intelligence systems. Here we formalize how attractor networks emerge from the free energy principle applied to a universal partitioning of random dynamical systems. Our approach obviates the need for explicitly imposed learning and inference rules and identifies emergent, but efficient and biologically plausible inference and learning dynamics for such self-organizing systems. These result in a collective, multi-level Bayesian active inference process. Attractors on the free energy landscape encode prior beliefs; updates integrate sensory data into posterior beliefs; and learning fine-tunes couplings to minimize long-term surprise. Analytically and via simulations, we establish that the proposed networks favor approximately orthogonalized attractor representations, a consequence of simultaneously optimizing predictive accuracy and model complexity. These attractors efficiently span the input subspace, enhancing generalization. Furthermore, while random data presentation leads to symmetric and sparse couplings, sequential data fosters asymmetric couplings and non-equilibrium steady-state dynamics, offering a natural extension to conventional Boltzmann Machines. Our findings offer a unifying theory of self-organizing attractor networks, providing novel insights for AI and neuroscience.

### ℹ️ Repository Info:

- Manuscript text is written in [Myst Markdown](https://mystmd.org/)
- Analysis code is written in Python (v3.12.5).
- The present implementation for FEP-based self-orthogonalizing attarctor networks favors clarity over efficiency (it implements both σ and boundary states as separate classes, and is not optimized for performance). 
- See [requirements.txt](requirements.txt) for requirements.
