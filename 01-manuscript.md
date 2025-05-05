---
title: "Attractor neural networks and the free energy principle"
exports:
  - format: docx
    output: autoconvert-attractor-math-manuscript.docx
  - format: pdf
    output: autoconvert-attractor-math-manuscript.pdf

abbreviations:
  RMB: Restricted Boltzmann Machine
  FEP: Free Energy Principle
  PDF: Probability Density Function
  BCI: Bayesian Causal Inference
---

%+++ {"part": "key-points"}
%**Key Points:**
%- todo
%+++


+++ {"part": "abstract"}
Attractor neural networks offer a robust computational framework, where recurrent dynamics have a natural tendency to converge to stable states. Attractor dynamics are a hallmark of many natural systems, including the brain, showcasing not only the computational capabilities of attractor networks but also their ability to emerge naturally through self-organization.
In our work, we show that a specific class of attractor networks emerges directly from first principles - sparse coupling and free energy minimization in far-from-equilibrium systems - without the need for explicitly imposed learning or inference rules. Minimizing variational free energy yields a Hopfield-style update mechanism that creates a self-organizing, continuous-state stochastic variety of Hopfiled networks. Minimizing expected free energy, on the other hand, manifests in a generalized Hebbian learning process that adjusts the coupling between the elements of the system by contrasting sensory inputs with generative model predictions. This ensures that the system develops attractors which are not mere replicas of the observed sensory patterns but optimized, orthogonalized variants, enhancing computational performance. 
Our findings underscore the potential of attractor neural networks as a unifying framework for understanding self-organization in complex systems and highlight promising applications in artificial intelligence.
+++

# Introduction

From whirlpools and bird flocks to neuronal and social networks, countless of natural systems can be characterized by dynamics organized around stable states, the so-called *attractor states*. Such systems can be decomposed into a collection of - less or more complex - building blocks or "particles" (e.g. water molecules, birds, neurons, or people), which are coupled through local interactions. Attractors are an emergent consequence of the collective dynamics of the system, that arise from these local interactions, without any individual particles having global control over them.

Attractors are a key concept in dynamical systems theory, where they are defined as a set of states in the phase space of the system to which nearby trajectories converge. Geometrically, attractors can be any set of states, like fixed points, lines, planes, limit cycles, or strange attractors associated with chaotic behavior. 

In artificial attractor neural networks, such as Hopfield networks and Reservoir Computing models, the attractor state is commonly considered as the outcome of computation. For example, Hopfield network's relaxation prcess to a an attractor is long been considered as a form of associative retrieval. Artificial attractor networks come in a variety of forms. The way we choose inference and learning rules greatly influences the network's characteristics - so much so that, for instance, even the same update rule can yield different behaviors when applied sequentially across the network nodes, rather than in parallel. 
Self-organization is a key feature of attractor networks, as the stable states emerge from the interactions between network elements without explicit external programming. This property makes them particularly relevant as models for self-organizing biological systems, including the brain.
It's clear that the brain is also a complex attractor network. Attractor dynamics have long been proposed to play a significant role in information integration at the circuit level (Freeman (1987); Amit (1989); Tsodyks (1999); Deco & Rolls (2003)) and have become established models for canonical brain circuits involved in motor control, sensory amplification, motion integration, evidence integration, memory, decision-making, and spatial navigation (see Khona & Fiete (2022) for a review). For instance, the activity of head direction cells, neurons that fire in a direction-dependent manner, are known to arise from a circular attractor state, produced by a so-called ring attractor network [refs]. Multi- and metastable attractor dynamics in neural circuits have also been proposed to extend to the meso- and macro-scales in the brain Rolls, 2009, "accommodating the coordination of heterogeneous elements" Kelso, 2012, rendering attractor dynamics an overarching computational mechanism across different scales of brain function.
The brain, as an instance of complex attractor networks, showcases not only the computational capabilities of these network architecture but also its ability to emerge and evolve through self-organization. 

At this point, we must differentiate between two crucial levels of self-organization in attractor networks. First, we will refer to a network that, once formed, exhibits self-organizing (attractor) dynamics during its operation as *operational self-organization*. This however does not encompass the network's ability to "build itself" – to emerge from simple, local interactions without explicit programming or global control, and to adaptively evolve its structure and function through learning. This latter level of self-organization is what we will refer to as *adaptive self-organization*. While all attractor network models can be considered as self-organizing in the *operational* sense, the crucial question is which variants are truly consistent with *adaptive self-organization*.

Such architectures would mirror the nervous system's capacity to not just function as an attractor network, but to become and remain one through a self-directed process of development and learning. Further, adaptive self-organization would also be a highly desirable property for robotics and artificial intelligence systems, not only boosting their robustness and adaptability, but potentially leading to systems that can increase their complexity and capabilities organically over time (e.g. developmental robotics).

Therefore, investigating adaptive self-organization in attractor networks is vital for advancing our understanding of the brain and for creating more autonomous, adaptable, brain-inspired AI systems.

The Free Energy Principle (FEP) offers a general framework to study self-organizing inference processes. The FEP has been pivotal in connecting the dynamics of complex self-organizing systems with computational and inferential processes, especially within the realms of brain function {cite:p}`https://doi.org/10.1371/journal.pone.0006421; https://doi.org/10.7551/mitpress/12441.001.0001; https://doi.org/10.1016/j.neubiorev.2016.06.022; https://doi.org/10.1016/j.biopsycho.2023.108741`. The FEP posits that any 'thing', in order to exists, must maintain conditional independence from its environment by establishing a boundary - the Markov blanket - that partitions the system into internal, external, and boundary (sensory and active) states. Maintaining this sparse coupling, also known as a *particular partition*, is equivalent to executing an inference process, where internal states deduce the causes of sensory inputs by minimizing variational free energy.

Here, we describe a specific class of self-organizing attractor networks that emerge directly from the FEP, without the need for explicitly imposed learning or inference rules. The emergence of these networks has a single main requirement: a persistent sparse coupling among the states (particular partitions) of far-from-equilibrium random dynamical systems. 
First, we show that a hierarchical formulation of *particular partitions* - a concept that is applicable to any complex dynamical system - can give rise to systems that have the same functional form as well-known artificial attractor network architectures. 
Second, we show that minimizing variational free energy in such systems yields a Hopfield-style update mechanism. This mechanism naturally leads to a self-organizing, continuous-state stochastic variant of classical Hopfield networks, where the dynamics steer the network toward stable attractor states.
Third, we show that minimizing free energy induces a generalized Hebbian learning process. In this context, coupling between network elements is continuously adjusted by comparing actual sensory inputs with predictions from an internal generative model. This adaptive process not only reinforces the storage of key sensory patterns but also transforms them into optimized, orthogonalized attractors that enhance the network's overall computational performance. This means that, the self-organizing attractor network emerging from the FEP turns out to approximate the an attractor network architecture with maximal information storage capacity, called projection network.
Finally, we demonstrate some of our mathematical results through simulations, discuss some testable predictions of our framework, and explore the broader implications of these findings for both natural and artificial intelligence systems.

> make the last sentence more specific

# Main Results

## Background: Particular Partitions and the Free Energy Principle

Our efforts to characterize computations in complex, self-organizing systems calls for an individuation of 'self' from nonself. *Particular partitions*, a concept that is at the core of the Free Energy Principle (FEP) {cite:p}`https://doi.org/10.1371/journal.pone.0006421; https://doi.org/10.7551/mitpress/12441.001.0001; https://doi.org/10.1016/j.neubiorev.2016.06.022; https://doi.org/10.1016/j.biopsycho.2023.108741`, is a natural way to formalize this individuation.
A particular partition is a partition that divides the states of a system $x$ into a particle or 'thing' $(s,a,\mu) \subset x$ and its external states $\eta \subset x$, based on their sparse coupling (see ([](fig-concept)A)):

:::{math}
:label: eq-particular-partition
\begin{align*}
\begin{bmatrix}
\dot{\eta}(\tau) \\
\dot{s}(\tau) \\
\dot{a}(\tau) \\
\dot{\mu}(\tau)
\end{bmatrix}
=
\begin{bmatrix}
f_{\eta}(\eta, s, a) \\
f_{s}(\eta, s, a) \\
f_{a}(s, a, \mu) \\
f_{\mu}(s, a, \mu)
\end{bmatrix}
+
\begin{bmatrix}
\omega_{\eta}(\tau) \\
\omega_{s}(\tau) \\
\omega_{a}(\tau) \\
\omega_{\mu}(\tau)
\end{bmatrix}
\end{align*}
:::

where $\mu$, $s$ and $a$ are the internal, sensory and active states of the particle, respectively. The fluctuations $\omega_i, i \in (\mu, s, a, \eta)$ are assumed to be mutually independent. The particular states $\mu$, $s$ and $a$ are coupled to each other with *particular flow dependencies*; namely, external states can only influence themselves and sensory states, while internal states can only influence themselves and active states. It can be shown that these coupling constraints mean that external and internal paths are statistically independent, when conditioned on blanket paths {cite:p}`https://doi.org/10.1016/j.plrev.2023.08.016`:

:::{math}
:label: eq-conditional-independence
\eta \perp \mu \mid s, a.
:::

As shown by {cite:t}`https://doi.org/10.1016/j.plrev.2023.08.016`, such a particle, in order to persist for an extended period of time, will necessarily have to maintain this conditional independence structure, a process that is equivalent to an inference process in which internal states infer external states through the blanket states (i.e., sensory and active states) by minimizing variational free energy {cite:p}`https://doi.org/10.1371/journal.pone.0006421; https://doi.org/10.1038/nrn2787; https://doi.org/10.1016/j.physrep.2023.07.001`:

:::{math}
:label: eq-free-energy-principle
\eta \perp \mu \mid s, a \quad \Rightarrow \quad \dot{\mu} = -\nabla_{\mu} F(s, a, \mu)
:::

where $F(s, a, \mu)$ is the variational free energy:

:::{math}
:label: eq-free-energy-functional
F(s,a,\mu) = \mathbb{E}_{Q(\mu)}[\ln Q(\mu) - \ln P(s,a,\mu)]
:::

with $Q(\mu)$ being a variational density over the internal states and $P(s,a,\mu)$ being the joint probability distribution of the sensory, active and internal states, a.k.a. the generative model {cite:p}`10.1016/j.neubiorev.2016.06.022`.


 :::{figure} fig/concept.png
 :name: fig-concept
 :width: 66%
 **From the Free Energy Principle to Top-Level Information Integration via Attractor Dynamics.** \
 **A** Schematic illustrating the a particular partition of a system into internal states ($\mu$) and external (hidden, latent) states ($\eta$), separated by a Markov blanket consisting of sensory states ($s$) and active states ($a$). The internal and sensory states minimize a free energy functional of the sensory inputs. The resulting self-organization of internal states corresponds to perception, while actions link the internal states back to the external states.
**B** The internal states $\mu \subset x$ can be arbitrarily complex. Without loss of generality, we can consider that $\mu$ can be decomposed into  set of nested, overlapping particular partitions, so that the internal state $\sigma_i \subset \mu$ of partition $i$ can be the external state of another nested particular partition $\sigma_j \subset \mu$. Some (or all) nested partitions can be directly connected to the external state $\eta$, giving a decomposition of the original boundary states into $s_i \subset s$ and $a_i \subset a$. The nested particular partitions are connected by $s_{ij}$ and $a_{ij}$. This notation considers the point-of-view of the $i$-th nested particular partition. From the perspective of the $j$-th nested particular partition, we could write $s_{ji}=a_{ij}$ and $a_{ji}=s_{ij}$. While the figure depicts the simplest case of two nested partitions, the same reasoning can be applied to any number of nested partitions and any coupling structure amongst them.
 **C** While we are interested in the general case, in our formalism we will start from a specific class of nested particular partitions, in which the internal states $\mu$ can be decomposed into a bipartite coupling structure, consisting of two layers: $\sigma_i \subset \mu$ and $\mu_j \subset \mu$. We will show that, when starting from realistic assumptions, this class of nested particular partitions is equivalent to a Restricted Boltzmann Machines (RBM). Furthermore, building on a mathematical duality between RBMs and Hopfield attractor networks, for any arbitrary nested coupling structures (like in B), there exist an infinite number of statistically equivalent particular partition with a bipartite coupling structure. Converting any arbitrary nested partition into an equivalent bipartite partition is a key step in our conceptual framework, as it allows us to draw parallels between integrative computations performed by attractor dynamics and active inference (e.g. Bayesian Causal Inference or predictive coding) as implemented by conventional hierarchical architectures.
 :::


## Subparticles and Complex Particular Partitions

Particular partitions provide a universal description of complex systems, in a sense that the internal states $\mu$ behave as a "black box"; the corresponding generative model, as well as the attached inference process (or computation), can be arbitrarily complex. At the same time, the concept of particular partitions speaks to a recursive composition of ensembles (of things) at increasingly higher spatiotemporal scales {cite:p}`https://doi.org/10.48550/arXiv.1906.10184; https://doi.org/10.1016/j.jtbi.2019.110089`, which yields a natural way to resolve the internal complexity of $\mu$. Dividing a particle into multiple overlapping subsets, that themselves are particular partitions equips the particle with a hierarchical or deep generative model. 
As a shorthand, we will refer to such nested particular partitions as *subparticles*. As illustrated on [](fig-concept)B, we allow subparticles to be coupled to each other through shared boundary states. That is, we allow the internal state of a subparticle $\sigma_i \subset \mu$ to be the external states for another subparticle in $\sigma_j \subset \mu$, and vice versa.

This implies that the internal state of a subparticle $\sigma_i$ is conditionally independent of any other internal state $\sigma_j$ with $j \neq i$, given the blanket states of the subparticles:

:::{math}
\sigma_i \perp \sigma_j \mid a_{ij}, s_{ij}, a_i, s_i
:::

where $a_{ij}$ and $s_{ij}$ are the active and sensory states that couple the subparticles $\sigma_i$ and $\sigma_j$, respectively and $a_i$ and $s_i$ (with a single subscript) are active and sensory states that connect the subparticle $\sigma_i$ to the external states. Note that the sensory state of a subparticle $\sigma_i$ is the active state of $\sigma_j$ and vice versa, i.e. $a_{ji}=s_{ij}$ and $s_{ji}=a_{ij}$. Note that this definition embraces sparse couplings across subparticles, as $a_{ij}$ and $s_{ij}$ may be empty for a given $j$ (no direct connection between two internal states), but we require the subparticles to yield a *complete coverage* of $\mu$: $\bigcup_{i=1}^n \pi_i = \mu$.

We will call a particular partition, which can be decomposed into subparticles in the above defined way - and possibly in a recursive manner - a *deep particular partition*.

## The Emergence of Neural Networks from Complex Particular Partitions

Next, we establish a prototypical mathematical parametrization for an arbitrary deep particular partition, shown on  [](fig-parametrization), with the aim of demonstrating that such complex sparsely coupled random dynamical systems can give rise to artificial attractor neural networks.

 ::: {figure} fig/parametrization.png
 :name: fig-parametrization
 :width: 50%
 **A basic subparticle in a bipartite complex particular partition.** <br>
A basic subparticle in a *bipartite complex particular partition* comprises an external state, which is typically the internal state of another subparticle, ($\sigma_i$), an active state ($a_{ij}$) and a sensory state ($s_{ij}$) establishing an inter-layer coupling, and an internal state, which acts like an integrator node ($\mu_j$) in the higher-level layer. In our prototypical parametrization, the prior distribution of $\sigma_i$ is modeled as a restricted continuous Bernoulli distribution with a prior bias ($b_i$) [](#Supplementary-Information-1). The higher-levelintegrator state $\mu_j$ follows a Gaussian distribution with a precision parameter ($\beta_j$). The sensory state applies a deterministic scaling to the lower-level state, with a weight ($W_{ij}$). Similarly, the active state deterministically scales the integrator state using the same weight ($W_{ij}$), both parameterized with a Dirac delta function shifted by $W_{ij}$. <br>
Note that in this parametrization, all states can be complex particular partitions themselves, where the internal mechanisms can implement deep generative models. The resulting computation can be encoded in the prior bias and precision of the nodes internal state.
 :::

In this example parametrization, we assume that the internal states of subparticles in a complex particular partition are continuous Bernoulli states (also known as truncated exponential), denoted by $\sigma_i \sim \mathcal{CB}_{b_i}$. Here, $\sigma_i \in [-1, +1] \subset \mathbb{R}$, and $b_i$ represents an a-priori bias (e.g. the level of prior log-odds evidence for an event) in $\sigma_i$ ([](fig-parametrization)). See [](#Supplementary-Information-1) for details.

:::{math}
:label: prior-sigma
P(\sigma_i) \propto e^{b\sigma_i}
:::

This probability is defined up to a normalization constant, $b_i / (2\sinh(b_i))$. See [](#Supplementary-Information-2) for details.

Next, we define the conditional probabilities of the sensory and active states, creating the boundary between two subparticles $\sigma_i$ and $\sigma_j$: $s_i|\sigma_j \sim \mathcal{\delta}_{J_{ij}\sigma_j}$ and $a_{ij} |\sigma_i \sim \mathcal{\delta}_{J_{ij}\sigma_i}$, where $\mathcal{\delta}$ is the Dirac delta function and $\bm{J}$ is a weight matrix. The elements $J_{ij}$ contains the weights fo the coupling matrix (see [](Supplementary-Information-3)).

Expressed as PDFs:

:::{math}
P(s_{ij} | \sigma_j) = \delta(s_{ij} - J_{ij}\sigma_j)
:::
:::{math}
P(a_{ij} | \sigma_i) = \delta(a_{ij} - J_{ij}\sigma_i) 
:::

We now write up the direct conditional probability describing $\sigma_i$ given $\sigma_j$, marginalizing out the sensory and active states:

:::{math}
:label: sigma-given-mu
P(\sigma_i | \sigma_j) &= \int P(\sigma_i | s_{i,j}) P(s_{i,j} | \sigma_j) \, d s_{i,j} \\
&\propto \int e^{(s_{i,j}+b_i)\sigma_i} \delta( s_{i,j} - J_{i,j} \sigma_j ) d s_{i,j} \\
&\propto e^{(b_i + J_{i,j}\sigma_j)\sigma_i}
:::

:::{note}
The expected value of $P(\sigma_i | \sigma_j)$ is a sigmoid function of $\sigma_j$ [](#supplementary-information-4), specifically the Langevin function. This property allows it to function as an activation function in neural networks, enabling the network to model more complex patterns and decision boundaries.
:::

Given that $P(\sigma_i, \sigma_j) = P(\sigma_i | \sigma_j) P(\sigma_j)$, and using equations [](prior-sigma) and [](sigma-given-mu), we can express the joint distribution as follows:

:::{math}
P(\sigma_i , \sigma_j) &= e^{(b_i + J_{i,j}\sigma_j)\sigma_i} e^{ b_j \sigma_j} \\
&= e^{b_i \sigma_i + J_{i,j} \sigma_i \sigma_j + b_j \sigma_j}
:::

Next, we observe that the states $s$ and $a$ are the 'blanket states' of the system, forming a Markov blanket {cite:p}`https://doi.org/10.1016/j.neubiorev.2021.02.003`, which implies that the joint probability for all $\bm{\sigma}$ nodes can be written as the product of the individual joint probabilities ([](fig-concept)C), $\prod_{i,j} P(\sigma_i, \sigma_j)$, which results in:

:::{math}
:label: multiple-integrator-joint
P(\bm{\sigma}) \propto e^{\sum_{i} b_i \sigma_i + \sum_{i \neq j} J_{i,j}\sigma_i\sigma_j}
:::

Since $\sigma_i\,\sigma_j = \sigma_j\,\sigma_i$, we can rearrange the double sum over distinct pairs:

:::{math}
\sum_{i\neq j} J_{ij}\,\sigma_i\,\sigma_j = \sum_{i<j} \Bigl(J_{ij}+J_{ji}\Bigr)\,\sigma_i\,\sigma_j.
:::

Thus, even though we started with non-symmetric couplings $J_{ij}$ and $J_{ji}$, the joint distribution ends up depending only on the sum $J_{ij}+J_{ji}$:

:::{math}
:label: hopfield-joint
\boxed{
P(\bm{\sigma}) \propto \exp \Biggl\{ \underbrace{\underbrace{\sum_{i} b_i \sigma_i}_{\textit{bias term}} + \underbrace{\frac{1}{2} \sum_{ij} J^{\dagger}_{ij}\sigma_i\sigma_j}_{\textit{interaction term}} }_{\textit{- energy (Hamiltonian)}} \Biggr\}
}
:::

with $J^{\dagger}_{ij} = \frac{1}{2} (J_{ij} + J_{ji})$ and $J^{\dagger}_{ii} = 0$ for all $i,j$. 

This joint probability distribution takes the functional form of a stochastic continuous-state Hopfield network (a specific type of Boltzmann machines).  As known in such systems, regions of high probability density in this stationary distribution will constitute "stochastic attractors", which are the regions of the state space that the system will tend to converge to. Furthermore, in case of asymmetric couplings, "solenoidal flows" occur, extending the attractor repertoire with "sequence attractors" (heteroclinic chains).
Importantly, our derivation shows that, while asymmetric couplings can break detailed balance in the system and induce non-equilibrium dynamics (sequence attractors, heteroclinic chains), the stationary distribution remains determined by the symmetric part $J^{\dagger}$ (under the usual FEP assumption of constant isotropic noise). This happens because the antisymmetric part acts only tangentially to the isosurfaces of the potential defined by the symmetric part and, therefore, does not contribute to the net probability flux. 

In other words, even though the local influences use the potentially asymmetric $\bm{J}$, the only joint distribution that can reproduce all the local conditionals $P(\sigma_i | \sigma_\backslash i)$ (backslashes denoting all states other than $i$) is one that involves the symmetric part $\bm{J}^{\dagger}$. The conditional independence structure enforces that all the local conditionals turn out to be consistent only with this single joint distribution that factorizes in an undirected (symmetric) way.
This step is neither a general feature of NESS systems nor an assumption - it is a necessary consequence of the conditional independence imposed by the Markov blanket (see [](#Supplementary-Information-6) for details). 
Thus, the *deep particular partition* structure ensures that while asymmetric couplings create transient non-equilibrium dynamics, the long-term statistical structure of attractors is governed by the symmetric coupling component $J^{\dagger}$. 

As we will see, the dynamics derived from local free energy minimization in this system depend on the potentially asymmetric couplings $J_{ij}$.

In sum, above we have demonstrated with a specific, plausible parametrization that the types of **recurrent artifical neural networks emerging naturally from deep particular partitions have a stationary distribution that is determined solely by the symmetric part of the coupling matrix**.

## Inference

Now we remember that, according to the free energy principle, (complex) particular partitions - in order to exist for an extended period of time - will inevitably minimize their variational free energy and consider the implications of this for the recurrent neural network we derived in the previous section. We start by writing up variational free energy (eq. [](#eq-free-energy-functional)) from the point of view of a single (lower-level) node of the attractor network, $\sigma_i$, given observations from all other nodes, $\sigma_{\backslash i}$:

:::{math}
:label: free-energy-rnn
F &= \mathbb{E}_{q(\sigma_i)}[\ln q(\sigma_i) &-& \ln P(\sigma_{\backslash i}, \sigma_i)]
 = &\underbrace{D_{KL}[\ln q(\sigma_i) || P(\sigma_i)]}_{\textit{complexity} } &- &\underbrace{\mathbb{E}_{q(\sigma)}[ \ln P(\sigma_{\backslash i} | \sigma_i)]}_{\textit{accuracy}}
:::

Where $q(\sigma_i)$ is the approximate posterior distribution over $\sigma_i$, which we will parametrize as a $\mathcal{CB}$ with variational bias $b_q$. $\sigma_{\backslash i}$ simply denotes all other states.

Now we are interested in how node $\sigma_i$ must update its bias, given the state of all other nodes, $\sigma_{\backslash i}$ and the weights $J_{ij}$. Intuitively, the last part of eq. [](#free-energy-rnn) tells us that minimizing free energy will lead to a (local) minimum of the energy of the attractor network (accuracy term), with the constraint that this has to lie close to the initial state (complexity term).

Let us verify this intuition by substituting our parametrization into eq. [](#free-energy-rnn). From eq. [](#rmb-to-hopfield), we have:
 :::{math}
P(\sigma_{\backslash i} | \sigma_i) \propto \exp\left( \sum_{j \neq i} \left( b_j + J_{i,j} \sigma_i \right) \sigma_j + \frac{1}{2} \sum_{j \neq i} \sum_{k \neq i} J_{j,k} \sigma_j \sigma_k \right)
:::

The complexity term in eq. [](#f-complexity-accuracy) is simply the KL-divergence term between two $\mathcal{CB}$ distributions:

:::{math}
:label: kl-divergence-cb
D_{\text{KL}}[ q(\sigma_i) \| p(\sigma_i) ]
= \bigl[ \ln\!\bigl(\tfrac{b_q}{\sinh(b_q)}\bigr) + b_q\,L(b_q) \bigr]
- \bigl[ \ln\!\bigl(\tfrac{b}{\sinh(b)}\bigr) + b\,L(b_q) \bigr]
:::

where $L(\cdot)$ is the expected value of the $\mathcal{CB}$, a sigmoid function of the bias, specifically the Langevin function (see [](#supplementary-information-4)) $L(x) = \mathbb{E}(x) = coth(b) - 1/b$ ([](#supplementary-information-4)). 
For details on the derivation, see [todo:SI](#supplementary-information-x).

The accuracy-term in eq. [](#f-complexity-accuracy) can be expressed as:

:::{math}
\mathbb{E}_{q(\sigma_i)} [ \ln P(\sigma_{\backslash i} \mid \sigma_i) ] = \text{const} + \sum_{j \ne i} b_j \sigma_j + L(b_q) \sum_{j \ne i} J_{ij} \sigma_j + \dfrac{1}{2} \sum_{j \ne i} \sum_{k \ne i} J_{jk} \sigma_j \sigma_k
:::

Leading to the following expression for the free energy:

:::{math}
:label: free-energy-parametrized
F = (b_q - b) L(b_q) - \sum_{j \ne i} \left( b_j + S(b_q) J_{ij} \right) \sigma_j - \dfrac{1}{2} \sum_{j \ne i} \sum_{k \ne i} J_{jk} \sigma_j \sigma_k + C
:::

where C denotes all constants in the equation that are independent of $\sigma$.

For details on the derivation, see [todo:SI](#supplementary-information-x).

Now, taking the partial derivative of the free energy with respect to the variational bias:

:::{math}
\dfrac{\partial F}{\partial b_q} = \left( b_q - b - \sum_{j \ne i} J_{ij} \sigma_j \right) \dfrac{dL}{db_q}
:::

Setting the derivative to zero, solving for $b_q$, and substituting the expected value of the $\mathcal{CB}$ distribution ([](#supplementary-information-4)), we get:

:::{math}
\boxed{
\mathbb{E}_{q}[\sigma_i] = L(b_q) = \underbrace{ L \left( \underbrace{ b_i}_{\textit{bias}} + \underbrace{\sum_{j \ne i} J_{ij} \sigma_j}_{\textit{weighted input}} \right) }_{ \textit{sigmoid (Langevin)} } 
}
:::

with L being the Langevin function. Thus, in case of symmetric couplings, this is exactly the update rule for the continuous-state stochastic Hopfield network. Of note, it can also been shown that the inverse temperature parameter of Hopfield networks can be derived from the precision of the integrator nodes (see [](#supplementary-information-9)).

While the deterministic variant of the above inference rule can be derived directly as a gradient decent on the energy function, the presented FEP-based derivation naturally extends this to a probabilistic framework, - with a naturally emerging sigmoid function (through the complexity term). While energy minimization points in the right direction, the FEP minimization provides the full probabilistic machinery, including regularizing constraints, instead of just moving down deterministically on an energy gradient.
The resulting stochastic dynamics leads to the optimal expected belief under variational inference, naturally incorporating prior biases, state constraints (sigmoid due to the $\{-1,1\}$ state space) and equals to a local approximate Bayesian inference, where the approximate posterior belief $q^$ balances prior information ($b_i$) and evidence from neighbours ($\sum J{ij} \sigma_j$).
As we will show later in the manuscript, the inherently stochastic characteristics of inference is what allows the network as a whole, too, to escape local energy minima over time - consistent with MCMC methods - and thereby perform macro-scale Bayesian inference.

## Learning

At optimum, $Q$ would match P (with $b^{(q)} = u_i$ and $\mathbb{E}[\sigma_i] = L(u_i)$, causing the VFE's derivative to vanish.
Learning happens, when there is a systematic change in the prior bias $b_i$ that counteracts the update process. This can correspond to an external input (e.g. sensory signal representing increased evidence for an external event).  In this case, the system can take use of another (slower) process, to decrease free energy: it can change the way its action states are generated; and rely on its vicarious effects on sensory signals. In our parametrization, this can be achieved by changing the coupling strength $J_{ji}$. Of note, while changing $J_{ji}$ corresponds to a change in action-generation at the local level of the subparticle, globally, it can be considered as a change in the whole system's internal belief about the statistical structure of the inputs.

Let's revisit VFE from the perspective of node *i*:
:::{math}
F = \mathbb{E}_{Q(\sigma_i)}\Bigl[\ln q(\sigma_i) - \ln P(\sigma_{\backslash i} \mid \sigma_i)\Bigr]
:::

and parameterize the distributions as:
:::{math}
\ln q(\sigma_i) \propto b_q\,\sigma_i, \qquad
\ln P(\sigma_{\backslash i} \mid \sigma_i) \propto u_i\,\sigma_i
:::

with $u_i$ being the net weighted input to node *i*: $u_i = b + \sum_{j\neq i} J_{ij}\,\sigma_j$

we obtain

:::{math}
F = \mathbb{E}_{q(\sigma_i)}\Bigl[(b_q - u_i)\,\sigma_i\Bigr] + \phi(u_i) - \phi(b_q)
:::

At equilibrium (i.e. when $b_q = u_i$), we have $\mathbb{E}_q[\sigma_i] = L(u_i)$, where $L(u_i)$ is the Langevin function.

For learning, we use a stochastic (sample-based) estimate by replacing the expectation $\mathbb{E}_q[\sigma_i]$ with the instantaneous value $\sigma_i$. A perturbation $\delta J_{ij}$ produces a change $\delta u_i = \sigma_j\,\delta J_{ij}$, and by applying the chain rule we get:

:::{math}
\frac{dF}{dJ_{ij}} = \frac{\partial F}{\partial u_i}\,\frac{\partial u_i}{\partial J_{ij}} = \bigl[L(u_i)-\sigma_i\bigr]\,\sigma_j
:::

Substituting back $u_i$ and rearranging we get:

:::{math}
\boxed{
\Delta J_{ij} \propto \underbrace{\sigma_i \sigma_j}_{\textit{observed correlation (Hebbian)}} - \underbrace{ L(b_i + \sum_{k\neq i} J_{ik}\,\sigma_k ) \sigma_j}_{\textit{predicted correlation (anti-Hebbian)}}
}
::: 

This learning rule belongs to the family of "Hebbian / anti-Hebbian" or "contrastive" learning rules and it explicitly implements predictive coding. As opposed to contrastive divergence (contrastive Hebbian learning), it does not require to contrast longer averages of separate "clamped" (fix inputs) and "free" (free running) phases, but rather uses the instantaneous correlation between presynaptic and postsynaptic activation to update the weight, lending a high degree of scalability for this architecture. A key feature of this rule is it's resemblance to Sanger's rule [REF], hinting that it imposes an approximate orthogonality across attractor states. We motivate this theoretically in the next section.

## Emergence of approximately orthogonal attractors

Let us assume that the network stores $P$ patterns (has P attractors) $\{\boldsymbol{\sigma}^{(p)}\}$. In this case, the Hebbian term builds the weight matrix as:

:::{math}
J_{ij} \propto \sum_{p=1}^{P} \sigma_i^{(p)} \sigma_j^{(p)}.
:::

In this raw sum, the inner product (overlap) between distinct patterns,
$O_{pq} = \sum_{i} \sigma_i^{(p)}\sigma_i^{(q)} \quad (p\neq q)$,
may be nonzero, leading to interference. The consequences of this are well known in case of deterministic Hopfield networks: the network will retrieve the wrong pattern (spurious attractor state), or a superposition of patterns, instead of the intended one.

To see how the predictive (anti-Hebbian) term cancels the contribution of overlaps between different stored patterns, let us consider that the network is presented with pattern $\boldsymbol{\sigma}^{(p)}$, which overlaps with another stored pattern $\boldsymbol{\sigma}^{(q)}$.
In this case, the current weights will partially "explain out" this overlap through the anti-Hebbian term. 
which acts to cancel the contribution of overlaps between different stored patterns. Under repeated updates - i.e., gradient descent on the variational free energy - this cancellation drives the off-diagonal overlaps toward zero:

:::{math}
\boxed{
\langle \boldsymbol{\sigma}^{(p)}, \boldsymbol{\sigma}^{(q)} \rangle \to 0\quad \text{for} \quad p\neq q.
}
:::

Thus, the attractor states become approximately orthogonal, reducing interference and enhancing distinct memory representations.

As an analogy, we can imagine that we are training a network to recognize both dogs and wolves. Initially, it might respond to both when seeing pointy ears (an overlapping feature). The anti-Hebbian term removes these already modelled similiraties and helps the network learn to focus on distinguishing features instead, effectively orthogonalizing the representations.

To summarize, here we have proved that both moment-to-moment inference (through a generalized, continuous version of the Hopfield update rule) and slow structural learning (through a generalized, orthogonalizing form of Hebb's postulate) in self-organizing attractor neural networks emerge from the free energy principle in complex particular partitions. The resulting self-organizing systems happen to be a special case of attractor networks, with a key characteristic: a tendency to orthogonalize their attractor states. This raises the question, how we can match the orthogonalized attractor the system converged to with the original non-orthogonal pattern -  a key requirement to function as an associative memory.
In the next section, we show that this is only a problem with deterministic dynamics. In the stochastic case, the network - by means of multistability - can be in a superposition of different random attractor states, and thereby retrieve the original (non-orthogonal) pattern as a combination of the orthogonal bases represented by the attractors. We show, further, that as the derived system is equivalent to Boltzmann machines (without hidden units), this stochastic retrieval can be interpreted as an approximate Bayesian inference at the macro-scale, performed by the entire network.

## Stochastic retrieval as macro-scale Bayesian inference

As a consequence of the Free Energy Principle, the inference process described above, where each subparticle $\sigma_i$ updates its state based on local information (its bias $b_i$ and input $\sum_j J_{ij} \sigma_j$), can be seen as a form of micro-scale inference, in which the prior - defined by the nodes internal bias, gets updated by the evidence collected from the neighboring subparticles to form the posterior.

 However, as the whole network itself is also a particular partition (specifically, a deep one), it must also perform Bayesian inference, at the macro-scale. 
 To show this, we can use the fact that the attractor network at hand is equivalent to Boltzmann machines (without hidden units), which have been thoroughly discussed to perform macro-scale approximate Bayesian inference, through Markov Chain Monte Carlo (MCMC) sampling.
 
Let us consider the network's learned weights $J$ (and potentially its baseline biases $b^{\text{base}}$) as defining a **prior distribution** over the collective states $\boldsymbol{\sigma}$:
:::{math}
P(\boldsymbol{\sigma}) \propto \exp \Biggl\{ \sum_{i} b_i^{\text{base}} \sigma_i + \frac{1}{2} \sum_{ij} J_{ij}\sigma_i\sigma_j \Biggr\}
:::

Now, suppose the network receives external input (evidence) $\mathbf{y}$, which manifests as persistent modulations $\delta b_i$ to the biases, such that the total bias is $b_i = b_i^{\text{base}} + \delta b_i$. This evidence can be associated with a **likelihood function**:
:::{math}
P(\mathbf{y} | \boldsymbol{\sigma}) \propto \exp \left( \sum_i \delta b_i \sigma_i \right)
:::

According to Bayes' theorem, the **posterior distribution** over the network states given the evidence is $P(\boldsymbol{\sigma} | \mathbf{y}) \propto P(\mathbf{y} | \boldsymbol{\sigma}) P(\boldsymbol{\sigma})$:
:::{math}
:label: posterior-distribution
P(\boldsymbol{\sigma} | \mathbf{y}) \propto \exp \Biggl\{ \sum_{i} b_i \sigma_i + \frac{1}{2} \sum_{ij} J_{ij}\sigma_i\sigma_j \Biggr\}
:::
Remarkably, this posterior distribution has the same functional form as the network's equilibrium distribution under the influence of the total biases \($b_i$\).

The stochastic update rule derived from minimizing local free energy (Eq. \ref{eq-free-energy-principle}), when implemented by sampling from the conditional distribution $q(\sigma_i)$ rather than using its mean, effectively performs Markov Chain Monte Carlo (MCMC) sampling (specifically, akin to Gibbs sampling) from the joint distribution defined by the current energy landscape. In the presence of evidence $\mathbf{y}$, the network dynamics therefore *sample* from the posterior distribution $P(\boldsymbol{\sigma} | \mathbf{y})$.

The significance of the stochasticity becomes apparent when considering the network's behavior over time. The time-averaged state $\langle \boldsymbol{\sigma} \rangle$ converges to the expected value under the posterior distribution:

:::{math}
\boxed{
\langle \boldsymbol{\sigma} \rangle = \lim_{T \to \infty} \frac{1}{T} \int_0^T \boldsymbol{\sigma}(t) dt \approx \mathbb{E}_{P(\boldsymbol{\sigma} | \mathbf{y})}[\boldsymbol{\sigma}]}
:::

Noise or stochasticity allows the system to explore the posterior landscape, escaping local minima inherited from the prior if they conflict with the evidence, and potentially mixing between multiple attractors that are compatible with the evidence. The resulting average activity $\langle \boldsymbol{\sigma} \rangle$ thus represents a Bayesian integration of the prior knowledge encoded in the weights and the current evidence encoded in the biases. This contrasts sharply with deterministic dynamics, which would merely settle into a single (potentially suboptimal) attractor within the posterior landscape.

With this, the loop is closed. The stochastic attractor network emerging from the FEP framework naturally implements macro-scale Bayesian inference through its collective sampling dynamics, providing a robust mechanism for integrating prior beliefs with incoming sensory evidence.
This reveals a potentially deep recursive application of the Free Energy Principle: the emergent collective behavior of the entire network, formed by interacting subparticles each minimizing their local free energy, recapitulates the inferential dynamics of a single, larger-scale particle. This recursion could extend to arbitrary depths, giving rise to a hierarchy of nested particular partitions and multiple emergent levels of description, each level performing Bayesian active inference according to the same fundamental principles.


> provides a principled way to build hierarchical systems

:::{figure} fig/network.png
:name: fig-results
:align: center
:width: 100%
**Free energy minimizing, adaptively self-organizing attractor network**
:::

# In silico demonstrations

We illsutrate three key features of the proposed framework with computer simulations. We build a 64-node attractor network and train it in a continous-learning fashion (i.e simultaneously performing inference and learning) on 10 images of handwritten digits (1 example of each of the 10 digits from 0 to 9, 8x8 pixels each).
First, we characterize the efficiency of learning in terms of (i) stochastic (Bayesian) pattern retrieval from noisy variations of the training images; and (ii) one-shot generalization to recontruct unseen digits. 
Importantly, we confirm that - with certain conditions - the network is able to produce approximately orthogonal attractor states  show that in this case the network provides the best compromise between training accuracy and generalization performance; rooted in an optimal balance between the complexity and accuracy of internal representations.
Next, we demonsterate that the network is resistant to catastrophic forgetting and is able to learn state transitions amongst the attractors (sequence attractors), if training data is intriduced in a reccurring order.

Our python implementation of the network, available at {cite:t}`https://github.com/tspisak/fep-attractor-networks`. The implementation favors clarity over efficiency - it implements both $\sigma$ and boundary states as separate classes, and is not optimized for performance. A crucial detail for the implementation of the learning rule is that - in line with the our formal framework - the predictive (anti-Hebbian) term uses the *previous* state of the network, before the new biases (novel input/evidence) were applied.

In the first example

:::{figure} fig/results.png
:name: fig-results
:align: center
:width: 100%
**Consequences of the proposed framework**
**A** Ring attractors represented by cosine basis functions in an equivalent two-layer network. \
**B** Predictive coding implemented by an attractor network. \
**C** Bayesian causal inference implemented by an attractor network. \
**D** Large-scale brain attractors from {cite:t}`https://doi.org/10.7554/eLife.98725.1` \
**E** Large-scale brain attractors are close to orthogonal to each other.
:::

# Discussion



First, it shows that the emergence of such networks poses not many requirements; all is about a specific structure of sparse coupling.

**Discussion:**


Operational self-organization provides the moment-to-moment stability, robustness, and coordinated action needed for effective functioning.
Adaptive self-organization provides the long-term learning, growth, and flexibility needed for intelligence and autonomy in complex, changing worlds.


>  We show that the inference dynamics and learning rules of these systems are a direct manifestation of the naturally emerging, self-organizing free-energy minimization process. In other words, we arrive at a complete derivation of the emergent, self-organizing recurrent neural networks in eq. [](#hopfield-joint) from first principles.

>Our analysis reveals that general complex particular partitions, under well-defined conditions, gives rise to both (i) bipartite restricted Boltzmann machine (RBM) architectures and (ii) functionally equivalent attractor neural networks. This discovery has profound implications for our understanding of neural computation: the emergent computational properties of neural networks are not contingent upon specialized learning rules or carefully designed relaxation dynamics. Instead, these properties represent fundamental characteristics inherent to the underlying random dynamical system, emerging naturally through the principles of free energy minimization.

>In the next paragraphs we will demonstrate this by showing that the update and learning rules of the specific Hopfield network architecture presented in the previous section are a direct consequence of the free energy principle.

> also: oja's rule vs. representational drift

Understanding the computational principles underlying the attractor dynamics of complex dynamical system is a key challenge in the field of dynamical systems theory, with broad implications for the field of neuroscience and artificial intelligence. 
In this paper, we introduce a new framework that provides a formal bridge between far-from-equilibrium thermodynamics and information processing in biological and artificial systems.

Below, we discuss the key points (\#1 - \#8) of our findings, also summarized in [](summary-main-findings).

:::{list-table} Summary of Main Findings
:name: summary-main-findings
:header-rows: 1
* - 
  - <div style="width:300px">Finding</div>
  - Description
* - \#1
  - **FEP $\rightarrow$ self-organizing ANNs**
  - Conventional artificial neural networks, like RBMs or HNs, emerge naturally from particular partitions of complex systems, as special cases. The free energy principle provides a formal account of the dynamical and computational properties of such systems. For instance the update rule of Hopfiled networks can be derived from free energy minimization performed independently on each node.
* - \#4
  - **Attractors $=$ emergent computations**
  - Computations implemented by attractor dynamics are separated from low-level, within-node computations in the system by a "computational closure", giving rise to an emergent macroscopic description of the system.
* - \#5
  - **Attractors $=$ efficience & robustness**
  - The implicit embedding of higher-level computations in the attractor dynamics of a (unipartite) system has several advantages over implementing the same computations explicitly in higher-level nodes of bipartite systems, including cost-efficiency, robustness and - if higher order interactions are allowed - exponential scalability.
:::

### 1. Deriving ANNs from first principles
We established formal links between computational architectures (artificial neural networks (ANNs)) and far-from-equilibrium dynamical systems at two levels.
First, we have shown that a universal partitioning that can be applied to any complex system (and also underwrites the Free Energy Principle) can give rise to network structures whose states have the same joint probability distributions as conventional artificial neural networks. Specifically, we first derived Restricted Boltzmann machines (RBMs) from complex particular partitions and then showed that Hopfield networks (HNs) can be derived from the same partition, but with a different parametrization.
Second, we have shown that the free energy principle provides a formal account for the emergence of dynamics in such systems that were previously introduced ad-hoc, to give rise to useful computations. Specifically, we have shown that the update rule of Hopfield networks can be derived from first principles, as gradient descent on the free energy landscape, performed independently by each node of the network.
For the derivation of these results, we used plausible and relaxable assumptions. The key assumption of the system having a complex particular partition is a minimal assumption for any complex far-from-equilibrium dynamical system, and the free energy principle provides a formal framework for understanding inference in such systems. An important additional assumption is that that that the couplings between the subparticles are symmetric. While this may seem a strong assumption, it is only needed to derive an exact equivalence to conventional ANNs. While many of the equations would become significantly more complex, if the symmetry assumption is dropped, there are good reasons to believe that the general principles would remain the same. For instance, it has been shown that recurrent ANNs with asymmetric weights can perform as well as their symmetric counterparts (see, e.g., {cite}`https://doi.org/10.1016/0893-6080(95)00114-X; https://doi.org/10.1006/jmaa.1995.1138`).
Deriving ANNs from first principles - with plausible assumptions - sheds light on the processes by which biological systems can self-organize into ANN-like computational structures through far-from-equilibrium dynamics, without explicit engineering.


### 4. Attractor dynamics as an emergent level of computation

By linking our results to a recent computational approach to emergence (REF), we showed that computations embedded into attractor dynamics can be interpreted as an emergent macroscopic description of the system. Specifically, attractor-level computations exhibit informational closure from lower-level states. This emergent property allows attractor-level computations to be macroscopic descriptions independent of microscopic details. This, in turn, allows for efficient state representation through dimensionality reduction in such systems (see eigenmodes and principal components).
The formal description of attractor-timeseries also substantiates that computations encoded in different attractors happen in parallel. This may challenge the common energy-landscape analogy, which suggests that the system's behavior at any given moment is entirely determined by the single attractor in whose basin the system resides.
The emergent nature of attractor-level computations can be used to construct a test to determine whether a computation related to a certain outcome or behavior of the system is implemented in the attractor space (as opposed to the lower-levels). 
Since such functioning should be causally closed with respect to the lower-level neural activity, attractor timeseries should exhibit the same predictive capacity towards the corresponding behavioral readouts as the timeseries of the lower-level neural activity (i.e., whole brain data), when lower-level "confounds" (such as reaction times) are neglected.

### 5. Attractor dynamics as a superior form of computation

We showed that attractor networks as a computational architecture, provides several advantages over implementing the same computations explicitly in higher-level nodes of bipartite systems, including cost-efficiency, robustness and - if higher order interactions are allowed - exponential scalability .

Attractor-based computation provides three key benefits: 1) Fault tolerance through distributed pattern storage, 2) Exponential memory capacity via higher-order interactions, and 3) Metabolic efficiency by reusing existing connectivity. Compared to explicit hierarchical architectures, this approach reduces wiring costs while increasing adaptability - critical advantages for resource-constrained biological systems.


---------------

## The Evolutionary Advantage of Attractor-Based Computation
Our findings reveal that embedding computations into global attractor dynamics offers three key advantages that make this architecture particularly suitable for biological systems like the brain. First, the distributed nature of attractor patterns provides inherent fault tolerance - there is no single physical "integrator node" that could become a critical failure point, and the informational weights are redundantly encoded across connections ([](#fig-economy)A). This matches the brain's well-documented resilience to localized damage {cite:p}`10.1038/nrn3506`. Second, the memory capacity of attractor networks scales favorably compared to explicit hierarchical architectures, with modern formulations achieving exponential storage through higher-order interactions {cite:p}`https://doi.org/10.48550/arXiv.1606.01164`. Third, the self-organizing nature of attractor dynamics enables emergent computational capabilities that transcend individual component functions, allowing biological systems to adaptively reconfigure their information processing without centralized control.

## Plausibility as a Model for Large-Scale Brain Dynamics
The proposed framework aligns remarkably well with several observed features of brain dynamics. First, the analytical reconstruction approach (Eq. [](reconstruction-analitical)) through eigenvector decomposition of functional connectivity matrices explains the success of PCA and related techniques in identifying large-scale brain networks from fMRI data {cite:p}`10.1002/hbm.20074`. Second, the duality between predictive coding and Bayesian causal inference implemented through attractor dynamics ([](#fig-bci), [](#si-predcode)) provides a unified mechanism for hierarchical processing observed across sensory and association cortices {cite:p}`10.1038/nrn2308`. Third, the emergent nature of attractor timeseries (Eq. [](attractor-timeseries)) offers a principled explanation for the phenomenon of computational closure observed in neural systems {cite:p}`https://doi.org/10.48550/arXiv.2402.09090`, where macroscopic brain states exhibit causal independence from their microscopic components.

## Implications for Artificial Intelligence
Our results suggest three key directions for AI development: 1) Energy-efficient neuromorphic architectures could leverage the inherent stability of attractor dynamics for robust in-memory computing {cite:p}`10.1038/s41928-023-01020-z`. 2) Modern Hopfield networks with higher-order interactions ([](#fig-economy)B) provide a biologically-inspired path toward systems with exponential memory capacity {cite:p}`https://doi.org/10.48550/arXiv.2008.06996`. 3) The Free Energy Principle provides a unified training framework where weights and precision parameters (β_j in Eq. [](prior-integrator)) can be optimized through variational inference, combining Hebbian-like local learning with global energy minimization {cite:p}`10.1016/j.neunet.2023.04.008`. This could enable self-supervised systems that automatically balance perception and action through attractor dynamics.

## Training Attractor Landscapes Through Free Energy Minimization
The FEP provides a natural mechanism for optimizing attractor landscapes through dual updates to: 1) Connection weights W_ij via gradient descent on variational free energy (Eq. [](f-complexity-accuracy)), implementing a form of predictive coding {cite:p}`10.1016/j.tics.2019.07.002`; and 2) Precision parameters β_j through precision-weighted prediction errors {cite:p}`10.1016/j.neubiorev.2016.06.022`. Biologically, this corresponds to combined synaptic plasticity (weight changes) and neuromodulation (precision adjustment). The resulting system automatically tunes its attractor basins to minimize surprise about sensory inputs while maintaining metabolic efficiency - a process observable as spike-timing dependent plasticity in neural circuits {cite:p}`10.1038/nn.2369`.


## Conclusion
By rigorously establishing the equivalence between far-from-equilibrium dynamics in complex systems and computational architectures like RBMs/Hopfield networks, we bridge a fundamental gap in dynamical systems theory. Our results demonstrate that attractor dynamics are not merely epiphenomena of neural activity, but rather constitute a fundamental computational primitive implemented by both biological and artificial systems. The framework provides: 1) New analysis tools for reconstructing computational states from neural data; 2) Design principles for robust AI systems; and 3) A mathematical foundation for understanding intelligence as an emergent property of self-organizing dynamical systems. Future work should focus on experimental validation through large-scale neural recordings and developing next-generation AI architectures that exploit these principles for open-ended learning.


> Todo: discuss these too: 
> [](https://doi.org/10.1016/j.neunet.2023.11.027)
> [](https://doi.org/10.1016/j.neuroimage.2022.119595)


### Derivation of the Learning Rule from the Variational Free Energy

Starting from the VFE for node *i*,
$$
F = \mathbb{E}_{q(\sigma_i)}\Bigl[\ln q(\sigma_i) - \ln P(\sigma_i \mid \sigma_{\backslash i})\Bigr],
$$
and parameterizing the distributions as
$$
\ln q(\sigma_i) = b_q\,\sigma_i - \phi(b_q), \qquad
\ln P(\sigma_i \mid \sigma_{\backslash i}) = u_i\,\sigma_i - \phi(u_i),
$$
with
$$
u_i = b + \sum_{j\neq i} J_{ij}\,\sigma_j,
$$
we obtain
$$
F = \mathbb{E}_{q(\sigma_i)}\Bigl[(b_q - u_i)\,\sigma_i\Bigr] + \phi(u_i) - \phi(b_q).
$$

At equilibrium (i.e. when \(b_q = u_i\)), we have \(\mathbb{E}_q[\sigma_i] = L(u_i)\), where \(L(u_i) = \phi'(u_i)\) is the Langevin function.

For learning, we use a stochastic (sample-based) estimate by replacing the expectation \(\mathbb{E}_q[\sigma_i]\) with the instantaneous value \(\sigma_i\). A perturbation \(\delta J_{ij}\) produces a change \(\delta u_i = \sigma_j\,\delta J_{ij}\), and by applying the chain rule we get:
$$
\frac{dF}{dJ_{ij}} = \frac{\partial F}{\partial u_i}\,\frac{\partial u_i}{\partial J_{ij}} = \bigl[L(u_i)-\sigma_i\bigr]\,\sigma_j.
$$

Using gradient descent to minimize \(F\) thus yields the weight update rule:
$$
\Delta J_{ij} \propto -\frac{dF}{dJ_{ij}} = \bigl[\sigma_i - L(u_i)\bigr]\,\sigma_j,
$$
or equivalently,
$$
J_{ij} \leftarrow J_{ij} + \eta\,[\sigma_i - L(u_i)]\,\sigma_j,
$$
which is identical in form to the rule derived from the expected free energy.

