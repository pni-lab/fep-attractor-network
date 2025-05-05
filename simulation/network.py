from abc import ABC, abstractmethod
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph, get_bipartite_layout
import warnings
from copy import deepcopy
from tqdm import tqdm
import scipy.optimize # Added for inverse Langevin root finding
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict, Any # Added for type hinting


def coth(x):
    """
    Computes the hyperbolic cotangent of the input array x.

    Parameters:
    x (array-like): Input values for which to compute the hyperbolic cotangent.

    Returns:
    array-like: The hyperbolic cotangent of the input values.
    """
    x = np.asarray(x)
    x = np.clip(x, -700, 700)  # clip to avoid overflow
    return np.cosh(x) / np.sinh(x)


def Langevin(x):
    """
    Computes the Langevin function, which is the expected value of the 
    continuous Bernoulli (CB) distribution given an input array x.

    The function handles the singularity at x = 0 by using a mask to 
    apply the computation only where x is non-zero. For non-zero x, 
    it calculates the Langevin function as the hyperbolic cotangent 
    of x minus the reciprocal of x.

    Parameters:
    x (array-like): Input values for which to compute the Langevin function.

    Returns:
    array-like: The Langevin function values for the input array.
    """
    x = np.asarray(x).astype(np.float64)
    result = np.zeros_like(x)
    mask = x != 0
    result[mask] = coth(x[mask]) - 1/x[mask]
    return result


def dLangevin(x): 
    """
    Computes the derivative of the Langevin function.

    The derivative of the Langevin function is calculated as:
    dL/dx = 1/x^2 - 1/sinh(x)^2

    The function handles the singularity at x = 0 by simply returning 1/3 
    when x is close to zero.

    Parameters:
    x (float or array-like): Input values for which to compute the derivative 
                             of the Langevin function.

    Returns:
    float or array-like: The derivative of the Langevin function for the input values.
    """
    if np.isclose(x, 0):
        return 1/3
    else:
        return 1/x**2 - 1/np.sinh(x)**2


def inverse_Langevin(x):
    """
    Approximates the inverse of the Langevin function.

    The inverse Langevin function does not have a closed-form solution.
    This function uses a popular approximation that works well for input
    values between -1 and 1.

    Parameters:
    x (float or array-like): Input values for which to compute the approximate
                             inverse Langevin function.

    Returns:
    float or array-like: The approximate inverse Langevin function values for
                         the input values.
    """
    return x * (3 - x*x) / (1 - x*x)


def inverse_Langevin_high_precision(y): #not used currently
    """
    Computes a high-precision approximation of the inverse Langevin function.
    
    The inverse Langevin function does not have a closed-form solution.
    This function uses a combination of Padé approximant for moderate values
    and numerical root finding for values close to the boundaries (-1, 1).
    
    Parameters:
    y (float or array-like): Input values for which to compute the inverse
                             Langevin function. Values should be in range [-1, 1].
    
    Returns:
    float or array-like: The inverse Langevin function values for the input values
                         with high precision.
    
    Note:
    This function is currently not used in the main implementation.
    """
    y = np.clip(y, -1 + 1e-9, 1 - 1e-9) # Avoid exact +/- 1 for numerical stability

    # Pade approximant L^-1(y) ~ y * (3 - y^2) / (1 - y^2) is good for |y| <~ 0.85
    # For larger |y|, use numerical root finding
    if np.abs(y) < 0.85:
        return y * (3.0 - y**2) / (1.0 - y**2)
    else:
        # Use root finding for higher precision when near +/- 1
        func_to_solve = lambda x: Langevin(x) - y
        # Bracket the root: for y > 0, L(x) increases, so find x where L(x)=y.
        # For y near 1, x is large. For y near -1, x is large negative.
        # Start search from the Pade result as an initial guess.
        initial_guess = y * (3.0 - y**2) / (1.0 - y**2)
        try:
            # Brentq is generally robust
            result = scipy.optimize.brentq(func_to_solve, initial_guess - 100, initial_guess + 100, xtol=1e-6, rtol=1e-6)
            return result
        except ValueError:
             # Fallback if root not bracketed, happens if initial guess is way off or y is extreme
             if y > 0:
                 return scipy.optimize.brentq(func_to_solve, 0.1, 700, xtol=1e-6, rtol=1e-6) # Search positive
             else:
                 return scipy.optimize.brentq(func_to_solve, -700, -0.1, xtol=1e-6, rtol=1e-6) # Search negative


def phi(b):
    """
    Calculates the log-normalization constant for the Continuous Bernoulli distribution.
    
    This function computes log(2*sinh(b)/b) which is the log of the normalization
    constant for the Continuous Bernoulli distribution.
    
    Parameters:
    b (float or array-like): The parameter of the Continuous Bernoulli distribution.
                            Can be a scalar or numpy array.
    
    Returns:
    float or array-like: The log-normalization constant for the given parameter(s).
                        Returns ln(2) when b is close to zero.
    
    Note:
    Special care is taken for values of b close to zero to ensure numerical stability.
    """
    b = np.clip(b, -700, 700) # Prevent overflow
    # Handle the case where b is a scalar or array
    if np.isscalar(b):
        if np.isclose(b, 0):
            return np.log(2.0)  # Limit as b->0 is ln(2)
    else:
        # For array inputs, create a mask for values close to zero
        zero_mask = np.isclose(b, 0)
        if np.any(zero_mask):
            result = np.empty_like(b, dtype=float)
            result[zero_mask] = np.log(2.0)  # Limit as b->0 is ln(2)
            return result
    # Handle potential division by zero if b is exactly 0, although clip should prevent this
    # sinh(b)/b -> 1 as b->0. So phi(0) = ln(2*1) = ln(2).
    # Need to be careful with very small b causing sinh(b)/b inaccuracies.
    abs_b = np.abs(b)
    # Handle scalar and array inputs differently
    if np.isscalar(abs_b):
        if abs_b < 1e-6:  # Use Taylor expansion for sinh(x)/x near 0: 1 + x^2/6 + ...
            sinh_b_over_b = 1.0 + b**2 / 6.0
        else:
            sinh_b_over_b = np.sinh(b) / b
    else:
        # For array inputs, create masks for small and normal values
        small_b_mask = abs_b < 1e-6
        # Apply Taylor expansion where |b| is small
        sinh_b_over_b = np.ones_like(b)
        sinh_b_over_b[small_b_mask] = 1.0 + b[small_b_mask]**2 / 6.0
        # Direct calculation for other values
        sinh_b_over_b[~small_b_mask] = np.sinh(b[~small_b_mask]) / b[~small_b_mask]

    return np.log(2.0 * sinh_b_over_b)


def CB(x, b, logodds=True):
    """
    Evaluates the Continuous Bernoulli probability density function.
    
    Parameters:
    x (float or array-like): Input values in the range [-1, 1].
    b (float or array-like): The parameter of the Continuous Bernoulli distribution.
                            Can be a scalar or numpy array.
    logodds (bool, optional): If True, b is interpreted as log-odds. If False, 
                             b is interpreted as probability. Default is True.
    
    Returns:
    float or array-like: The probability density at the given input value(s).
    
    Note:
    When b is close to zero, the distribution approaches a uniform distribution.
    """
    if not logodds:
        b=np.log(b/(1-b))
    if np.isclose(b, 0):
        return np.ones_like(x)/2
    else:
        return b * np.exp(b*x) / (2*np.sinh(b)) 
    
def CB_entropy(b):
    """
    Calculates the entropy of a Continuous Bernoulli distribution.
    
    Parameters:
    b (float or array-like): The parameter of the Continuous Bernoulli distribution.
                            Can be a scalar or numpy array. Interpreted as log-odds.
    
    Returns:
    float or array-like: The entropy of the Continuous Bernoulli distribution.
    
    Note:
    When b is close to zero, the entropy approaches ln(2), which is the entropy
    of a uniform distribution on [-1, 1].
    """
    b = np.asarray(b) # Ensure input is numpy array
    entropy = np.zeros_like(b, dtype=float) # Initialize entropy array
    # Handle non-zero bias values
    mask = ~np.isclose(b, 0)
    b_m = b[mask]
    # Clip large b values to prevent overflow in sinh and avoid issues near 0
    b_clipped = np.clip(b_m, -700, 700)
    # Calculate Langevin function (mean)
    L = Langevin(b_clipped) 
    # Calculate sinh(b)/b, handling b=0 case
    abs_b = np.abs(b_clipped)
    sinh_b_over_b = np.ones_like(b_clipped) # Initialize with limit for b->0
    # Use Taylor expansion for very small |b| to avoid numerical issues
    taylor_mask = abs_b < 1e-6
    sinh_b_over_b[taylor_mask] = 1.0 + b_clipped[taylor_mask]**2 / 6.0
    # Use direct calculation otherwise
    non_taylor_mask = ~taylor_mask
    sinh_b_over_b[non_taylor_mask] = np.sinh(b_clipped[non_taylor_mask]) / b_clipped[non_taylor_mask]
    # Calculate entropy using the formula: ln(2*sinh(b)/b) - b*L(b)
    # Add small epsilon to log argument to prevent log(0)
    log_argument = 2.0 * sinh_b_over_b + 1e-12 
    entropy[mask] = np.log(log_argument) - b_clipped * L
    # Handle zero bias case (uniform distribution on [-1, 1])
    # Entropy of U(-1, 1) is ln(1 - (-1)) = ln(2)
    entropy[~mask] = np.log(2.0) 
    # Ensure no NaN/inf values sneak in
    entropy = np.nan_to_num(entropy, nan=np.log(2.0), posinf=0.0, neginf=0.0) 
    return entropy

    
def random_CB(b, rng=np.random.default_rng()):
    """
    Generates a random sample from a Continuous Bernoulli distribution.
    
    Parameters:
    b (float): The parameter of the Continuous Bernoulli distribution, interpreted as log-odds.
    rng (numpy.random.Generator, optional): Random number generator. Defaults to a new Generator.
    
    Returns:
    float: A random sample from the Continuous Bernoulli distribution in the range [-1, 1].
    
    Note:
    When b is close to zero, this approaches a uniform distribution on [-1, 1].
    For large absolute values of b, the function clips b to ±500 to prevent numerical issues.
    """
    if abs(b) > 500:
        b = np.sign(b) * 500

    # Handle the nearly-zero case (uniform distribution on [-1, 1])
    if np.isclose(b, 0):
        return rng.uniform(-1, 1)
    
    # Draw a uniform random number in [0, 1]
    u = rng.uniform(0, 1)
    # Inverse CDF for CB:
    # x = (1/L)*ln(exp(-L) + u*(exp(L)-exp(-L)))
    return (1.0 / b) * np.log(np.exp(-b) + u * (np.exp(b) - np.exp(-b)))


class State(ABC): # abstract base class for all nodes
    """
    Abstract base class representing a state in a particular partition of the network.
    
    This class serves as the foundation for different types of states in a deep particular partition,
    including sigma (deep internal states), mu (external states),
    s (sensory states), and a (action states) and implements their common behavior.
    
    Attributes:
        bias (float): The intrinsic bias of the node.
        activation (float): The current activation value of the node.
        afferent_connections (list): Incoming connections to this node.
        rng (numpy.random.Generator): Random number generator for stochastic processes.

    Note:
        This is an inefficient implementation (as it neither uses vectorization nor caching).
        It aims to yield a clear view on the underlying architecture (e.g. locality of rules).
    """
    bias: np.float64
    activation: np.float64
    afferent_connections: List[Any] # Could be more specific if Connection type exists
    rng: np.random.Generator

    def __init__(self, bias: float = 0, activation: float = 0, rng: Optional[np.random.Generator] = None):
        """
        Initialize the state.
        
        Parameters:
            bias (float): The intrinsic bias (prior evidence or sensory drive) of the state. Defaults to 0.
            activation (float): The initial activation value of the state. Defaults to 0.
            rng (numpy.random.Generator): Random number generator for stochastic processes.
                Defaults to a new Generator.
        """
        self.bias = np.float64(bias)
        self.activation = np.float64(activation)
        self.afferent_connections = []
        self.rng = rng if rng is not None else np.random.default_rng()

    def update(self, inverse_temperature=1, least_action=False):
        """
        Update the states's activation based on its inputs.
        The sum of the bias and all inputs yields the sufficient statistics of the state.
        The speed of the update is controlled by the inverse temperature.
        If least_action is True, the update is deterministic, and the state
        evolves over the expected value (least action). Otherwise, the update
        is stochastic.
        
        Parameters:
            inverse_temperature (float): Controls the speed of update and, indirectly,
             the stochasticity of the node. Higher values make the update faster 
             and more "deterministic". Default is 1.
            least_action (bool): If True, uses Langevin dynamics (expected value) for the update.
                If False, uses the node's sampling method (sample). Defaults to False.
        """
        input = self.bias
        for connection in self.afferent_connections:
            input += connection.activation
        if least_action:
            self.activation =  self._expected_value(input * inverse_temperature)
        else:
            self.activation = self._sample(input * inverse_temperature)

    @abstractmethod
    def _expected_value(self, input):
        """
        Compute the expected value of the state's activation.
        
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _sample(self, input):
        """
        Sample a new activation value based on the input.
        
        This method must be implemented by subclasses.
        
        Parameters:
            input (float): The total input to the node.
            
        Returns:
            float: The new activation value.
        """
        pass

    def connect(self, other):
        """
        Connect this node to another node.
        
        Parameters:
            other (State): The node to connect to.
        """
        self.connections.append(other)


class SigmaNode(State):
    """
    A node that implements a deep internal state with continuous Bernoulli distribution.
    
    Parameters:
        bias (float): The bias term (prior evidence or sensory drive) for the node.
        rng (numpy.random.Generator): Random number generator. Defaults to a new instance.

    Note:
        This is an inefficient implementation (as it neither uses vectorization nor caching).
        It aims to yield a clear view on the underlying architecture (e.g. locality of rules).
    """
    def __init__(self, bias: float, rng: Optional[np.random.Generator] = None):
        super().__init__(bias=bias, rng=rng)

    def _expected_value(self, input):
        """
        Compute the expected value using the Langevin function.
        
        Parameters:
            input (float): The total bias to the node.
            
        Returns:
            float: The expected activation value.
        """
        return Langevin(input)
    
    def _sample(self, input):
        """
        Sample a new activation value from the Continuous Bernoulli distribution.
        
        Parameters:
            input (float): The total input to the node.
            
        Returns:
            float: The sampled activation value.
        """
        return random_CB(input, rng=self.rng)
    

class BoundaryNode(State): # implements both sensory and action nodes
    """
    A node that implements boundary states in a deep particular partition.
    
    The BoundaryNode class can serve as both sensory and action nodes. They provide deterministic
    input/output interfaces to the sigma nodes. They transform their inputs by a weight factor.
    
    Parameters:
        weight (float): The weight factor to apply to inputs.

    Note:
        This is an inefficient implementation (as it neither uses vectorization nor caching).
        It aims to yield a clear view on the underlying architecture (e.g. locality of rules).
    """
    weight: float

    def __init__(self, weight: float):
        super().__init__(bias=0)
        self.weight = weight

    def _expected_value(self, input):
        """
        Not implemented for boundary nodes as they are deterministic by default.
        """
        raise NotImplementedError("Boundary nodes are deterministic by default!")
    
    def _sample(self, input):
        """
        Deterministically transform the input by the weight factor.
        
        Parameters:
            input (float): The input to the node.
            
        Returns:
            float: The weighted output.
        """
        output = input * self.weight  # always deterministic
        return output
    
    def train(self, sigma_a, sigma_b, antihebbian, learning_rate=1):
        """
        Update the weight using the FEP-derived learning rule.
        
        Parameters:
            sigma_a (float): Presynaptic activation.
            sigma_b (float): Postsynaptic activation.
            antihebbian (float): Anti-Hebbian term to regulate learning.
            learning_rate (float, optional): Rate of weight update. Defaults to 1.
        """
        self.weight += learning_rate * (sigma_a * sigma_b -  antihebbian)

class AttractorNetwork:
    """
    A network built from a deep particular partition that can exhibit attractor dynamics.
    
    The AttractorNetwork class implements a recurrent neural network with 
    deep particular partition structure. It performs inference (running
    the network to find stable states) and learning (adjusting connection weights) simultaneously.
    
    Attributes:
        sigmas (List[SigmaNode]): The list of internal state nodes (sigma nodes).
        connections (List[BoundaryNode]): The list of connection nodes (boundary nodes representing weights).
        rng (np.random.Generator): Random number generator for stochastic processes.

    Parameters:
        J (NDArray[np.float64]): Initial connection weight matrix where J[i, j] is the weight from node j to node i.
        biases (NDArray[np.float64]): External bias values for each sigma node.
        rng (np.random.Generator, optional): Random number generator. Defaults to None, which creates a new generator.
    
    Note:
        This is an inefficient implementation (as it neither uses vectorization nor caching).
        It aims to yield a clear view on the underlying architecture (e.g. locality of rules).
    """
    sigmas: List[SigmaNode]
    connections: List[BoundaryNode]
    rng: np.random.Generator

    def __init__(self, J: NDArray[np.float64], biases: NDArray[np.float64], rng: Optional[np.random.Generator] = None):
        """
        Initializes the AttractorNetwork.

        Args:
            J (NDArray[np.float64]): Weight matrix (J[i, j] = weight from j to i).
            biases (NDArray[np.float64]): Bias for each sigma node.
            rng (np.random.Generator, optional): Random number generator. Defaults to a new one if None.
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sigmas = [SigmaNode(bias, rng=self.rng) for bias in biases]
        self.connections = []
        n_nodes = len(self.sigmas)

        if J.shape != (n_nodes, n_nodes):
            raise ValueError(f"Weight matrix J shape {J.shape} incompatible with number of biases {n_nodes}")
        if biases.shape != (n_nodes,):
             raise ValueError(f"Biases shape {biases.shape} incompatible with number of nodes {n_nodes}")

        # Create BoundaryNode for each connection weight J[i, j]
        # This node receives input from sigma_j and sends weighted output to sigma_i
        for i, target_node in enumerate(self.sigmas):
            afferents_for_target = []
            for j, source_node in enumerate(self.sigmas):
                # Create a boundary node representing the connection weight J[i, j]
                connection_node = BoundaryNode(weight=J[i, j])
                # Connect the source sigma node to this boundary node
                connection_node.afferent_connections.append(source_node)
                # Store this connection node
                self.connections.append(connection_node)
                # Add this connection node as an afferent to the target sigma node
                afferents_for_target.append(connection_node)
            target_node.afferent_connections = afferents_for_target # Assign all afferents at once

    def update(self, inverse_temperature: float = 1.0, least_action: bool = False, learning_rate: float = 0.0) -> None:
        """
        Update the network by sequentially updating each node and training connections.

        This method performs a full network update cycle:
        1. Computes input potentials for each sigma node by updating afferent BoundaryNodes.
        2. Updates each sigma node's activation state based on total input.
        3. If learning_rate > 0, trains connection weights using the FEP-based Hebbian/anti-Hebbian rule.

        Parameters:
            inverse_temperature (float, optional): Controls update stochasticity. Defaults to 1.0.
            least_action (bool, optional): Use deterministic (Langevin) updates if True. Defaults to False.
            learning_rate (float, optional): Weight update rate. No learning if 0. Defaults to 0.0.
        """
        n_nodes = len(self.sigmas)
        # Store total input potential for each sigma node (excluding bias) for learning rule
        h = np.zeros(n_nodes, dtype=np.float64)

        # 1. Update connection nodes (BoundaryNode) and calculate input potentials (h)
        # Must be done before updating sigma nodes, as sigma updates depend on these potentials
        for i, node in enumerate(self.sigmas):
            for connection in node.afferent_connections:
                # BoundaryNode update computes weighted output from its source sigma node
                connection.update() 
                h[i] += connection.activation # Accumulate weighted inputs to node i

        # 2. Update sigma node activations
        for i, node in enumerate(self.sigmas):
            node.update(inverse_temperature=inverse_temperature, least_action=least_action)

        # 3. Adjust connection weights (train BoundaryNodes)
        if learning_rate > 0:
            current_activations = np.array([n.activation for n in self.sigmas], dtype=np.float64)
            # Precompute Langevin of input potentials for anti-Hebbian term
            L_h = Langevin(h) # Langevin of total input *before* adding bias

            connection_idx = 0 # Keep track of which connection in the flat self.connections list
            for i, target_node in enumerate(self.sigmas):
                for j, source_node in enumerate(self.sigmas):
                    connection_node = self.connections[connection_idx] # Assumes specific order from __init__
                    if i != j:  # Don't train self-connections (Kanter & Sompolinsky, 1987)
                        # FEP learning rule: dW/dt = learning_rate * (presynaptic * postsynaptic - antiHebbian)
                        antiHebbian_term = L_h[i] * source_node.activation
                        # Call train on the specific BoundaryNode representing J[i, j]
                        connection_node.train(
                            sigma_a=target_node.activation, # Postsynaptic activation
                            sigma_b=source_node.activation, # Presynaptic activation
                            antihebbian=antiHebbian_term,
                            learning_rate=learning_rate
                        )
                    connection_idx += 1

    def get_J(self) -> NDArray[np.float64]:
        """
        Get the current weight matrix of the network.

        Returns:
            NDArray[np.float64]: A 2D array J where J[i, j] is the weight
                                 of the connection from node j to node i.
        """
        n_nodes = len(self.sigmas)
        # Assumes self.connections stores weights in row-major order (J[0,0], J[0,1], ..., J[n-1,n-1])
        return np.array([connection.weight for connection in self.connections]).reshape(n_nodes, n_nodes)

    def set_J(self, J: NDArray[np.float64]) -> None:
        """
        Set the weight matrix of the network.

        Parameters:
            J (NDArray[np.float64]): A 2D array J where J[i, j] is the weight
                                     of the connection from node j to node i.
        """
        n_nodes = len(self.sigmas)
        if J.shape != (n_nodes, n_nodes):
             raise ValueError(f"Input J shape {J.shape} does not match network size {n_nodes}")
        # Flatten J to match the order in self.connections
        flat_J = J.flatten()
        if len(flat_J) != len(self.connections):
             raise RuntimeError("Mismatch between flattened J size and number of connections") # Should not happen if J shape is checked

        for i, connection in enumerate(self.connections):
            connection.weight = flat_J[i]

    def _get_variational_params(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """ Helper to get current state needed for VFE calculations """
        m = np.array([n.activation for n in self.sigmas], dtype=np.float64) # Current mean activations
        J = self.get_J() # Current weights
        b = np.array([n.bias for n in self.sigmas], dtype=np.float64)  # Prior (generative) biases
        # Calculate effective bias for the variational distribution q
        b_q = b + J @ m # Variational biases: prior bias + message passing term
        return m, J, b, b_q

    def complexity(self) -> float:
        """
        Calculate the complexity term (KL divergence) of the variational free energy.

        Computes D_KL(q || p), the KL divergence between the approximate posterior q
        (parameterized by b_q) and the prior p (parameterized by b).

        Returns:
            float: The complexity value (summed over all nodes).
        """
        _m, _J, b, b_q = self._get_variational_params()

        # Use KL divergence formula: D_KL(q||p) = E_q[log(q/p)]
        # For CB: D_KL(CB(x|b_q) || CB(x|b)) = log(b_q/(2sinh(b_q))) - log(b/(2sinh(b))) + (b_q - b) * L(b_q)

        _m, _J, b, b_q = self._get_variational_params()
        
        # Protect against overflow by handling cases where b or b_q are near zero
        epsilon = np.finfo(np.float64).tiny  # Small constant to avoid division by zero or overflow
        # Ensure b and b_q are not exactly zero to prevent overflow in log and sinh
        b_safe = np.where(np.abs(b) < epsilon, epsilon, b)
        b_q_safe = np.where(np.abs(b_q) < epsilon, epsilon, b_q)

        # Use the safe versions of b and b_q in the calculations
        return np.sum((np.log(b_q_safe/np.sinh(b_q_safe)) + b_q_safe * Langevin(b_q_safe)) - \
               (np.log(b_safe/np.sinh(b_safe)) + b_safe * Langevin(b_q_safe)))

    def accuracy(self) -> float:
        """
        Calculate the accuracy term (expected log-likelihood) of the variational free energy.

        Computes E_q[log p(data)], where data is implicitly represented by the biases `b`.

        Returns:
            float: The accuracy value.
        """
        _m, J, b, b_q = self._get_variational_params()
        Lbq = Langevin(b_q) # Expected activations under q
        accuracy_term = b.dot(Lbq) + Lbq.T @ J @ Lbq
        return np.nan_to_num(accuracy_term)

    def vfe(self) -> float:
        """
        Calculate the variational free energy (VFE = Complexity - Accuracy) of the network.

        Returns:
            float: The variational free energy value.
        """
        return self.complexity() - self.accuracy()

    def plot_network(self, node_size: float = 1.0, fig_size: Tuple[float, float] = (4, 4), symmetric: bool = True,
                     edge_cmap: str = 'bwr', node_cmap: str = 'coolwarm',
                     plot_edges: bool = True, plot_bias: bool = True, min_node_size: float = 1.0, edge_width: float = 2.0,
                     node_vminmax: Optional[Tuple[float, float]] = None) -> None:
        """
        Visualizes the network using netgraph.

        Nodes are sigma nodes. Edges represent weighted connections (BoundaryNodes).

        Args:
            node_size (float): Base size for nodes.
            fig_size (Tuple[float, float]): Figure size.
            symmetric (bool): If True, average weights for bi-directional edges for plotting.
            edge_cmap (str): Colormap for edge weights.
            node_cmap (str): Colormap for node activations.
            plot_edges (bool): Whether to draw edges.
            plot_bias (bool): If True, node size reflects bias.
            min_node_size (float): Minimum node size when `plot_bias` is True.
            edge_width (float): Width of edges.
            node_vminmax (Optional[Tuple[float, float]]): Min/max values for node activation colormap normalization.
                                                        If None, uses symmetric range around max absolute activation.
        """
        plt.figure(figsize=fig_size)
        G = nx.DiGraph()
        n_nodes = len(self.sigmas)

        # Add nodes with attributes
        for idx, sigma in enumerate(self.sigmas):
            G.add_node(idx, bias=sigma.bias, activation=sigma.activation)

        # Get weights and determine normalization range
        weights = self.get_J() # Get the current weight matrix
        epsilon = 1e-6
        max_abs_weight = np.max(np.abs(weights))
        edge_norm = plt.Normalize(vmin=-max_abs_weight - epsilon, vmax=max_abs_weight + epsilon)
        edge_colors_dict: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}

        # Add edges based on weights
        for i in range(n_nodes): # Target node index
            for j in range(n_nodes): # Source node index
                weight = weights[i, j] # Weight from j to i
                if i == j: continue # Skip self-loops for clarity unless desired

                if symmetric:
                    # If edge (i, j) exists (meaning we added (j, i) previously), average weights
                    if G.has_edge(i, j):
                         existing_weight = G[i][j]['weight']
                         new_weight = (existing_weight + weight) / 2.0
                         G[i][j]['weight'] = new_weight
                         # Update color based on averaged weight
                         edge_colors_dict[(i, j)] = plt.cm.get_cmap(edge_cmap)(edge_norm(new_weight))
                    # Otherwise, add edge (j, i)
                    elif not G.has_edge(j, i):
                         G.add_edge(j, i, weight=weight)
                         edge_colors_dict[(j, i)] = plt.cm.get_cmap(edge_cmap)(edge_norm(weight))
                else:
                    # Add directed edge (j, i)
                    G.add_edge(j, i, weight=weight)
                    edge_colors_dict[(j, i)] = plt.cm.get_cmap(edge_cmap)(edge_norm(weight))

        # Node colors based on activation
        activations = np.array([node.activation for node in self.sigmas])
        if node_vminmax is None:
            max_abs_activation = np.max(np.abs(activations))
            node_norm = plt.Normalize(vmin=-max_abs_activation - epsilon, vmax=max_abs_activation + epsilon)
        else:
            node_norm = plt.Normalize(vmin=node_vminmax[0], vmax=node_vminmax[1])
        node_colors_array = plt.cm.get_cmap(node_cmap)(node_norm(activations))
        node_colors_dict = {idx: tuple(node_colors_array[idx]) for idx in range(n_nodes)}

        # Node sizes
        if not plot_bias:
            node_sizes_dict = {idx: node_size for idx in range(n_nodes)}
        else:
            biases = np.array([ns.bias for ns in self.sigmas])
            min_bias = np.min(biases)
            max_bias = np.max(biases)
            # Normalize bias to affect size positively
            if max_bias > min_bias:
                 bias_range = max_bias-min_bias
                 normalized_bias = (biases - min_bias) / bias_range if bias_range > 0 else np.zeros_like(biases)
                 node_sizes_dict = {idx: min_node_size + node_size * normalized_bias[idx] for idx in range(n_nodes)}
            else: # All biases are the same
                 node_sizes_dict = {idx: min_node_size + node_size * 0.5 for idx in range(n_nodes)} # Use a mid-size


        # Hide edges if requested
        final_edge_colors = edge_colors_dict
        if not plot_edges:
            final_edge_colors = {edge: (1, 1, 1, 0) for edge in G.edges()} # Transparent

        # Create and draw the graph using netgraph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore netgraph runtime warnings
            _ = Graph(G, node_layout='circular', edge_layout='curved',
                      node_size=node_sizes_dict,
                      node_color=node_colors_dict,
                      node_edge_width=0, # No border around nodes
                      edge_color=final_edge_colors,
                      edge_width=edge_width
                     )
        plt.show()


def relax(network: AttractorNetwork, # Type hint for network
          input: Optional[NDArray[np.float64]] = None,
          bias: Optional[NDArray[np.float64]] = None,
          inverse_temperature: float = 1.0,
          least_action: bool = False,
          max_steps: int = 1000,
          tol: float = 1e-6) -> Tuple[NDArray[np.float64], int]:
    """
    Relax the network to a stable attractor state by iteratively updating until convergence.

    Parameters:
        network (AttractorNetwork): The network to relax.
        input (Optional[NDArray[np.float64]]): Initial activations. Uses current if None.
        bias (Optional[NDArray[np.float64]]): Biases to set. Uses current if None.
        inverse_temperature (float): Update stochasticity control.
        least_action (bool): Use deterministic updates if True.
        max_steps (int): Maximum update steps.
        tol (float): VFE change tolerance for convergence.

    Returns:
        Tuple[NDArray[np.float64], int]: Final activations (NaN if no convergence) and steps taken.
    """
    # house keeping
    nw = deepcopy(network)
    if input is not None:
        if len(input) != len(nw.sigmas):
             raise ValueError("Input length mismatch")
        for i, node in enumerate(nw.sigmas):
            node.activation = input[i]
    if bias is not None:
        if len(bias) != len(nw.sigmas):
             raise ValueError("Bias length mismatch")
        for i, node in enumerate(nw.sigmas):
            node.bias = bias[i]

    prev_vfe = nw.vfe()
    for i in range(max_steps):
        nw.update(inverse_temperature=inverse_temperature, least_action=least_action, learning_rate=0.0) # Ensure no learning during relaxation
        current_vfe = nw.vfe()
        if np.abs(current_vfe - prev_vfe) < tol: # Check absolute difference
            return np.array([node.activation for node in nw.sigmas]), i + 1 # Return steps taken (1-based)
        prev_vfe = current_vfe
    # Did not converge
    return np.full(len(nw.sigmas), np.nan), max_steps
