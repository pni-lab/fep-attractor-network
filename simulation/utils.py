from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pingouin as pg
from simulation.network import AttractorNetwork, Langevin, relax
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm import tqdm

def fetch_digits_data():
    """
    Fetches the digits dataset from sklearn and displays the first 20 images.

    Returns:
        sklearn.utils.Bunch: The digits dataset object containing data, images, target, etc.
    """
    digits = datasets.load_digits(as_frame=True)
    _, axes = plt.subplots(nrows=1, ncols=20, figsize=(10, 3))
    # Display the first 20 digits as an example
    for ax, image, label in zip(axes, digits.images[:20], digits.target[:20]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("%i" % label)
    plt.show() # Show the plot
    return digits

def preprocess_digits_data(digits):
    """
    Preprocesses the digits dataset by squaring the pixel values and normalizing the data.

    Args:
        digits: The digits dataset object containing data, images, target, etc.

    """
    data = digits.data.values
    data = np.power(data, 2)
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

    train_data = data[:10]
    test_data = data[10:]
    
    train_data = torch.from_numpy(train_data).double()
    test_data = torch.from_numpy(test_data).double()

    # visualize
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < train_data.shape[0]:
            image = train_data[i].reshape(8, 8) + torch.normal(0, 0.1, (8,8))

            for j in range(image.shape[0]):
                for k in range(image.shape[1]):
                    image[j, k] = Langevin(image[j, k])

            ax.imshow(image, cmap="coolwarm", interpolation="nearest")
            ax.set_axis_off()
        else:
            ax.set_visible(False)
    plt.show()
    plt.figure(figsize=(3, 3))
    sns.histplot(train_data.flatten().numpy())

    return train_data, test_data


def continous_inference_and_learning(nw: EfficientAttractorNetwork, 
                                     data: torch.Tensor, 
                                     inverse_temperature: float = 1.0, 
                                     learning_rate: float = 0.001, 
                                     num_steps: int = 100):
    """
    Wrapper function to continuous inference and learning

    Runs the networks with the same pattern (input biases) for a given number of iterations.
    Note that training and inference happens simultaneously in this architecture.

    Args:
        nw: EfficientAttractorNetwork instance
        data: torch.Tensor of shape (num_nodes,)
        inverse_temperature: controls the temperature during the state activation update
        learning_rate: controls the step size of the weight updates
        num_steps: controls the number of steps in the current epoch (with the same pattern)
    """
    
    weight_change = [] # how much has the weight matrix changed
    prev_J = nw.get_J().clone()

    # put in the data as biases (e.g. external sensory drive or internal computations)
    nw.biases.data = data.to(nw.J.device)

    activations = []
    vfe = []
    accuracy = []
    complexity = []
    for i in range(num_steps):
        nw.update(inverse_temperature=inverse_temperature, learning_rate=learning_rate, least_action=False)
        activations.append(nw.activations.cpu().clone())
        weight_change.append(torch.sum(torch.pow(nw.get_J() - prev_J, 2)).item())
        prev_J = nw.get_J().clone()
        accuracy.append(nw.accuracy())
        complexity.append(nw.complexity())
        vfe.append(complexity[-1] - accuracy[-1]) # also available as nw.vfe()

    # clean up the network, just in case
    nw.biases.data.zero_()

    return activations, weight_change, accuracy, complexity, vfe


def run_network(data: torch.Tensor, 
                evidence_level: float, 
                inverse_temperature: float, 
                learning_rate: float,
                num_epochs: int, 
                num_steps: int,
                progress_bar: bool = True,
                device: str = 'cpu'):
    """
    Main function to run the network

    Runs the network for the given input data with the given parameters:
        - `data`: the input data
        - `evidence_level`: the evidence level, scaling the input data
        - `inverse_temperature`: the temperature during the state activation update
        - `learning_rate`: the step size of the weight updates
        - `num_epochs`: the number of epochs
        - `num_steps`: the number of steps in each epoch

    Returns:
        - network
        - vfe-curve
        - accuracy-curve
        - complexity-curve
        - pattern-curve (which pattern was used as bias, for each epoch)
        - weight-change-curve
    """
    
    data = data.clone() * evidence_level

    # initialize empty network
    n_nodes = data.shape[1]
    nw = EfficientAttractorNetwork(
        J=torch.zeros((n_nodes, n_nodes), dtype=torch.float64), 
        biases=torch.zeros(n_nodes, dtype=torch.float64)
    ).to(device)

    weight_change = []
    pattern = []
    vfe = []
    accuracy = []
    complexity = []
    
    rng = np.random.default_rng()

    for i in tqdm(range(num_epochs), disable=not progress_bar):
        # select a pattern randomly
        di = rng.integers(0, data.shape[0])
        pattern.append(di)
        _, e, acc, comp, this_vfe = continous_inference_and_learning(
            nw, 
            data[di], 
            inverse_temperature=inverse_temperature, 
            learning_rate=learning_rate, 
            num_steps=num_steps)
        weight_change += e
        vfe += this_vfe
        accuracy += acc
        complexity += comp

    return nw, weight_change, pattern, accuracy, complexity, vfe


def evaluate_reconstruction_accuracy(nw: EfficientAttractorNetwork, 
                                     data: torch.Tensor, 
                                     sample: bool,
                                     signal_strength: float,
                                     num_trials: int, 
                                     SNR: float, 
                                     inverse_temperature: float, 
                                     num_steps: int, 
                                     plot: bool = True):
    """
    Helper function to evaluate reconstruction accuracy
    """
    r2_test_original = []
    r2_reconstructed_original = []
    
    device = nw.J.device
    data = data.to(device)
    rng = np.random.default_rng()


    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(10, 3))
    
    for i in tqdm(range(num_trials), disable=not plot):
        if sample:
            idx = rng.integers(0, data.shape[0])
        else:
            idx = i % data.shape[0]
            
        original_pattern = data[idx] * signal_strength
        noise = torch.normal(0, original_pattern.std()/SNR, original_pattern.shape, device=device)
        test_pattern = original_pattern + noise
        
        acts, _, _, _, _ = continous_inference_and_learning(nw, data=test_pattern, 
                                                            inverse_temperature=inverse_temperature, 
                                                            learning_rate=0.0, 
                                                            num_steps=num_steps)
        
        mean_activity = torch.mean(torch.stack(acts), axis=0)
        
        test_pattern_np = test_pattern.cpu().numpy()
        original_pattern_np = original_pattern.cpu().numpy()
        mean_activity_np = mean_activity.cpu().numpy()
        
        r2_test_original.append(np.round(np.corrcoef(test_pattern_np, original_pattern_np)[0, 1]**2, 3))
        r2_reconstructed_original.append(np.round(np.corrcoef(mean_activity_np, original_pattern_np)[0, 1]**2, 3))

        if plot and i < 10:
            axes[0, i].imshow(test_pattern_np.reshape(8, 8), cmap="gray")
            axes[0, i].set_axis_off()
            axes[1, i].imshow(mean_activity_np.reshape(8, 8), cmap="gray")
            axes[1, i].set_axis_off()
            axes[2, i].imshow(original_pattern_np.reshape(8, 8), cmap="gray")
            axes[2, i].set_axis_off()
    if plot:
        plt.show()

    if plot:
        df = pd.DataFrame({"input vs.": np.hstack([np.repeat("original", num_trials), 
                                                                     np.repeat("reconstructed", num_trials)]), 
                       "R^2": np.hstack([r2_test_original, r2_reconstructed_original]),
                       "trial": np.hstack([np.arange(num_trials), np.arange(num_trials)])})
        plt.figure(figsize=(4, 2))
        pg.plot_paired(df, within="input vs.", dv="R^2", subject="trial", pointplot_kwargs={"alpha": 0.2})
        plt.show()

        plt.figure(figsize=(10, 1))
        acts_np = torch.stack(acts).numpy()
        sns.lineplot(acts_np, legend=False, linestyle='-', alpha=0.5, linewidth = 1, palette='Spectral')
        plt.show()

    return r2_test_original, r2_reconstructed_original

def vfe(nw, acts):
    """
    Small helper function to compute the VFE.
    """
    nw.activations = torch.tensor(acts, dtype=torch.float64, device=nw.J.device)
    return np.round( nw.vfe(), 4 )

def get_deterministic_attractors(nw: EfficientAttractorNetwork,
                                 data: torch.Tensor, 
                                 noise_levels=(0.0, 0.1, 0.2, 0.5),
                                 inverse_temperature=1,
                                 plot=True):
    
    device = nw.J.device
    data = data.to(device)
    all_attractors = []

    for noise in noise_levels:
        attractors = []
        if plot:
            fig, axes = plt.subplots(2, data.shape[0], figsize=(20, 4))
            print(f"  ** Noise: {noise}")
        for i in tqdm(range(data.shape[0]), disable=not plot):
            
            rng = torch.Generator(device=device)
            
            nw_relax = EfficientAttractorNetwork(nw.get_J(), biases = torch.zeros(nw.get_J().shape[0], device=device, dtype=torch.float64), rng=rng)
            
            noisy_input = data[i].clone() * 0.1 + torch.normal(0, 0.1*noise*data[i].std(), data.shape[1], device=device)
            noisy_input = Langevin(noisy_input)
            
            attractor, steps = relax(nw_relax, input=noisy_input, bias=torch.zeros(nw.get_J().shape[0], device=device, dtype=torch.float64),
                                     inverse_temperature=inverse_temperature, least_action=True, max_steps=200)
            
            attractor_np = attractor.cpu().numpy()
            
            is_new = True
            for at in attractors:
                if np.corrcoef(attractor_np, at)[0,1] > 0.99:
                    is_new = False
            
            if is_new:
                attractors.append(attractor_np)
                if plot:
                    axes[0,i].imshow(noisy_input.cpu().numpy().reshape(int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))), cmap='gray_r', vmin=-1, vmax=1)
                    axes[0,i].set_title(f'Noisy input {i+1}')
                    axes[0,i].set_axis_off()
                    axes[1,i].imshow(attractor_np.reshape(int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))), cmap='gray_r', vmin=-1, vmax=1)
                    axes[1,i].set_title(f'Attractor {len(attractors)}')
                    axes[1,i].set_axis_off()
        if plot:
            plt.show()
        all_attractors.append(np.array(attractors))
        
    all_attractors = np.vstack(all_attractors)
    unique_attractors = []
    for attractor in all_attractors:
        is_new = True
        for at in unique_attractors:
            if np.corrcoef(attractor, at)[0,1] > 0.99:
                is_new = False
        if is_new:
            unique_attractors.append(attractor)
            
    return np.array(unique_attractors)

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def orthogonality(data, attractors, plot=True):

    attractors = np.array(attractors)

    angles_data = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                angles_data[i, j] = angle_between(data[i], data[j])

    angles_attractors = np.zeros((attractors.shape[0], attractors.shape[0]), dtype=float)
    for i in range(attractors.shape[0]):
        for j in range(attractors.shape[0]):
            if i != j:
                angles_attractors[i, j] = angle_between(attractors[i], attractors[j])

    # Plot the angles as a polar histogram
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), subplot_kw=dict(projection='polar'))
        angles_no_self = angles_data[~np.eye(angles_data.shape[0], dtype=bool)]
        axes[0].hist(np.deg2rad(angles_no_self.flatten()), 
                     bins=np.linspace(0, np.pi, 19), color='blue', alpha=0.7, density=True )
        axes[0].set_title('Data')
        axes[0].set_xlim(0, np.pi)  # Only show the upper half of the circle

        angles_no_self = angles_attractors[~np.eye(angles_attractors.shape[0], dtype=bool)]
        axes[1].hist(np.deg2rad(angles_no_self.flatten()), 
                     bins=np.linspace(0, np.pi, 19), color='blue', alpha=0.7)
        axes[1].set_title('Attractors')
        axes[1].set_xlim(0, np.pi)  # Only show the upper half of the circle
        plt.show()

        print('Data: mean', angles_data.mean(), 'std', angles_data.std(), 'median', angles_data.median())
        print('Attractors: mean', angles_attractors.mean(), 'std', angles_attractors.std(), 'median', angles_attractors.median())

    return angles_data, angles_attractors


def performance_metrics(training_output, 
                        train_data, 
                        test_data, 
                        evidence_level,
                        params_retreival, 
                        params_generalization, 
                        inverse_temperature_deterministic):
    """
    Returns some performance metrics silently.
    - median_delta_r2_retrieval: median r^2 improvement in retrieval
    - median_delta_r2_generalization: median r^2 improvement in generalization
    - orthogonality_data: mean root squared deviation from orthogonality
    - orthogonality_attractors: mean root squared deviation from orthogonality
    """

    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data *= evidence_level
    test_data *= evidence_level

    r2_test_original, r2_reconstructed_original = evaluate_reconstruction_accuracy(training_output[0], train_data,
                                                                                      False,
                                                                                      params_retreival["signal_strength"], 
                                                                                      params_retreival["num_trials"], 
                                                                                      params_retreival["SNR"], 
                                                                                      params_retreival["inverse_temperature"], 
                                                                                      params_retreival["num_steps"],
                                                                                      plot=False)
    
    
    median_delta_r2_retrieval = np.median(np.array(r2_reconstructed_original) - np.array(r2_test_original))

    r2_test_original, r2_reconstructed_original = evaluate_reconstruction_accuracy(training_output[0], test_data, 
                                                                                      True,
                                                                                      params_generalization["signal_strength"], 
                                                                                      params_generalization["num_trials"], 
                                                                                      params_generalization["SNR"], 
                                                                                      params_generalization["inverse_temperature"], 
                                                                                      params_generalization["num_steps"],
                                                                                      plot=False)
    median_delta_r2_generalization = np.median(np.array(r2_reconstructed_original) - np.array(r2_test_original))

    attractors = get_deterministic_attractors(training_output[0], 
                                              train_data, 
                                              noise_levels=[0.0], 
                                              inverse_temperature=inverse_temperature_deterministic,
                                              plot=False)
    
    num_attractors = len(np.unique(np.round(attractors, 2), axis=0))  # unique attractors, with tolerance of 0.01

    orthogonality_data, orthogonality_attractors = orthogonality(train_data.cpu().numpy(), attractors, plot=False)

    # remove attractors that are counted multiple times
    orthogonality_attractors = orthogonality_attractors[orthogonality_attractors < 179]
    orthogonality_attractors = orthogonality_attractors[orthogonality_attractors > 1]

    orthogonality_data = np.mean(np.sqrt(np.power(90.0 - orthogonality_data, 2)))
    orthogonality_attractors = np.mean(np.sqrt(np.power(90.0 - orthogonality_attractors, 2)))

    return median_delta_r2_retrieval, median_delta_r2_generalization, num_attractors, orthogonality_data, orthogonality_attractors


def report_network_evaluation(training_output,
                              evidence_level,
                              train_data,
                              test_data,
                              params_retreival,
                              params_generalization,
                              inverse_temperature_deterministic,
                              title="test run"):
    """
    Function to run the network, get attractors, and evaluate performance in a single pipeline.

    Args:
        training_output: a tuple containing the output of the `run_network` function
        evidence_level: the evidence level, scaling the input data
        train_data: training data
        test_data: testing data
        params_retreival: dictionary of parameters for the `evaluate_reconstruction_accuracy` function
        params_generalization: dictionary of parameters for the `evaluate_reconstruction_accuracy` function
        inverse_temperature_deterministic: temperature for the deterministic attractor search
        title: title for the report
    """
    
    # Unpack training output
    nw, weight_change, pattern, accuracy, complexity, vfe = training_output

    # Print title
    print("#" * 50)
    print(f"## {title}")
    print("#" * 50)
    
    # Get deterministic attractors from the trained network
    attractors = get_deterministic_attractors(
        nw, 
        train_data, 
        noise_levels=[0.0], 
        inverse_temperature=inverse_temperature_deterministic, 
        plot=True
    )
    
    # Evaluate orthogonality of attractors
    orthogonality(train_data.cpu().numpy(), attractors, plot=True)

    # Evaluate reconstruction accuracy on training data (retrieval)
    print("\n" + "="*40)
    print("  Evaluating reconstruction accuracy (retrieval)")
    print("="*40)
    evaluate_reconstruction_accuracy(
        nw, 
        data=train_data, 
        **params_retreival
    )

    # Evaluate reconstruction accuracy on test data (generalization)
    print("\n" + "="*40)
    print("  Evaluating reconstruction accuracy (generalization)")
    print("="*40)
    evaluate_reconstruction_accuracy(
        nw, 
        data=test_data, 
        **params_generalization
    )

    # Plot VFE curve
    print("\n" + "="*40)
    print("  VFE curve")
    print("="*40)
    plt.figure(figsize=(10, 3))
    plt.plot(vfe)
    plt.show()

    # Plot network graph
    print("\n" + "="*40)
    print("  Final network connectivity")
    print("="*40)
    nw.plot_network(symmetric=True, node_size=1, edge_width=0.5, plot_bias=False)
    
    print("\n" + "="*40)
    print("  Report finished")
    print("="*40)


            
        
