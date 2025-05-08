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

    # visualize
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < train_data.shape[0]:
            image = train_data[i].reshape(8, 8) + np.random.normal(0, 0.1, (8,8)) 

            for j in range(image.shape[0]):
                for k in range(image.shape[1]):
                    image[j, k] = Langevin(image[j, k])

            ax.imshow(image, cmap="coolwarm", interpolation="nearest")
            ax.set_axis_off()
        else:
            ax.set_visible(False)
    plt.show()
    plt.figure(figsize=(3, 3))
    sns.histplot(train_data.flatten())

    return train_data, test_data


def continous_inference_and_learning(nw, 
                                     data, 
                                     inverse_temperature=1, 
                                     learning_rate=0.001, 
                                     num_steps=100):
    """
    Wrapper function to continuous inference and learning

    Runs the networks with the same pattern (input biases) for a given number of iterations.
    Note that training and inference happens simultaneously in this architecture.

    Args:
        nw: AttractorNetwork instance
        data: numpy array of shape (num_nodes,)
        inverse_temperature: controls the temperature during the state activation update
        learning_rate: controls the step size of the weight updates
        num_steps: controls the number of steps in the current epoch (with the same pattern)
    """

    weight_change = [] # how much has the weight matrix changed
    prev_J = nw.get_J()

    for i, node in enumerate(nw.sigmas):
        node.bias = data[i] # put in the data as biases (e.g. external sensory drive or internal computations)

    activations = []
    vfe = []
    accuracy     = []
    complexity = []
    for i in range(num_steps):
        nw.update(inverse_temperature=inverse_temperature, learning_rate=learning_rate, least_action=False)
        activations.append([node.activation for node in nw.sigmas])
        weight_change.append( np.sum(np.power(nw.get_J() - prev_J, 2)))
        prev_J = nw.get_J()
        accuracy.append(nw.accuracy())
        complexity.append(nw.complexity())
        vfe.append(complexity[-1] - accuracy[-1]) # also available as nw.vfe()

    # clean up the network, just in case
    for i, node in enumerate(nw.sigmas):
        node.bias = 0

    return activations, weight_change, accuracy, complexity, vfe


def run_network(data, evidence_level, 
                inverse_temperature, 
                learning_rate,
                num_epochs, 
                num_steps,
                progress_bar=True):
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
    
    data = data.copy()
    data *= evidence_level

    # initialize empty network
    nw = AttractorNetwork(np.zeros((data.shape[1], data.shape[1])), biases=np.zeros(data.shape[1]))

    weight_change = []
    pattern = []
    vfe = []
    accuracy = []
    complexity = []

    for i in tqdm(range(num_epochs), disable=not progress_bar):
        # select a pattern randomly
        di = np.random.randint(0, data.shape[0])
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


def evaluate_reconstruction_accuracy(nw, 
                                     data, 
                                     sample,
                                     signal_strength,
                                     num_trials, 
                                     SNR, 
                                     inverse_temperature, 
                                     num_steps, 
                                     plot=True):
    """
    Helper function to evaluate reconstruction accuracy
    """
    r2_test_original = []
    r2_reconstructed_original = []

    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(10, 3))
    
    for i in tqdm(range(num_trials), disable=not plot):
        if sample:
            idx = np.random.randint(0, data.shape[0])
        else:
            idx = i % data.shape[0]
            
        original_pattern = data[idx] * signal_strength
        test_pattern = original_pattern + np.random.normal(0, original_pattern.std()/SNR, data[0].shape)
        acts, _, _, _, _ = continous_inference_and_learning(nw, data=test_pattern, 
                                                            inverse_temperature=inverse_temperature, 
                                                            learning_rate=0.0, 
                                                            num_steps=num_steps)
        mean_activity = np.mean(acts, axis=0)
        r2_test_original.append(np.round(np.corrcoef(test_pattern, original_pattern)[0, 1]**2, 3))
        r2_reconstructed_original.append(np.round(np.corrcoef(mean_activity, original_pattern)[0, 1]**2, 3))

        if plot and i < 10:
            axes[0, i].imshow(test_pattern.reshape(8, 8), cmap="gray")
            axes[0, i].set_axis_off()
            axes[1, i].imshow(mean_activity.reshape(8, 8), cmap="gray")
            axes[1, i].set_axis_off()
            axes[2, i].imshow(original_pattern.reshape(8, 8), cmap="gray")
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
        sns.lineplot(np.array(acts), legend=False, linestyle='-', alpha=0.5, linewidth = 1, palette='Spectral')
        plt.show()

    return r2_test_original, r2_reconstructed_original

def vfe(nw, acts):
    """
    Small helper function to compute the VFE.
    """
    for i, node in enumerate(nw.sigmas):
        node.activation = float(acts[i])
    #return np.round( nw.expected_energy() - nw.entropy(), 8 )
    return np.round( nw.vfe(), 4 )

def get_deterministic_attractors(nw, data, 
                                 noise_levels=(0.0, 0.1, 0.2, 0.5),
                                 inverse_temperature=1,
                                 plot=True):

    for noise in noise_levels:
        attractors = []
        if plot:
            fig, axes = plt.subplots(2, data.shape[0], figsize=(20, 4))
            print(f"  ** Noise: {noise}")
        for i in tqdm(range(data.shape[0]), disable=not plot):
            rng=np.random.default_rng(None)
            nw_relax = AttractorNetwork(nw.get_J(), biases = np.zeros(nw.get_J().shape[0]), rng=rng)
            noisy_input = np.copy(data[i]) * 0.1 + rng.normal(0, 0.1*noise*data[i].std(), data.shape[1]) 
            noisy_input = np.array([Langevin(x) for x in noisy_input])
            attractor, steps = relax(nw_relax, input=noisy_input, bias=np.zeros(nw.get_J().shape[0]),
                                inverse_temperature=inverse_temperature, least_action=True, max_steps=1000, tol=1e-12)
            
            #print("* noisy_Q", vfe(nw, noisy_Q))
            vfe_noisy = vfe(nw, noisy_input)
            #print("* attractor", vfe(nw, attractor))
            vfe_result = vfe(nw, attractor)

            #print(steps)
            attractors.append(attractor)
            # plot
            if plot:
                axes[0, i].imshow(noisy_input.reshape(8, 8), cmap='gray_r')
                axes[0, i].set_title(f'{vfe_noisy}, ({steps})')
                axes[0, i].set_xticks([])
                axes[0, i].set_yticks([])
                axes[0, i].set_axis_off()   

                axes[1, i].imshow(attractor.reshape(8, 8), cmap='gray_r')
                axes[1, i].set_title(f'{vfe_result}, ({steps})')
                axes[1, i].set_xticks([])
                axes[1, i].set_yticks([])
                axes[1, i].set_axis_off()
        if plot:
            plt.show()

    return attractors

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

    orthogonality_data, orthogonality_attractors = orthogonality(train_data, attractors, plot=False)

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
    Report full network evaluation

    A long function that creates a full report on the network's performance.
    - plots the network weights
    - plots vfe and co.
    - evaluates retrieval accuracy
    - evaluates 1-shot generalizability
    - reconstructs attractors corresponding to the input data
    - evaluates orthogonality of attractors
    """

    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data *= evidence_level
    test_data *= evidence_level

    print("* Visualize training data")
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < train_data.shape[0]:
            image = train_data[i].reshape(8, 8) + np.random.normal(0, 0.1, (8,8)) 

            for j in range(image.shape[0]):
                for k in range(image.shape[1]):
                    image[j, k] = Langevin(image[j, k])

            ax.imshow(image, cmap="coolwarm", interpolation="nearest")
            ax.set_axis_off()
        else:
            ax.set_visible(False)
    plt.show()
    plt.figure(figsize=(3, 3))
    sns.histplot(train_data.flatten())
    
    nw = training_output[0]
    weight_change = training_output[1]
    pattern = training_output[2]
    accuracy = training_output[3]
    complexity = training_output[4]
    vfe = training_output[5]

    #print("* Plot network")
    #plt.figure(figsize=(5, 5))
    #training_output[0].plot_network(plot_bias=False)
    #plt.show()
    print("* Network weights",  f"({title})")
    plt.figure(figsize=(5, 5))
    sns.heatmap(nw.get_J(), cmap="coolwarm", center=0, square=True)
    plt.show()
    print("* Weight change", f"({title})")
    plt.figure(figsize=(8, 2))
    sns.lineplot(np.array(weight_change))
    num_steps = len(weight_change) // len(pattern)
    for i, pattern_label in enumerate(pattern):
        plt.text(i*num_steps, max(weight_change[i*num_steps:i*num_steps+num_steps]), str(pattern_label), ha='center', va='bottom')
    plt.show()
    print("* Accuracy", f"({title})")
    plt.figure(figsize=(4, 2))
    sns.lineplot(np.array(accuracy), legend=False, linestyle='-', alpha=1, linewidth = 1)
    plt.show()
    print("* Complexity", f"({title})")
    plt.figure(figsize=(4, 2))
    sns.lineplot(np.array(complexity), legend=False, linestyle='-', alpha=1, linewidth = 1)
    plt.show()
    print("* VFE", f"({title})")
    plt.figure(figsize=(4, 2))
    sns.lineplot(vfe, legend=False, linestyle='-', alpha=0.2, linewidth = 1, color='blue')
    # Calculate and plot smoothed VFE using a rolling window
    # Assumes pandas (as pd) is imported and num_steps is defined
    vfe_series = pd.Series(vfe)
    window_size = 200
    smoothed_vfe = vfe_series.rolling(window=window_size, center=True, min_periods=1).mean()
    sns.lineplot(smoothed_vfe, legend=False, linestyle='-', alpha=1, linewidth=1.5, color='black')
    plt.show()

    print("* Retrieval accuracy", f"({title})")
    evaluate_reconstruction_accuracy(nw, data=train_data, 
                                     sample=False,
                                     signal_strength=params_retreival["signal_strength"],
                                     num_trials=params_retreival["num_trials"],
                                     SNR=params_retreival["SNR"], 
                                     inverse_temperature=params_retreival["inverse_temperature"], 
                                     num_steps=params_retreival["num_steps"])
    
    print("* Generalization accuracy", f"({title})")
    evaluate_reconstruction_accuracy(nw, data=test_data, 
                                     sample=True,
                                     signal_strength=params_generalization["signal_strength"],
                                     num_trials=params_generalization["num_trials"],
                                     SNR=params_generalization["SNR"], 
                                     inverse_temperature=params_generalization["inverse_temperature"], 
                                     num_steps=params_generalization["num_steps"])
    
    print("* Attractors")
    attractors = get_deterministic_attractors(nw, train_data, noise_levels=(0.0, 0.5), inverse_temperature=inverse_temperature_deterministic)

    print("* Orthogonality")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    r_data = np.corrcoef(train_data)
    print('Data correlation: mean', r_data.mean(), 'std', r_data.std(), 'median', r_data.median())
    sns.histplot(r_data[np.triu_indices_from(r_data, k=1)], ax=axes[0])
    r_attractors = np.corrcoef(attractors)
    print('Attractors correlation: mean', r_attractors.mean(), 'std', r_attractors.std(), 'median', r_attractors.median())
    sns.histplot(r_attractors[np.triu_indices_from(r_attractors, k=1)], ax=axes[1])
    plt.show()

    orthogonality(train_data, np.array(attractors))

        


            
        
