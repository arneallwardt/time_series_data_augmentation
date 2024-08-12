from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform 
import numpy as np
import matplotlib.pyplot as plt

### SINGLE EVALUATION METRICS

def maximum_mean_discrepancy(X, Y):
    '''
    maximum mean discrepancy (mmd) is a metric for comparing the distribution of 2 datasets
    '''

    def _gaussian_kernel(x, y, sigma=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    
    XX = squareform(pdist(X, metric=lambda x, y: _gaussian_kernel(x, y)))
    YY = squareform(pdist(Y, metric=lambda x, y: _gaussian_kernel(x, y)))
    XY = np.array([[_gaussian_kernel(x, y) for y in Y] for x in X])
    return XX.mean() + YY.mean() - 2 * XY.mean()


def pca(real, syn):
    '''
    Principal Component Analysis (PCA) is a dimensionality reduction technique that can be used to reduce a large number of variables to a smaller number of variables while preserving as much variance as possible.
    It is used to visualize the data in 2D and compare the distributions of the original and synthetic data.

    Args:
        - real: np.array of size (n_samples, seq_len*features), flattened original data
        - syn: np.array of size (n_samples, seq_len*features), flattened synthetic data
    '''

    # Fit PCA
    pca = PCA(n_components = 2)
    pca.fit(real)
    pca_real_results = pca.transform(real)
    pca_syn_results = pca.transform(syn)

    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_real_results[:,0], pca_real_results[:,1], c='red', alpha = 0.2, label = "Original")
    plt.scatter(pca_syn_results[:,0], pca_syn_results[:,1], c='blue', alpha = 0.2, label = "Synthetic")

    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()


def tsne(real, syn, no_samples):
    '''
    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique for embedding high-dimensional data for visualization in a low-dimensional space

    Args:
        - real: np.array of size (n_samples, seq_len*features), flattened original data
        - syn: np.array of size (n_samples, seq_len*features), flattened synthetic data
        - no_samples: int, number of samples for each synthetic and real data to be used for visualization
    '''

    colors = ["red" for i in range(no_samples)] + ["blue" for i in range(no_samples)]  

    # Do t-SNE Analysis together       
    data_final = np.concatenate((real, syn), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(data_final) # shape: (2*no_samples, 2)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:no_samples,0], tsne_results[:no_samples,1], 
                c = colors[:no_samples], alpha = 0.2, label = "Original") # original data can be accessed by slicing the first no_samples
    plt.scatter(tsne_results[no_samples:,0], tsne_results[no_samples:,1], 
                c = colors[no_samples:], alpha = 0.2, label = "Synthetic") # synthetic data can be accessed by slicing the last no_samples
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()  

def get_sampling_indices(data_real, data_syn, no_samples):
    pass

### EVALUATION WORKFLOW

def visual_evaluation(data_real, data_syn, mean_flatten=False, verbose=False):
    '''
    Visualizes the original and synthetic data using PCA or tSNE

    Args:
        - data_real: np.array of size (n_samples, seq_len, n_features), original data
        - data_syn: np.array of size (n_samples, seq_len, n_features), synthetic data
        - mean_flatten: bool, whether or not to use the mean of the features or original values of the features for evaluating
    '''

    # check if data syn is array
    multi_evaluation = isinstance(data_syn, list)
    print(f'Evaluating multiple datasets?: {multi_evaluation}') if verbose else None

    ### Preprocessing 
    data_real = np.asarray(data_real)
    data_syn = np.asarray(data_syn)  

    # get random samples 
    no_samples = min([1000, len(data_real), len(data_syn)]) # number of samples
    sampling_indices = np.random.permutation(min([len(data_real), len(data_syn)]))[:no_samples] # get random indices for random samples (only as much as no_samples)

    data_real = data_real[sampling_indices]
    data_syn = data_syn[sampling_indices]

    # get shape of original data
    _, seq_len, _ = data_real.shape

    # flatten data to be in 2 dims
    if mean_flatten: # whether or not to use the mean of the features or original values of the features
        for i in range(no_samples):
            if (i == 0):
                prep_data_real = np.reshape(np.mean(data_real[0], 1), [1,seq_len]) # get mean over features and save as array with shape (1, 24)
                prep_data_syn = np.reshape(np.mean(data_syn[0], 1), [1,seq_len])
            else:
                prep_data_real = np.concatenate((prep_data_real, 
                                            np.reshape(np.mean(data_real[i],1), [1,seq_len]))) # same as above, but append to existing array
                prep_data_syn = np.concatenate((prep_data_syn, 
                                            np.reshape(np.mean(data_syn[i],1), [1,seq_len])))
    else:
        prep_data_real = data_real.reshape(no_samples, -1)
        prep_data_syn = data_syn.reshape(no_samples, -1)
    
    
    print(f'Data has been preprocessed. Shape: {prep_data_real.shape}') if verbose else None

    ### Apply metrics

    