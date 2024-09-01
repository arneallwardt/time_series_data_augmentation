from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform 
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

original_color = os.getenv('GRAY', 'red')
autoencoder_color = os.getenv('BLUE', 'blue')
timegan_color = os.getenv('GREEN', 'green')
random_transformation_color = os.getenv('RED', 'red')

plot_colors = {
    'Jittering': random_transformation_color,
    'Time Warping': random_transformation_color,
    'AE': autoencoder_color,
    'VAE': autoencoder_color,
    'TimeGAN LSTM': timegan_color,
    'TimeGAN GRU': timegan_color,
}
color_alpha = 0.4

### EVALUATION METRICS ###

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


def get_pca_results(real, syn):
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

    print(real.shape, syn.shape)

    pca_real_results = pca.transform(real)
    pca_syn_results = pca.transform(syn)

    return pca_real_results, pca_syn_results


def plot_pca_results(pca_real_results, eval_datasets):
    
    num_results = len(eval_datasets)
    f, axarr = plt.subplots(1, num_results, figsize=(5*num_results, 5))

    # In case only one subplot is needed, put axarr in a list, so that it can be iterated over
    if num_results == 1:
        axarr = [axarr]

    for i in range(num_results):

        axarr[i].scatter(pca_real_results[:,0], pca_real_results[:,1], c=original_color, alpha=color_alpha, label = "Original")
        axarr[i].scatter(eval_datasets[i].pca_results[:,0], eval_datasets[i].pca_results[:,1], c=plot_colors[eval_datasets[i].type], alpha=color_alpha, label = "Synthetisch")
        axarr[i].legend()  
        axarr[i].set_title(f'{eval_datasets[i].type} PCA')
        axarr[i].set_xlabel('x-pca')
        axarr[i].set_ylabel('y_pca')

    plt.tight_layout()
    plt.show()


def get_tsne_results(real, syn, no_samples):
    '''
    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique for embedding high-dimensional data for visualization in a low-dimensional space

    Args:
        - real: np.array of size (n_samples, seq_len*features), flattened original data
        - syn: np.array of size (n_samples, seq_len*features), flattened synthetic data
        - no_samples: int, number of samples for each synthetic and real data to be used for visualization
    '''

    # Do t-SNE Analysis together       
    data_final = np.concatenate((real, syn), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300, random_state=69)
    tsne_results = tsne.fit_transform(data_final) # shape: (2*no_samples, 2)

    return tsne_results


def plot_tsne_results(eval_datasets, no_samples):


    num_results = len(eval_datasets)
    f, axarr = plt.subplots(1, num_results, figsize=(5*num_results, 5))

    # In case only one subplot is needed, put axarr in a list, so that it can be iterated over
    if num_results == 1:
        axarr = [axarr]

    for i in range(num_results):

        colors = [original_color for _ in range(no_samples)] + [plot_colors[eval_datasets[i].type] for _ in range(no_samples)]  
        
        axarr[i].scatter(eval_datasets[i].tsne_results[:no_samples,0], eval_datasets[i].tsne_results[:no_samples,1], 
                        c=colors[:no_samples], alpha=color_alpha, label="Original")
        axarr[i].scatter(eval_datasets[i].tsne_results[no_samples:,0], eval_datasets[i].tsne_results[no_samples:,1], 
                        c=colors[no_samples:], alpha=color_alpha, label="Synthetisch")
        
        axarr[i].legend()
        axarr[i].set_title(f'{eval_datasets[i].type} t-SNE')
        axarr[i].set_xlabel('x-tsne')
        axarr[i].set_ylabel('y-tsne')

    plt.tight_layout()
    plt.show()



### DATA PREPROCESSING ###

def get_sampling_indices(data_real, data_syn, no_samples):
    sampling_indices = np.random.permutation(min([len(data_real), len(data_syn)]))[:no_samples] # get random indices for random samples (only as much as no_samples)
    return sampling_indices

def get_preprocessed_data(data, mean_flatten, no_samples, verbose):
    print(f'shape of data before preprocessing: {data.shape}')
    _, seq_len, _ = data.shape

    # flatten data to be in 2 dims
    if mean_flatten: # whether or not to use the mean of the features or original values of the features
        for i in range(no_samples):
            if (i == 0):
                prep_data = np.reshape(np.mean(data[0], 1), [1,seq_len]) # get mean over features and save as array with shape (1, 24)
            else:
                prep_data = np.concatenate((prep_data, np.reshape(np.mean(data[i],1), [1,seq_len]))) # same as above, but append to existing array
    else:
        prep_data = data.reshape(no_samples, -1)
    
    print(f'shape of data after preprocessing: {prep_data.shape}') if verbose else None
    return prep_data



### EVALUATION WORKFLOW

def visual_evaluation(data_real, eval_datasets, no_samples=1000, mean_flatten=False, verbose=False):
    '''
    Visualizes the original and synthetic data using PCA or tSNE

    Args:
        - data_real: np.array of size (n_samples, seq_len, n_features), original data
        - data_syn: np.array of size (n_samples, seq_len, n_features), synthetic data
        - mean_flatten: bool, whether or not to use the mean of the features or original values of the features for evaluating
    '''

    print(f'Number of datasets to evaluate: {len(eval_datasets)}') if verbose else None

    # get random sampling indices
    sampling_indices = get_sampling_indices(data_real=data_real, 
                                            data_syn=eval_datasets[0].syn_data, 
                                            no_samples=no_samples)
    
    # get permuted real data and preprocess
    data_real = data_real[sampling_indices]
    prep_data_real = get_preprocessed_data(data_real, mean_flatten, no_samples, verbose)

    ### Apply metrics
    for eval_dataset in eval_datasets:        
        
        pca_real_results, eval_dataset.pca_results = get_pca_results(real=prep_data_real,
                                        syn=get_preprocessed_data(eval_dataset.syn_data[sampling_indices], mean_flatten, no_samples, verbose))
        
        eval_dataset.tsne_results = get_tsne_results(real=prep_data_real,
                                            syn=get_preprocessed_data(eval_dataset.syn_data[sampling_indices], mean_flatten, no_samples, verbose),
                                            no_samples=no_samples)

    ### Plot metrics
    plot_pca_results(pca_real_results, eval_datasets)
    plot_tsne_results(eval_datasets, no_samples)
    

   

    