import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Parameters
D = 3  # Number of binary variables
K = 2  # Number of mixture components
N = 5000  # Increased data size for stability
np.random.seed(42)  # For reproducibility

# True parameters
true_pi = np.array([0.6, 0.4])
true_mu = np.array([[0.2, 0.5, 0.8], [0.7, 0.3, 0.4]])


# Generate data
def generate_data(N, true_pi, true_mu):
    z = np.random.choice(K, size=N, p=true_pi)
    X = np.zeros((N, D))
    for n in range(N):
        X[n] = bernoulli.rvs(true_mu[z[n]])
    return X, z


X, true_labels = generate_data(N, true_pi, true_mu)


# Improved EM Algorithm
def em_bernoulli_mixture(X, K, max_iter=200, tol=1e-8):
    N, D = X.shape

    # pi = np.ones(K) / K  # Initialize mixing coefficients
    # mu = np.random.rand(K, D)  # Initialize Bernoulli parameters
    # responsibilities = np.zeros((N, K))  # Responsibilities (posterior probabilities)

    # Better initialization using k-means
    kmeans = KMeans(n_clusters=K, n_init=10).fit(X)
    pi = np.array([np.mean(kmeans.labels_ == k) for k in range(K)])
    mu = np.array([X[kmeans.labels_ == k].mean(axis=0) for k in range(K)])
    responsibilities = np.zeros((N, K))
    log_likelihood_history = []

    for iteration in range(max_iter):
        # E-step with log-sum-exp for numerical stability
        log_resp = np.zeros((N, K))
        for k in range(K):
            log_resp[:, k] = np.log(pi[k] + 1e-16) + np.sum(
                X * np.log(mu[k] + 1e-16) + (1 - X) * np.log(1 - mu[k] + 1e-16), axis=1
            )
        log_resp -= np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        responsibilities = resp

        # M-step with Laplace smoothing to avoid zero probabilities
        Nk = responsibilities.sum(axis=0) + 1e-16
        pi = Nk / N
        for k in range(K):
            mu[k] = (
                np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) + 1e-6
            ) / (
                Nk[k] + 2e-6
            )  # Regularization

        # Compute log-likelihood for convergence check
        log_likelihood = np.sum(np.log(np.sum(resp, axis=1)))
        log_likelihood_history.append(log_likelihood)
        if iteration > 0 and abs(log_likelihood - log_likelihood_history[-2]) < tol:
            break

    return pi, mu, responsibilities


# Run EM algorithm
estimated_pi, estimated_mu, responsibilities = em_bernoulli_mixture(X, K)

# Evaluation
print("True Mixing Coefficients (pi):", true_pi)
print("Estimated Mixing Coefficients (pi):", np.round(estimated_pi, 3))
print("\nTrue Bernoulli Parameters (mu):\n", true_mu)
print("Estimated Bernoulli Parameters (mu):\n", np.round(estimated_mu, 3))
# |%%--%%| <P1k7upFr42|kd6rwkW4ZK>

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assign each data point to the component with the highest responsibility
estimated_labels = np.argmax(responsibilities, axis=1)


# Plotting
def plot_bernoulli_mixture(X, true_labels, estimated_labels, true_mu, estimated_mu):
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot true labels
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for k in range(K):
        plt.scatter(
            X_pca[true_labels == k, 0],
            X_pca[true_labels == k, 1],
            label=f"True Component {k+1}",
        )
    plt.title("True Labels")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()

    # Plot estimated labels
    plt.subplot(1, 2, 2)
    for k in range(K):
        plt.scatter(
            X_pca[estimated_labels == k, 0],
            X_pca[estimated_labels == k, 1],
            label=f"Estimated Component {k+1}",
        )
    plt.title("Estimated Labels")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot Bernoulli parameters
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(D), true_mu[0], alpha=0.5, label="True Component 1")
    plt.bar(np.arange(D), true_mu[1], alpha=0.5, label="True Component 2")
    plt.title("True Bernoulli Parameters")
    plt.xlabel("Dimension")
    plt.ylabel("Probability")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(D), estimated_mu[0], alpha=0.5, label="Estimated Component 1")
    plt.bar(np.arange(D), estimated_mu[1], alpha=0.5, label="Estimated Component 2")
    plt.title("Estimated Bernoulli Parameters")
    plt.xlabel("Dimension")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot results
plot_bernoulli_mixture(X, true_labels, estimated_labels, true_mu, estimated_mu)
