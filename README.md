Certainly! Let's further enhance the `README.md` file to include the introduction, project details, and a summary of findings with suggestions for improvement.

---

# Low Light Image Denoising Project

## Introduction

### Architecture Used

The low light image denoising project employs a combination of clustering algorithms, Gaussian mixture models (GMM), singular value decomposition (SVD) for dimensionality reduction, and optimization techniques. These methods are applied sequentially to enhance image quality by reducing noise and preserving essential image features.

### Specifications

- **Programming Languages and Libraries**: Python, NumPy, Pandas, OpenCV, scikit-image (skimage), scikit-learn, Matplotlib, PIL (Pillow)
- **Algorithms**: K-Means Clustering, Gaussian Mixture Models (GMM), Singular Value Decomposition (SVD), Support Vector Machines (SVM)
- **Image Format**: RGB images
- **Data Source**: Local image files

### Paper Implemented

The project is inspired by the following paper:

- **Link**: (https://arxiv.org/pdf/2112.14022}
            (https://arxiv.org/abs/2207.10564)
            (https://arxiv.org/pdf/2404.14248)
            (https://arxiv.org/abs/2310.17577)
            )

## Project Details

### Workflow Explanation

#### 1. Image Processing and Clustering

##### Code Snippet 1: Image Reading and Reshaping

```python
import numpy as np
from skimage import io

# Read the image
img = io.imread(path)

# Reshape the image data for clustering
img_new = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
```

- **Functionality**: Reads an image and reshapes it into a format suitable for clustering algorithms.
- **Importance**: Prepares the image data to apply clustering techniques for segmentation and denoising.

##### Code Snippet 2: K-Means Clustering

```python
from sklearn.cluster import KMeans

# Apply K-Means clustering with different cluster counts
kmeans = KMeans(n_clusters=3, max_iter=100000, random_state=5)
kmeans2 = KMeans(n_clusters=5, max_iter=100000, random_state=5)
kmeans3 = KMeans(n_clusters=10, max_iter=100000, random_state=5)

kmeans.fit(img_new)
kmeans2.fit(img_new)
kmeans3.fit(img_new)

labels = kmeans.predict(img_new)
labels2 = kmeans2.predict(img_new)
labels3 = kmeans3.predict(img_new)

Ccentres = kmeans.cluster_centers_.astype(int)
Ccentres2 = kmeans2.cluster_centers_.astype(int)
Ccentres3 = kmeans3.cluster_centers_.astype(int)
```

- **Functionality**: Performs K-Means clustering on image data with different numbers of clusters.
- **Importance**: Segments the image into clusters based on color similarity, aiding in denoising and image enhancement.

##### Code Snippet 3: Visualization of K-Means Results

```python
import matplotlib.pyplot as plt

# Construct new images from clustered data
newimage = np.array([Ccentres[label] for label in labels])
newimagesee = newimage.reshape(img.shape[0], img.shape[1], img.shape[2])

plt.imshow(newimagesee)
plt.title("K-Means Clustering (3 clusters)")
plt.show()

# Repeat for kmeans2 and kmeans3
```

- **Functionality**: Visualizes the images reconstructed from K-Means clustering results.
- **Importance**: Helps in understanding the segmentation achieved by different cluster counts and their impact on image quality.

#### 2. Gaussian Mixture Models (GMM)

##### Code Snippet 4: GMM Clustering

```python
from sklearn.mixture import GaussianMixture as GMM

# Fit GMM with different number of components
gmm = GMM(n_components=3)
gmm.fit(img_new)
glabels = gmm.predict(img_new)

# Repeat for gmm2 and gmm3
```

- **Functionality**: Applies Gaussian Mixture Models to cluster image data.
- **Importance**: Captures more complex patterns and distributions in the image, potentially offering more refined segmentation than K-Means.

##### Code Snippet 5: Visualization of GMM Results

```python
# Construct new images from GMM clustered data
gnewimage = np.array([gCcentres[glabels[i]] for i in range(len(glabels))])
gnewimagesee = gnewimage.reshape(img.shape[0], img.shape[1], img.shape[2])

plt.imshow(gnewimagesee)
plt.title("Gaussian Mixture Model (3 components)")
plt.show()

# Repeat for gmm2 and gmm3
```

- **Functionality**: Visualizes the images reconstructed from GMM clustering results.
- **Importance**: Offers insight into how GMM captures different modes in the image data compared to K-Means.

#### 3. Dimensionality Reduction and Optimization

##### Code Snippet 6: Singular Value Decomposition (SVD)

```python
# Perform Singular Value Decomposition (SVD)
U, s, V = np.linalg.svd(img)

# Reconstruct images with different numbers of components
num_components = 5
reconst_img_5 = np.dot(U[:, :num_components], np.dot(np.diag(s[:num_components]), V[:num_components, :]))
reconst_img_5 = np.round(reconst_img_5).astype(int)

plt.imshow(reconst_img_5)
plt.title("Image Reconstruction using SVD (5 components)")
plt.show()

# Repeat for num_components = 10 and 100
```

- **Functionality**: Uses SVD for dimensionality reduction and image reconstruction.
- **Importance**: Reduces noise and compresses image information while retaining important features, aiding in denoising.

#### 4. Miscellaneous

##### Code Snippet 7: Support Vector Machine (SVM) Optimization

```python
from sklearn.svm import SVC

# Example of SVM usage (seems misplaced, consider reordering)
modelSVM = SVC(kernel='linear', C=0.01)
modelSVM.fit(X, Y)

print(f"Intercept (b): {modelSVM.intercept_}")
print(f"Coefficients (w1, w2): {modelSVM.coef_[0]}")
```

- **Functionality**: Demonstrates SVM usage, though contextually unrelated to image processing.
- **Importance**: Provides an example of applying supervised learning for classification tasks.

#### 5. Data Visualization and Analysis (Additional)

##### Code Snippet 8: Data Visualization with PCA and LDA

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Example of PCA and LDA usage (seems misplaced, consider reordering)
pca = PCA(n_components=2)
results = pca.fit_transform(iris.data)

plt.scatter(results[:, 0], results[:, 1], c=iris.target, cmap='viridis')
plt.title("PCA Visualization of Iris Dataset")
plt.show()

dimensiondown = LDA(n_components=2)
ldatransformeddata = dimensiondown.fit_transform(iris.data, iris.target)

plt.scatter(ldatransformeddata[:, 0], ldatransformeddata[:, 1], c=iris.target, cmap='viridis')
plt.title("LDA Visualization of Iris Dataset")
plt.show()
```

- **Functionality**: Visualizes high-dimensional data (like Iris dataset) after reducing dimensions using PCA and LDA.
- **Importance**: Demonstrates techniques for visualizing data distribution and separability, which can aid in understanding image feature representations as well.

## Summary and Future Improvements

### Summary of Findings

The project successfully applies various image processing techniques to denoise low light images:

- **Clustering**: K-Means and GMM effectively segment images based on color similarity.
- **Dimensionality Reduction**: SVD reduces noise while preserving image features.
- **Visualization**: PCA and LDA provide insights into data distribution and feature separability.

### Methods for Further Improvement

To enhance the project further, consider the following:

- **Advanced Denoising Algorithms**: Explore deep learning-based approaches such as convolutional neural networks (CNNs) for better denoising results.
- **Hyperparameter Tuning**: Optimize parameters in clustering algorithms and dimensionality reduction techniques for improved performance.
- **Enhanced Evaluation Metrics**: Incorporate more comprehensive metrics beyond PSNR to evaluate denoising quality.

---
