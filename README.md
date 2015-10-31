# image-classification
Texture image classification with Python and MATLAB 

Image source: http://www.nada.kth.se/cvap/databases/kth-tips/download.html (use link greyscale PNG Images - 23MB)

Textures Images used: Aluminium Foil, Corduroy, and Orange Peel. 

Train set: 120 images (40 images from each class)

Test set: 120 images (40 images from each class)

Features (extracted using Matlab):
1. Gray-level co-occurrence matrix (GLCM): Energy and Entropy
2. Fast Fourier Transform (FFT): Mean and Variances 

Classification method: 
1. K-nearest neighbor
2. Gaussian Naïve-bayes

Evaluation: classification accuracy

# The Recipes

Extracting Features with Matlab

1. Download texture image dataset 
2. Collect in one folder, rename images3.
3. Run .m file 
4. Save dataku.mat file 

Number of features: 4: 
1. attribute 1: Entropy of GLCM 
2. attribute 2: Energy of GLCM
3. attribute 3: Mean of FFT
4. attribute 4: Variance of FFT 

Classifying the images 

1. KNN

```python
python imageclassification3_knn.py 
```
Note: You can change line 14 to switch the features of GLCM, FFT, or all features.

2. GNB

```python
python imageclassification4_knn.py 
```
Note: You can change line 14 to switch the features of GLCM, FFT, or all features.  


# Result

1. KNN with GLCM, FFT
![alt text](https://raw.githubusercontent.com/arikunco/fmatch-homography/master/result_samsung.jpg "Test Smartphone 1")

2. GNB with GLCM, FFT
![alt text](https://raw.githubusercontent.com/arikunco/fmatch-homography/master/result_samsung2.jpg "Test Smartphone 1")

3. KNN with GLCM + FFT and GNB with GLCM + FFT
![alt text](https://raw.githubusercontent.com/arikunco/fmatch-homography/master/result_tnt.jpg "Test Trinity Book")

