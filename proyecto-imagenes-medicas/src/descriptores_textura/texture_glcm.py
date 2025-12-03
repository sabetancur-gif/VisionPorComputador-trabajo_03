# Local Binary Patterns (LBP)

import numpy as np
from skimage.feature import graycomatrix, graycoprops

def compute_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    
    glcm = graycomatrix(image, 
                        distances=distances, 
                        angles=angles, 
                        symmetric=True, 
                        normed=True)

    features = {
        "contrast": graycoprops(glcm, 'contrast').mean(),
        "correlation": graycoprops(glcm, 'correlation').mean(),
        "energy": graycoprops(glcm, 'energy').mean(),
        "homogeneity": graycoprops(glcm, 'homogeneity').mean()
    }

    return features
