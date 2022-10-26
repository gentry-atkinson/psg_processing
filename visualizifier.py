import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt


channels = [
    'Thermistor','Respiratory Belt','ECG'
]

if __name__ == '__main__':
    for channel in channels:
        f = np.load(f'{channel}_feature_learner_weights.pt')

