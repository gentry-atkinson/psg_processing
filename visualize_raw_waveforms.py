import numpy as np
from matplotlib import pyplot as plt

chan_dic = {
    'Thermistor Flow': [9],
    'Respiratory Belt' : [10, 11],
    'ECG' : [8],
    'EOG' : [3, 4],
    'EMG' : [5],
    'Leg' : [6, 7]
}

axes = []
fig, axes = plt.subplots(6, 1, sharey=False)