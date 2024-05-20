"""Plot the voxel number of the 23 NSD visual ROIs.

Parameters
----------
all_subs : list of int
	List with all subject numbers.
all_rois : list of str
	List with all modeled ROIs.
ned_dir : str
	Neural encoding dataset directory.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--all_subs', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--all_rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'EBA', 'FBA-2', 'OFA', 'FFA-1', 'FFA-2', 'PPA', 'RSC', 'OPA',
	'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'early', 'midventral',
	'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral'])
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding model stats
# =============================================================================
results_dir = os.path.join(args.ned_dir, 'results',
	'encoding_models_prediction_accuracy', 'modality-fmri',
	'train_dataset-nsd', 'model-fwrf', 'prediction_accuracy.npy')

results = np.load(results_dir, allow_pickle=True).item()


# =============================================================================
# Get the voxel number
# =============================================================================
voxel_num = np.zeros((len(args.all_subs), len(args.all_rois)))

for r, roi in enumerate(args.all_rois):

	for s, sub in enumerate(args.all_subs):
		voxel_num[s,r] = len(results['noise_normalized_encoding']\
			['s'+str(sub)+'_'+roi])


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 15
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
colors = [(170/255, 118/255, 186/255)]


# =============================================================================
# Plot the voxel number
# =============================================================================
fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
x = np.arange(len(voxel_num))
width = 0.4

for r, roi in enumerate(args.all_rois):

	# Plot the encoding accuracies
	axs[r].bar(x, voxel_num[:,r], width=width, color=colors[0])

	# Plot the encoding accuracies subject-mean
	y = np.mean(voxel_num[:,r], 0)
	axs[r].plot([min(x), max(x)], [y, y], '--', color='k', linewidth=2,
		alpha=0.4, label='Subjects mean')

	# y-axis
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Voxel number',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)

	# x-axis
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)
		xticks = x
		xlabels = ['1', '2', '3', '4', '5', '6', '7', '8']
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=fontsize)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

# y-axis
axs[23].set_xlabel('Subjects', fontsize=fontsize)

#plt.savefig('encoding_models_prediction_accuracy_all_rois_bar', dpi=600)
