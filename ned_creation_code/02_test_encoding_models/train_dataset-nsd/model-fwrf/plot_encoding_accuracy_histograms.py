"""Histograms of noise-ceiling-normalized encoding accuracy scores.

Parameters
----------
sub : int
	Used NSD subject.
all_rois : list of str
	List with all modeled ROIs.
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED

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
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8]
parser.add_argument('--all_rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'EBA', 'FBA-2', 'OFA', 'FFA-1', 'FFA-2', 'PPA', 'RSC', 'OPA',
	'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'early', 'midventral',
	'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral'])
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding model stats
# =============================================================================
results_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-fmri', 'train_dataset-nsd', 'model-fwrf',
	'encoding_accuracy.npy')

results = np.load(results_dir, allow_pickle=True).item()

acc = results['noise_normalized_encoding']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 15
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 2
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
# Plot the encoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

for r, roi in enumerate(args.all_rois):

	# Plot the results
	axs[r].hist(acc['s'+str(args.sub)+'_'+roi]*100, bins=100, density=True,
		color=colors[0])

	# y-axis

	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Voxels\nprobability density', fontsize=fontsize)
	axs[r].set_ylim(bottom=0, top=.1)

	# x-axis
	ticks = np.arange(0, 110, 20)
	labels = [0, 20, 40, 60, 80, 100]
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Explained variance (%)', fontsize=fontsize)
		plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize)
	axs[r].set_xlim(left=0, right=100)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

# y-axis
axs[23].set_xlabel('Explained variance (%)', fontsize=fontsize)

#fig.savefig('encoding_accuracy_histogram_sub-01.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-02.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-03.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-04.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-05.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-06.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-07.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-08.png', dpi=600, bbox_inches='tight')

#fig.savefig('encoding_accuracy_histogram_sub-01.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-02.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-03.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-04.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-05.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-06.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-07.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_histogram_sub-08.svg', bbox_inches='tight')

