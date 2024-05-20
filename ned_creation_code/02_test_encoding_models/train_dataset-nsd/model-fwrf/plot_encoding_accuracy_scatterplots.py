"""Scatterplots of encoding models r2 scores against noise ceilings.

Parameters
----------
sub : int
	Used NSD subject.
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
parser.add_argument('--sub', type=int, default=1)
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

r2 = results['r2']
nc = results['noise_ceiling']


# =============================================================================
# Format the results for plotting
# =============================================================================
r2 = {}
nc = {}

for r in args.all_rois:

	r2[r] = results['r2']['s'+str(args.sub)+'_'+r]
	nc[r] = results['noise_ceiling']['s'+str(args.sub)+'_'+r]


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

	# Plot diagonal dashed line
	axs[r].plot(np.arange(-1,1.1,.1), np.arange(-1,1.1,.1), '--k', linewidth=2,
		alpha=.5, label='_nolegend_')

	# Plot the results
	axs[r].scatter(nc[roi], r2[roi], color=colors[0], alpha=.3)
	axs[r].set_aspect('equal')

	# y-axis
	ticks = np.arange(0, 1.1, .2)
	labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('$r²$', fontsize=fontsize)
		plt.yticks(ticks=ticks, labels=labels)
	axs[r].set_ylim(bottom=-.1, top=.9)

	# x-axis
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Noise ceiling', fontsize=fontsize)
		plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize)
	axs[r].set_xlim(left=-.1, right=.9)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

# y-axis
axs[23].set_xlabel('Noise ceiling', fontsize=fontsize)

#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-01', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-02', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-03', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-04', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-05', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-06', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-07', dpi=600)
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-08', dpi=600)

#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-01.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-02.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-03.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-04.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-05.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-06.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-07.svg')
#plt.savefig('encoding_accuracy_scatter_train_dataset-nsd_sub-08.svg')

