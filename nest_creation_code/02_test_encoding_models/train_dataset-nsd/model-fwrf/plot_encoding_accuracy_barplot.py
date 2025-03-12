"""Plot the noise-ceiling-normalized encoding accuracy.

Parameters
----------
all_subs : list of int
	List with all subject numbers.
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
results_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-fmri', 'train_dataset-nsd', 'model-fwrf',
	'encoding_accuracy.npy')

results = np.load(results_dir, allow_pickle=True).item()


# =============================================================================
# Format the results for plotting
# =============================================================================
acc = np.zeros((len(args.all_subs), len(args.all_rois)))
ci = np.zeros((2, len(args.all_rois)))
sig = np.zeros((len(args.all_rois)))

for r, roi in enumerate(args.all_rois):

	# Prediction accuracy
	for s, sub in enumerate(args.all_subs):
		acc[s,r] = np.mean(results['noise_normalized_encoding']\
			['s'+str(sub)+'_'+roi]) * 100

	# Aggregate the confidence intervals
	ci[0,r] = results['ci_lower'][roi] * 100
	ci[1,r] = results['ci_upper'][roi] * 100

	# Aggregate the significance
	sig[r] = results['significance'][roi]

# Difference bewteen prediction accuracy and confidence intervals
ci[0] = np.mean(acc, 0) - ci[0]
ci[1] = ci[1] - np.mean(acc, 0)


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
x = np.arange(len(acc))
width = 0.4

for r, roi in enumerate(args.all_rois):

	# Plot the encoding accuracies
	axs[r].bar(x, acc[:,r], width=width, color=colors[0])

	# Plot the encoding accuracies subject-mean
	y = np.mean(acc[:,r], 0)
	axs[r].plot([min(x), max(x)], [y, y], '--', color='k', linewidth=2,
		alpha=0.4, label='Subjects mean')

	# y-axis
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Explained\nvariance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=0, top=100)

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

#fig.savefig('encoding_accuracy_bar_plot.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_bar_plot.svg', bbox_inches='tight')

