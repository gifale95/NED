"""Plot the encoding accuracy of the synthetic EEG responses, averaged across
channel groups.

Parameters
----------
all_subs : list of int
	List with all subject numbers.
n_synthetic_repeats : int
	Number of synthetic data repeats.
channels : str
	Used EEG channels.
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
parser.add_argument('--all_subs', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--n_synthetic_repeats', default=4, type=int)
parser.add_argument('--channels', type=str, default='all') # ['O', 'P', 'T', 'C', 'F', 'all']
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding model stats
# =============================================================================
results_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_accuracy.npy')

results = np.load(results_dir, allow_pickle=True).item()
ch_names = results['ch_names']
times = results['times']


# =============================================================================
# Channels selection
# =============================================================================
if args.channels != 'OP' and args.channels != 'all':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if args.channels in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	ch_names_new = kept_ch_names
elif args.channels == 'OP':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if 'O' in chan or 'P' in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	ch_names_new = kept_ch_names
elif args.channels == 'all':
	ch_names_new = ch_names
	idx_ch = np.arange(0, len(ch_names))

if args.channels == 'O':
	chan = 'Occipital'
elif args.channels == 'P':
	chan = 'Parietal'
elif args.channels == 'T':
	chan = 'Temporal'
elif args.channels == 'C':
	chan = 'Central'
elif args.channels == 'F':
	chan = 'Frontal'
elif args.channels == 'all':
	chan = 'All'


# =============================================================================
# Format the results for plotting
# =============================================================================
corr_all_rep = np.zeros((len(args.all_subs), len(times)))
corr_single_rep = np.zeros((len(args.all_subs), args.n_synthetic_repeats,
	len(times)))
nc_up = np.zeros((len(args.all_subs), len(times)))
nc_low = np.zeros((len(args.all_subs), len(times)))

for s, sub in enumerate(args.all_subs):

	# Load the results
	corr_sub_all_rep = results['correlation_all_repetitions']['s'+str(sub)]
	corr_sub_single_rep = results['correlation_single_repetitions']['s'+str(sub)]
	nc_up_sub = results['noise_ceiling_upper']['s'+str(sub)]
	nc_low_sub = results['noise_ceiling_lower']['s'+str(sub)]

	# Reshape to Channels x Time
	corr_sub_all_rep = np.reshape(corr_sub_all_rep, (len(ch_names), len(times)))
	corr_sub_single_rep = np.reshape(corr_sub_single_rep,
		(args.n_synthetic_repeats, len(ch_names), len(times)))
	nc_up_sub = np.reshape(nc_up_sub, (len(ch_names), len(times)))
	nc_low_sub = np.reshape(nc_low_sub, (len(ch_names), len(times)))

	# Channels selection
	corr_all_rep[s] = np.mean(corr_sub_all_rep[idx_ch], 0)
	corr_single_rep[s] = np.mean(corr_sub_single_rep[:,idx_ch], 1)
	nc_up[s] = np.mean(nc_up_sub[idx_ch], 0)
	nc_low[s] = np.mean(nc_low_sub[idx_ch], 0)


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
color_noise_ceiling = (150/255, 150/255, 150/255)


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

for s, sub in enumerate(args.all_subs):

	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
		linewidth=3, label='_nolegend_')

	# Plot the noise ceiling
	axs[s].fill_between(times, nc_low[s], nc_up[s], color=color_noise_ceiling,
		alpha=.3, label='_nolegend_')

	# Plot the correlation results (all repeats)
	axs[s].plot(times, corr_all_rep[s], color=colors[0], linewidth=3)

	# Plot the correlation results (single repeats)
	for r in range(args.n_synthetic_repeats):
		if r == 0:
			axs[s].plot(times, corr_single_rep[s,r], '--', color='k',
				linewidth=2, alpha=0.5)
		else:
			axs[s].plot(times, corr_single_rep[s,r], '--', color='k',
				linewidth=2, alpha=0.5, label='_nolegend_')

	# x-axis parameters
	if s in [8, 9, 10, 11]:
		axs[s].set_xlabel('Time (s)', fontsize=fontsize)
		xticks = [0, .1, .2, .3, .4, .5]
		xlabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
		plt.xticks(ticks=xticks, labels=xlabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[10].set_xlabel('Time (s)', fontsize=fontsize)
	axs[11].set_xlabel('Time (s)', fontsize=fontsize)

	# y-axis parameters
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=fontsize)
#		yticks = np.arange(0, 1.01, 0.5)
#		ylabels = [0, 0.5, 1]
#		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_ylim(bottom=-.075, top=1)

	# Title
	tit = chan + ' channels, sub-' + str(sub)
	axs[s].set_title(tit, fontsize=fontsize)

	# Legend
	if s in [0]:
		labels = ['All repeats encoding models',
			'Single repeats encoding models']
		axs[s].legend(labels, ncol=2, fontsize=fontsize, bbox_to_anchor=(1.33, -2.63))

#fig.savefig('encoding_accuracy_channels-O.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-P.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-T.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-C.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-F.png', dpi=600, bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-all.png', dpi=600, bbox_inches='tight')

#fig.savefig('encoding_accuracy_channels-O.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-P.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-T.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-C.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-F.svg', bbox_inches='tight')
#fig.savefig('encoding_accuracy_channels-all.svg', bbox_inches='tight')

