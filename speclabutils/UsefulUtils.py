import win32clipboard
import numpy as np
from io import StringIO
import string
import IPython
import matplotlib
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import re
from ipywidgets import widgets, interact
import csv


############################################
# 
# Miscellaneous functions
#
############################################

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

		
def cm2inch(*tupl):
	inch = 2.54
	if type(tupl[0]) == tuple:
		return tuple(i/inch for i in tupl[0])
	else:
		return tuple(i/inch for i in tupl)

def change_mpl_dpi(dpi_set=108.8):
	rcParams = matplotlib.rcParams.copy()
	IPython.display.set_matplotlib_formats('png', facecolor='#FFFFFF', dpi=dpi_set)
	matplotlib.rcParams.update({'figure.figsize': rcParams['figure.figsize']})
	
def norm(a):
	return a/np.max(a)
	
def get_file_title(filepath):
	m = re.search('(\w+)[.].+$', filepath)
	return m.group(1)
		
		
#########################################
#
# Windows clipboard functions
#
#########################################
		
def paste_to_array():
	win32clipboard.OpenClipboard()
	rawData = win32clipboard.GetClipboardData()
	# Some Winspec snapins put in a bad character at the end...remove it if 
	# present.
	lastchar = rawData[-1]
	if (not is_number(lastchar)) and (lastchar not in string.whitespace):
		rawData = rawData[:-1]
	win32clipboard.CloseClipboard()
	
	#is the first line column titles?
	lines = rawData.split('\n')
	col =lines[0].split('\t')
	  
	data = np.genfromtxt(StringIO(rawData),delimiter='\t')

	if sum(map(is_number, col)) < len(col):
		data = data[1:,:]
	else:
		col = []    
	
	return data, col

def copy_string(s):
	win32clipboard.OpenClipboard()
	win32clipboard.EmptyClipboard()
	win32clipboard.SetClipboardText(s)    
	win32clipboard.CloseClipboard()

def copy_array(a, rowdelim='\n', coldelim='\t'):
	dims = a.shape
	
	if len(dims) > 2:
		print('Unable to copy arrays with more than 2 dimensions')
		return 
		
	if len(dims) == 1:
		copy_string(rowdelim.join([str(x) for x in a]))
		
	if len(dims) == 2:
		s = ''
		for i in range(dims[0]):
			s = s + coldelim.join([str(x) for x in a[i]]) + rowdelim
		
		copy_string(s)
	
#########################################
#
# Smoothing functions for 1D data
#
#########################################
	
def fourier_smooth(a, num_points=5):
	a_fft = np.fft.rfft(a)
	a_fft[num_points:] = 0
	
	return np.fft.irfft(a_fft, n=a.shape[0])
   
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	#From:  http://wiki.scipy.org/Cookbook/SavitzkyGolay
	import numpy as np
	from math import factorial

	try:
		window_size = np.abs(int(window_size))
		order = np.abs(int(order))
	except ValueError:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

	
################################################
#
# Matplotlib drawing and formatting functions
#
################################################
def draw_roi(x1, y1, x2, y2, **draw_params):
	codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	
	ax = plt.gca()

	# Form a path
	verts = [(x1, y1),
			 (x1, y2),
			 (x2, y2),
			 (x2, y1),
			 (0, 0)]
	path = Path(verts, codes)

	# Draw the BG region on the image
	patch = patches.PathPatch(path, **draw_params)
	ax.add_patch(patch)

def no_ticks():
	plt.xticks([])
	plt.yticks([])
	
def format_axis(axis_in, major=None, minor=None, direction='out', position=''):
	if axis_in == 'x':
		axis = plt.gca().xaxis
	elif axis_in == 'y':
		axis = plt.gca().yaxis
	
	
	if major != None:
		axis.set_major_locator(MultipleLocator(major))
	if minor != None:
		axis.set_minor_locator(MultipleLocator(minor))

	if len(direction) > 0:
		axis.set_tick_params(which='both', direction=direction)
	else:
		axis.set_tick_params(which='both', direction='out')

	if len(position) > 0:
		axis.set_ticks_position(position)
	else:
		if axis_in == 'x':
			axis.set_ticks_position('bottom')
		else:
			axis.set_ticks_position('left')

def format_axis_loc(axis, major=None, minor=None, offset=0):
	
	if major != None:
		axis.set_major_locator(MultipleLocator(major, offset=offset))
	if minor != None:
		axis.set_minor_locator(MultipleLocator(minor, offset=offset))

# This is a bit more of an advanced programming thing for my usage.  I basically 
# create objects for each data file and use them to keep track of the analysis 
# progression and steps.
class Bunch(object):
	def __init__(self, **kwds):
		self.__dict__.update(kwds)
	
def init_notebook(dpi=120, fontsize=10, linewidth=0.3, major_tick_size=3):
	# load plotting libraries.
	import matplotlib

	
	#matplotlib is the python library for making plots.  The below dictionary updates
	#change the default settings such as font size, tick size, etc.
	matplotlib.rcParams.update(
		{'font.sans-serif': 'Arial',
		 'font.size': fontsize,
		 'font.family': 'Arial',
		 'mathtext.default': 'regular',
		 'axes.linewidth': linewidth, 
		 'axes.labelsize': fontsize,
		 'xtick.labelsize': fontsize,
		 'ytick.labelsize': fontsize,     
		 'lines.linewidth': 0.5,
		 'legend.frameon': False,
		 'xtick.major.width': linewidth,
		 'xtick.minor.width': linewidth,
		 'ytick.major.width': linewidth,
		 'ytick.minor.width': linewidth,
		 'xtick.major.size': major_tick_size,
		 'ytick.major.size': major_tick_size,
		 'xtick.minor.size': major_tick_size/3,
		 'ytick.minor.size': major_tick_size/3
		})

	# This tells matplotlib to create the plots in an "inline" format that the 
	# web browser can display.
	#%matplotlib inline

	# This adjusts the DPI (pixel resolution of the plots displayed).  Increase the DPI 
	# for higher resolution images.
	import IPython
	IPython.display.set_matplotlib_formats('png', facecolor='#FFFFFF', dpi=dpi)
	
	# This sets the default size the plots generated by matplotlib.  It has to be called after the 
	# previous lines that sets the default DPI.
	# This needs to occur after the IPython call.
	matplotlib.rcParams.update(
		{'figure.figsize': (4,3)})
		
def percentile_scale(axis, lb=1, ub=98):
	#scale axis with percentages
	ax = plt.gca()
	
	l = ax.lines[0]
	x_data = l.get_xdata()
	y_data = l.get_ydata()
	
	for l in ax.lines:
		x_data = np.append(x_data, l.get_xdata())
		y_data = np.append(y_data, l.get_ydata())
	
	if axis == 'x' or axis == 'b':
		plt.xlim(np.percentile(x_data, lb), np.percentile(x_data, ub))
	
	if axis == 'y' or axis == 'b':
		plt.ylim(np.percentile(y_data, lb), np.percentile(y_data, ub))
		

def make_axis(fig, lm=5, rm=1, tm=1, bm=6, **kwargs):
	# make an axes object for the figure with mm-sized margins as specified.
	
	w = fig.get_figwidth()*25.4 #convert from inches to mm
	h = fig.get_figheight()*25.4
	
	lm_r = lm/w
	rm_r = rm/w
	tm_r = tm/h
	bm_r = bm/h
	
	ax = fig.add_axes([lm_r, bm_r, 1-lm_r-rm_r, 1-bm_r-tm_r], **kwargs)
	return ax

def norm_smooth(a, window_size=-1, order=1):
	if window_size==-1:
		window_size = int(a.shape[0]/10)
		if window_size % 2 == 0:
			window_size = window_size + 1
	#print(window_size, order, a.shape)
	
	return a/np.max(savitzky_golay(a, window_size=window_size, order=order))

def fast_interact(func, N=-1):
	if N>-1:
		interact(func, i=widgets.IntSlider(value=0, min=0, max=N))
	else:
		interact(func, i=widgets.IntSlider(value=0, min=0))
		
		
######################
#
#Horiba AIST functions
#
######################

def aist_AFM(aist_txt_filepath, save_dir, save_file_name):
	"""
	Saves AFM data as txt for subsequent import into Gwiddyion
	Use 'raw data import' and 'any whitespace' as delimiter
	Returns dictionary of data
	"""
	assert isinstance(aist_txt_filepath, str), '\'aist_txt_filepath\' must by type str'
	assert isinstance(save_dir, str), '\'save_dir\' must by type str'
	assert isinstance(save_file_name, str), '\'save_file_name\' must by type str'
	
	file = aist_txt_filepath
	with open(file) as f:
		iterator = csv.reader(f, delimiter = '\t')
		data = dict()
		num_rows, num_cols = 0, 0
		for row in iterator:
			if row[0][0].isalpha():
				key, val = tuple(row[0].split(' = '))
				data[key] = val
			else:
				if int(row[0]) > num_rows: num_rows = int(row[0])
				if int(row[1]) > num_cols: num_cols = int(row[1])
		topo_map = np.zeros((num_rows + 1, num_cols + 1))
		f.seek(0)
		for row in iterator:
			if row[0][0].isalpha():
				continue
			else:
				topo_map[int(row[0]), int(row[1])] = float(row[2])
		data['topo_map'] = topo_map
		np.savetxt("{}.txt".format(save_dir + save_file_name), topo_map, delimiter="\t", fmt = '%10.5f')
	return data
		
def aist_hyperspec(aist_txt_filepath):
	"""
	Returns dictionary of data
	"""
	assert isinstance(aist_txt_filepath, str), '\'aist_txt_filepath\' must by type str'
	#assert isinstance(save_dir, str), '\'save_dir\' must by type str'
	#assert isinstance(save_file_name, str), '\'save_file_name\' must by type str'
	
	with open(aist_txt_filepath, 'r') as f:
		iterator = csv.reader(f, delimiter = '\t')
		hspec_data= dict()
		num= 0
		d = []
		d_sum = []
		for row in iterator:
			if num == 0:
				hspec_data['info'] = row
			elif num == 1:
				hspec_data['wls'] = [float(i) for i in row]
			else:
				d.append([float(i) for i in row])
				d_sum.append(sum(d[-1]))
			num += 1
		dims = (int(hspec_data['info'][1]), int(hspec_data['info'][0]))
		hdims = (dims[0], dims[1], int(hspec_data['info'][2]))
		
		hspec_data['intensity_map'] = np.reshape(d_sum, dims, order = 'C')
		hspec_data['hyperspec_map'] = np.reshape(d, hdims, order = 'C')
	return hspec_data

# Make a colormap that mimics the default colormap in gwyddion.
rgb = [(22, 4, 1),
(48, 11, 4),
(74, 17, 6),
(100, 24, 9),
(126, 29, 11),
(152, 36, 13),
(172, 49, 18),
(184, 72, 31),
(195, 94, 42),
(206, 118, 54),
(218, 142, 66),
(229, 165, 78),
(241, 188, 90),
(245, 203, 114),
(247, 212, 141),
(249, 222, 169),
(251, 232, 195),
(253, 243, 222),
(255, 254, 250),
(255, 255, 255)]


gwyddion_cmap = LinearSegmentedColormap.from_list('gwyddion', np.array(rgb)/256, N=1000)
	
