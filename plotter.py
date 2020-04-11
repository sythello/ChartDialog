import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import json
import os
from tqdm import tqdm
import pickle
from collections import OrderedDict

import argparse

'''
Param name design: natural name k_nat = k.capitalize().split('_')
About None:
	Empty string '' should never be passed in; should use None, which will be treated as ''.
	Otherwise, when None is in the valid value set and means empty, no ambiguity.
	When None is not in the valid value set and means default, be careful... (see following)
	For params of '...size' and '...width', use 0 to specify disabling it. For example, line_width=0 means no line.
	When None is not in the valid value set, None shoule be passed in when not applicable or disabled by other slots.
		For example, when marker_size=0, marker_edge_width=None; when plot_type=line, bin_edge_width=None.

Default value: used to make an empty or default plot, especially when plot type is wrong.
'''

PLOT_TYPES = [
	'line',
	'histogram',
	'scatter',
	'bar',
	'matrix_display',
	'contour',
	'streamline',
	'surface_3d',
	'pie'
]

PLOT_TYPE_GROUPS = [
    ['bar plot', 'line chart', 'pie chart'],
    ['streamline plot', 'contour plot'],
    ['histogram', 'scatter plot'],
    ['3D surface', 'matrix display']	# Easy group
]

TUTORIAL_TYPE_GROUPS = [
    ['bar plot', 'line chart', 'pie chart'],
    ['streamline plot', 'contour plot', 'contour plot'],
    ['histogram', 'scatter plot', 'scatter plot'],
    ['3D surface', 'matrix display']	# Easy group
]
# Assume 3rd tutorial polarized

def plotter(
	x, y=None, z=None, u=None, v=None, c_type=None,		# c_type: the correct type for input data
	plot_type=None,				# User specified type
	# ------- Line -------
	line_color=None,			# {None, colors...}
	line_style=None,			# {None (''), '-', '--', '-.', ':'}
	line_width=None,			# {1, 3, 5}
	marker_type=None,			# {None, 'o', 'v', '^', 'D'}
	marker_size=None,			# {5, 9, 13}
	marker_edge_width=None,		# {1, 3}
	marker_edge_color=None,
	marker_face_color=None,		# {None, colors...}
	marker_interval=None,		# {1, 2, 5}

	# ------- Histogram -------
	hist_range=(140, 200),		# Fixed to correct value
	number_of_bins=None,		# {6, 8, 10, 12}
	bar_relative_width=None,	# {0.6, 0.8, 1}, already natural
	bar_edge_width=0,			# {1, 3}
	bar_edge_color=None,
	bar_face_color=None,

	# ------- Scatter -------
	## marker_type=None,		# {'o', 'v', '^', 'D'}
	## marker_size=0,			# {5, 9, 13, diff}
	## marker_edge_width=None,	# {1, 3}
	## marker_edge_color=None,
	## marker_face_color=None,	# {colors..., diff}
	color_map=None,

	# ------- Bar -------
	bar_base=None,				# Fixed to correct value (min - c)
	bar_orientation='vertical',
	bar_width=None,				# {0.6, 0.8, 1}, already natural
	bar_height=None,			# {0.6, 0.8, 1}, already natural
	## bar_edge_width=0,		# {1, 3}
	## bar_edge_color=None,
	## bar_face_color=None,

	# ------- Matrix_display (imshow) -------
	## color_map=None,

	# ------- Contour -------
	contour_plot_type=None,		# {'lined', 'filled'}
	## color_map=None,
	number_of_levels=None,		# {6, 10, 15}, natural
	## line_style=None,			# {'-', '--', '-.', ':'}
	## line_width=0,			# {1, 3, 5}

	# ------- Streamline -------
	density=0.5,				# {0.25, 0.5, 0.75}
	## line_color=None,			# {colors..., diff}
	## color_map=None,
	## line_width=None,			# {1, 3, 5, diff}
	arrow_size=None,			# {1, 3}
	arrow_style=None,			# {lined('->'), solid('-|>')}

	# ------- surface_3d -------
	surface_color=None,			# {colors..., diff}
	## color_map=None,

	# ------- Pie -------
	explode=None,				# {None, 0.03, 0.1}
	precision_digits=0,			# {0, 1, 2, 3}
	percentage_distance_from_center=0.6,	# {0.45, 0.6, 0.75}
	label_distance_from_center=1.1,			# {1.1, 1.25}
	radius=None,				# {0.75, 1, 1.25}
	section_edge_width=None,	# {None, 1, 3}
	section_edge_color=None,

	# ------- Errorbar -------
	show_error_bar=False,
	error_bar_cap_size=0,		# {3, 6, 10}
	error_bar_cap_thickness=0,	# {1, 3}
	error_bar_color=None,

	# ------- Colorbar -------
	color_bar_orientation='vertical',
	color_bar_length=None,		# {0.8, 1}, ratio of length vs. plot height/width
	color_bar_thickness=None,	# {0.1, 0.05, 0.033}, ratio of width vs. length

	# ------- Shared -------
	# plot_title=None,				# Active: all (plot_type.capitalize() + '\n\n\n')
	# x_axis_label=None,			# Fixed to 'X': line, hist, scatter, bar, contour, streamline, surface_3d
	# y_axis_label=None,			# Fixed to 'Y': line, hist, scatter, bar, contour, streamline, surface_3d
	# z_axis_label=None,			# Fixed to 'Z': surface_3d
	data_series_name=None,		# Active: line, hist, scatter (but not vary), bar
	font_size=None,				# {8, 10, 12}; active: all
	invert_x_axis=False,		# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	invert_y_axis=False,		# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	invert_z_axis=False,		# Active: surface_3d
	## Grid lines: active: line, histogram, scatter, bar, contour, streamline
	grid_line_type=None,		# {None, 'horizontal', 'vertical', 'both'} -> Primary
	grid_line_color=None,		# {colors..., favor black & gray}
	grid_line_style=None,		# {'-', '--', '-.', ':'}
	grid_line_width=None,		# {0.5, 1}
	## Axis position: active: line, histogram, scatter, bar, imshow, contour, streamline
	x_axis_position='bottom',
	y_axis_position='left',
	## Scale: active: line, scatter, contour (both axis); bar (non-base axis); histogram, imshow, streamline, surface_3d (always linear)
	x_axis_scale='linear',
	y_axis_scale='linear',
	polarize=False,				# Active: line, hist, contour (can be True); scatter, bar, streamline (always False)
	save_legend=False):

	fig = plt.figure(figsize=(8, 6))
	if polarize and plot_type not in {'line', 'scatter', 'contour'}:
		print('Invalid: polarize = True with plot_type = {}'.format(plot_type))

	if polarize and plot_type in {'line', 'scatter', 'contour'}:
		ax = fig.gca(projection='polar')
	elif plot_type == 'surface_3d':
		ax = fig.gca(projection='3d')
	else:
		ax = fig.gca()

	# Dummy data, avoiding error
	# if y is None:
	# 	y = x
	# if z is None:
	# 	z = np.zeros_like(x)
	# if u is None:
	# 	u = np.zeros_like(x)
	# if v is None:
	# 	v = np.zeros_like(y)

	if plot_type is not None and c_type != plot_type:
		print('Type error: c_type = {}, plot_type = {}. Forcing plot_type = None'.format(c_type, plot_type))
		plot_type = None

	# Universal None transforming (for -style, -size, -width)
	if line_style is None:
		line_style = ''
	if line_width is None:
		line_width = 0
	if marker_size is None:
		marker_size = 0
	if marker_edge_width is None:
		marker_edge_width = 0
	if bar_edge_width is None:
		bar_edge_width = 0
	if explode is None:
		explode = 0

	artist = None

	# Do plotting based on type
	if plot_type == 'line':
		plot_func_kwargs = {
			'color' : line_color,
			'linestyle' : line_style,
			'linewidth' : line_width,
			'marker' : marker_type,
			'markersize' : marker_size,
			'markeredgewidth' : marker_edge_width,
			'markeredgecolor' : marker_edge_color,
			'markerfacecolor' : marker_face_color,
			'markevery' : marker_interval,
			'label' : data_series_name}
		artist = ax.plot(x, y, **plot_func_kwargs)
		if show_error_bar:
			# Can show error bar w/o markers
			errorbar_func_kwargs = {
				'capsize': error_bar_cap_size,
				'capthick': error_bar_cap_thickness,
				'ecolor': error_bar_color,
				'elinewidth': 3,
				# 'markeredgewidth': 3,
				'errorevery': 1 if marker_interval is None else marker_interval,
				'linestyle': '',
				}
			ax.errorbar(x, y, yerr=z, **errorbar_func_kwargs)

	elif plot_type == 'histogram':
		func_kwargs = {
			'bins' : number_of_bins,
			'range' : hist_range,
			'rwidth' : bar_relative_width,
			'color' : bar_face_color,
			'linewidth' : bar_edge_width,
			'edgecolor' : bar_edge_color,
			'label' : data_series_name}

		ax.hist(x, **func_kwargs)
		artist = Rectangle((0,0), 1, 1, facecolor=bar_face_color, edgecolor=bar_edge_color, linewidth=bar_edge_width)	# Fake artist for legend

	elif plot_type == 'scatter':
		# slots: [marker_type, marker_size, marker_edge_width, marker_edge_color, marker_face_color, color_map]
		func_kwargs = {
			'marker' : marker_type,
			's' : (z ** 2) if marker_size == 'diff' else (marker_size ** 2),
			'c' : z if marker_face_color == 'diff' else marker_face_color,
			'linewidths' : marker_edge_width,
			'edgecolors' : marker_edge_color,
			'label' : data_series_name}
		if marker_face_color == 'diff':
			func_kwargs['cmap'] = color_map
		artist = ax.scatter(x, y, **func_kwargs)

	elif plot_type == 'bar':
		# slots: [bar_orientation, bar_relative_width, bar_relative_height, bar_edge_width, bar_edge_color, bar_face_color]
		func_kwargs = {
			'color' : bar_face_color,
			'linewidth' : bar_edge_width,
			'edgecolor' : bar_edge_color,
			'label' : data_series_name,
			'error_kw' : {'capsize': error_bar_cap_size, 'capthick': error_bar_cap_thickness, 'ecolor': error_bar_color}}
		if isinstance(x[0], str):
			print('X are strings, can\'t have error bar')

		if bar_orientation == 'horizontal':
			# print('--- bar-horizontal ---')
			func_kwargs['height'] = bar_height
			func_kwargs['left'] = bar_base
			artist = ax.barh(x, y, xerr=z if show_error_bar else None, **func_kwargs)
		elif bar_orientation == 'vertical':
			# print('--- bar-vertical ---')
			func_kwargs['width'] = bar_width
			func_kwargs['bottom'] = bar_base
			artist = ax.bar(x, y, yerr=z if show_error_bar else None, **func_kwargs)
		else:
			print('Invalid bar_orientation: {}'.format(bar_orientation))

	elif plot_type == 'matrix_display':
		artist = ax.imshow(x, cmap=color_map)

	elif plot_type == 'contour':
		func_kwargs = {
			# 'levels': number_of_levels,
			'levels': np.linspace(z.min(), z.max(), number_of_levels + 1),
			'cmap': color_map
		}
		if contour_plot_type == 'filled':
			artist = ax.contourf(x, y, z, **func_kwargs)
		elif contour_plot_type == 'lined':
			func_kwargs['linewidths'] = line_width
			func_kwargs['linestyles'] = line_style
			artist = ax.contour(x, y, z, **func_kwargs)
		else:
			print('Invalid contour_plot_type: {}'.format(contour_plot_type))

	elif plot_type == 'streamline':
		func_kwargs = {
			'color': z if line_color == 'diff' else line_color,
			'linewidth': (0.5 + 4.5 * (z - z.min()) / (z.max() - z.min())) if line_width == 'diff' else line_width,
			'density': density,
			'arrowsize': arrow_size,
			'arrowstyle': arrow_style
		}
		if line_color == 'diff':
			func_kwargs['cmap'] = color_map
		# print(func_kwargs)
		artist_ = ax.streamplot(x, y, u, v, **func_kwargs)
		artist = artist_.lines

	elif plot_type == 'surface_3d':
		if surface_color == 'diff':
			artist = ax.plot_surface(x, y, z, cmap=color_map)
		else:
			artist = ax.plot_surface(x, y, z, color=surface_color)

	elif plot_type == 'pie':
		# slots: [explode, precision_digits, percentage_distance_from_center, label_distance_from_center, radius, section_edge_width, section_edge_color]

		func_kwargs = {
			'explode': [explode] * len(x),
			'autopct': '%.{}f%%'.format(precision_digits),
			'pctdistance': percentage_distance_from_center,
			'labeldistance': label_distance_from_center,
			'radius': radius,
			'wedgeprops': {'edgecolor': section_edge_color, 'linewidth': section_edge_width},
			'textprops': {'fontsize': font_size}
		}

		_wedges, _labels, _pcts = ax.pie(y, labels=x, **func_kwargs)
		# Workaround for label fontsize problem
		for _l in _labels:
			_l.set_fontsize(font_size)


	elif plot_type is not None:
		print('Invalid plot type: {}'.format(plot_type))
		return fig

	if plot_type is not None and color_map is not None:
		func_kwargs = {
			'orientation': color_bar_orientation,
			'shrink': color_bar_length,
			'aspect': 1.0 / color_bar_thickness,
			'pad': 0.15
		}
		fig.colorbar(artist, ax=ax, **func_kwargs)

	if plot_type is not None:
		plot_title = PLOT_TYPE_V2S[plot_type].capitalize() + '\n\n\n'
		ax.set_title(plot_title)

	if plot_type not in {'pie'}:
		ax.tick_params(labelsize=font_size)

	legend_possible = (plot_type in {'line', 'histogram', 'bar'}) or (plot_type == 'scatter' and marker_face_color != 'diff')

	if data_series_name is not None:
		if not legend_possible:
			print('Invalid: data_series_name given for {}{}'.format(plot_type, '-diff' if plot_type == 'scatter' else ''))
		else:
			ax.legend(fontsize=font_size)

	if x_axis_scale == 'log':
		if (plot_type not in {'line', 'scatter', 'bar', 'contour'}) or (plot_type == 'bar' and bar_orientation == 'vertical'):
			print('Invalid x_axis_scale: {} for {}{}'.format(x_axis_scale, plot_type, '-' + bar_orientation if plot_type == 'bar' else ''))
		else:
			ax.set_xscale('log')
	if y_axis_scale == 'log':
		if (plot_type not in {'line', 'scatter', 'bar', 'contour'}) or (plot_type == 'bar' and bar_orientation == 'horizontal'):
			print('Invalid y_axis_scale: {} for {}{}'.format(y_axis_scale, plot_type, '-' + bar_orientation if plot_type == 'bar' else ''))
		else:
			ax.set_yscale('log')

	x_axis_label = 'X'
	y_axis_label = 'Y'
	z_axis_label = 'Z'
	if plot_type != 'pie' and (not polarize):
		ax.set_xlabel(x_axis_label, fontsize=font_size)
	if plot_type != 'pie' and (not polarize):
		ax.set_ylabel(y_axis_label, fontsize=font_size)
	if plot_type == 'surface_3d':
		ax.set_zlabel(z_axis_label, fontsize=font_size)

	if invert_x_axis:
		if plot_type == 'pie':
			print('Invalid: invert_x_axis set for {}'.format(plot_type))
		elif plot_type == 'surface_3d':
			ax.set_xlim3d(ax.get_xlim3d()[::-1])
		else:
			ax.invert_xaxis()
	if invert_y_axis:
		if plot_type == 'pie':
			print('Invalid: invert_y_axis set for {}'.format(plot_type))
		elif plot_type == 'surface_3d':
			ax.set_ylim3d(ax.get_ylim3d()[::-1])
		else:
			ax.invert_yaxis()
	if invert_z_axis:
		if plot_type != 'surface_3d':
			print('Invalid: invert_z_axis set for {}'.format(plot_type))
		elif plot_type == 'surface_3d':
			ax.set_zlim3d(ax.get_zlim3d()[::-1])
		# else:
		# 	ax.invert_zaxis()

	if plot_type not in {'surface_3d', 'pie', 'matrix_display'}:
		# Can add grid lines
		if grid_line_type in {'vertical', 'both'}:
			ax.xaxis.grid(True, color=grid_line_color, linewidth=grid_line_width, linestyle=grid_line_style)
		else:
			ax.xaxis.grid(False)
		if grid_line_type in {'horizontal', 'both'}:
			ax.yaxis.grid(True, color=grid_line_color, linewidth=grid_line_width, linestyle=grid_line_style)
		else:
			ax.yaxis.grid(False)

		if plot_type in {'contour'}:
			ax.set_axisbelow(False)
		else:
			ax.set_axisbelow(True)

	if (plot_type not in {'surface_3d', 'pie'}) and (not polarize):
		# Can do axis moving, ticklabels setting
		if x_axis_position == 'top':
			ax.xaxis.tick_top()
			ax.xaxis.set_label_position('top')
		elif x_axis_position == 'bottom':
			ax.xaxis.tick_bottom()
			ax.xaxis.set_label_position('bottom')
		elif x_axis_position is not None:
			print('Invalid x_axis_position: {}'.format(x_axis_position))

		if y_axis_position == 'right':
			ax.yaxis.tick_right()
			ax.yaxis.set_label_position('right')
		elif y_axis_position == 'left':
			ax.yaxis.tick_left()
			ax.yaxis.set_label_position('left')
		elif y_axis_position is not None:
			print('Invalid y_axis_position: {}'.format(y_axis_position))

	# To solve title cutting-off
	fig.tight_layout()

	if save_legend:
		fig_legend = plt.figure(figsize=(4, 0.5))
		if legend_possible:
			fig_legend.legend([artist], [label], loc='center', handlelength=12)
		return fig, fig_legend
	else:
		return fig


### Constant global dictionaries for slot & value mapping, all slots
def _invert_dict(d):
	return dict([(v, k) for k, v in d.items()])

## V: value, S: natural string
COLOR_V2S = {
	'red': 'red',
	'orange': 'orange',
	'green': 'green',
	'blue': 'blue',
	'm': 'magenta',
	'gray': 'gray',
	'black': 'black',
	'diff': 'different'
}
COLOR_S2V = _invert_dict(COLOR_V2S)

WIDTH_V2S = {
	0.8: 'very thin',
	1.2: 'thin',
	3: 'medium',
	5: 'thick',
	'diff': 'different'
}
WIDTH_S2V = _invert_dict(WIDTH_V2S)

PLOT_TYPE_V2S = {
	'line': 'line chart',
	'histogram': 'histogram',
	'scatter': 'scatter plot',
	'bar': 'bar plot',
	'matrix_display': 'matrix display',
	'contour': 'contour plot',
	'streamline': 'streamline plot',
	'surface_3d': '3D surface',
	'pie': 'pie chart'
}
PLOT_TYPE_S2V = _invert_dict(PLOT_TYPE_V2S)

LINE_STYLE_V2S = {
	'-' : 'solid',
	'--' : 'dashed',
	'-.' : 'dashed dots',
	':' : 'dotted',
}
LINE_STYLE_S2V = _invert_dict(LINE_STYLE_V2S)

MARKER_TYPE_V2S = {
	'o' : 'circles',
	'v' : 'down triangles',
	'^' : 'triangles',
	'D' : 'diamonds',
}
MARKER_TYPE_S2V = _invert_dict(MARKER_TYPE_V2S)

MARKER_SIZE_V2S = {
	8: 'small',
	12: 'medium',
	16: 'large',
	'diff': 'different'
}
MARKER_SIZE_S2V = _invert_dict(MARKER_SIZE_V2S)

COLOR_MAP_V2S = {
	'Reds': 'transparent to solid red',
	'Blues': 'transparent to solid blue',
	'Greens': 'transparent to solid green',
	'YlOrRd': 'transparent yellow to solid red',
	'GnBu': 'transparent green to solid blue',
	'BuPu': 'transparent blue to dark purple',
	'spring': 'magenta to yellow',
	'autumn': 'red to yellow',
	'cool': 'light cyan to magenta',
	'PRGn': 'purple to white to green',
	'RdBu': 'red to white to blue',
	'RdYlGn': 'red to yellow to green'
}
COLOR_MAP_S2V = _invert_dict(COLOR_MAP_V2S)

DENSITY_V2S = {
	0.25: 'loose',
	0.5: 'medium',
	0.75: 'dense'
}
DENSITY_S2V = _invert_dict(DENSITY_V2S)

ARROW_SIZE_V2S = {
	2.5: 'small',
	4.5: 'large'
}
ARROW_SIZE_S2V = _invert_dict(ARROW_SIZE_V2S)

ARROW_STYLE_V2S = {
	'->': 'curve',
	'-|>': 'solid'
}
ARROW_STYLE_S2V = _invert_dict(ARROW_STYLE_V2S)

EXPLODE_V2S = {
	0.03: 'small',
	0.1: 'large'
}
EXPLODE_S2V = _invert_dict(EXPLODE_V2S)

PERCENTAGE_DISTANCE_FROM_CENTER_V2S = {
	0.45: 'near',
	0.6: 'medium',
	0.75: 'far'
}
PERCENTAGE_DISTANCE_FROM_CENTER_S2V = _invert_dict(PERCENTAGE_DISTANCE_FROM_CENTER_V2S)

LABEL_DISTANCE_FROM_CENTER_V2S = {
	1.1: 'near',
	1.25: 'far'
}
LABEL_DISTANCE_FROM_CENTER_S2V = _invert_dict(LABEL_DISTANCE_FROM_CENTER_V2S)

RADIUS_V2S = {
	0.8: 'small',
	1: 'medium',
	1.2: 'large'
}
RADIUS_S2V = _invert_dict(RADIUS_V2S)

ERROR_BAR_CAP_SIZE_V2S = {
	4 : 'small',
	7 : 'medium',
	10 : 'large'
}
ERROR_BAR_CAP_SIZE_S2V = _invert_dict(ERROR_BAR_CAP_SIZE_V2S)

ERROR_BAR_CAP_THICKNESS_V2S = {
	1 : 'thin',
	3 : 'thick'
}
ERROR_BAR_CAP_THICKNESS_S2V = _invert_dict(ERROR_BAR_CAP_THICKNESS_V2S)

COLOR_BAR_LENGTH_V2S = {
	0.8 : 'short',
	1 : 'long'
}
COLOR_BAR_LENGTH_S2V = _invert_dict(COLOR_BAR_LENGTH_V2S)

COLOR_BAR_THICKNESS_V2S = {
	0.1 : 'thick',
	0.05 : 'medium', 
	0.033 : 'thin'
}
COLOR_BAR_THICKNESS_S2V = _invert_dict(COLOR_BAR_THICKNESS_V2S)

FONT_SIZE_V2S = {
	8 : 'small',
	10 : 'medium',
	12 : 'large'
}
FONT_SIZE_S2V = _invert_dict(FONT_SIZE_V2S)

SLOTS_NEED_MAPPING = ['plot_type', 'marker_type', 'marker_size', 'color_map', \
	'density', 'arrow_size', 'arrow_style', \
	'explode', 'percentage_distance_from_center', 'label_distance_from_center', 'radius', \
	'error_bar_cap_size', 'error_bar_cap_thickness', 'color_bar_length', 'color_bar_thickness', 'font_size']

# 53 slots
# Default dict, initial state
DEFAULT_DICT = {
	'plot_type' : None,
	# ------- Line -------
	'line_color' : None,
	'line_style' : None,
	'line_width' : None,
	'marker_type' : None,
	'marker_size' : None,
	'marker_edge_width' : None,
	'marker_edge_color' : None,
	'marker_face_color' : None,
	'marker_interval' : None,
	# ------- Histogram -------
	'number_of_bins' : None,
	'bar_relative_width' : None,
	'bar_edge_width' : None,
	'bar_edge_color' : None,
	'bar_face_color' : None,
	# ------- Scatter -------
	## marker_type,
	## marker_size,
	## marker_edge_width,
	## marker_edge_color,
	## marker_face_color,
	'color_map' : None,
	# ------- Bar -------
	'bar_orientation' : None,
	'bar_width' : None,				# {0.6, 0.8, 1}, already natural
	'bar_height' : None,			# {0.6, 0.8, 1}, already natural
	## bar_edge_width,
	## bar_edge_color,
	## bar_face_color,
	# ------- Matrix_display (imshow) -------
	## color_map=None,
	# ------- Contour -------
	'contour_plot_type': None,		# {'lined', 'filled'}
	## color_map=None,
	'number_of_levels': None,		# {6, 10, 15}, natural
	## line_style=None,				# {'-', '--', '-.', ':'}
	## line_width=0,				# {1, 3, 5}
	# ------- Streamline -------
	'density': None,				# {0.25, 0.5, 0.75}
	## line_color=None,				# {colors..., diff}
	## color_map=None,
	## line_width=None,				# {1, 3, 5, diff}
	'arrow_size': None,				# {1, 3}
	'arrow_style': None,			# {lined('->'), solid('-|>')}
	# ------- surface_3d -------
	'surface_color': None,			# {colors..., diff}
	## color_map=None,
	# ------- Pie -------
	'explode': None,				# {None, 0.03, 0.1}
	'precision_digits': None,		# {0, 1, 2, 3}
	'percentage_distance_from_center': None,	# {0.45, 0.6, 0.75}
	'label_distance_from_center': None,			# {1.1, 1.25}
	'radius': None,					# {0.75, 1, 1.25}
	'section_edge_width': None,		# {None, 1, 3}
	'section_edge_color': None,

	# ------- Errorbar -------
	'show_error_bar' : None,
	'error_bar_cap_size' : None,		# {3, 7, 10}, can't be smaller than line_width, otherwise cap invisible
	'error_bar_cap_thickness' : None,	# {1, 3}
	'error_bar_color' : None,
	# ------- Colorbar -------
	'color_bar_orientation' : None,
	'color_bar_length' : None,		# {0.8, 1}, ratio of length vs. plot height/width
	'color_bar_thickness' : None,	# {0.1, 0.05, 0.033}, ratio of width vs. length
	# ------- Shared -------
	# plot_title=None,				# Active: all (plot_type.capitalize() + '\n\n\n')
	# x_axis_label=None,			# Fixed to 'X': line, hist, scatter, bar, contour, streamline, surface_3d
	# y_axis_label=None,			# Fixed to 'Y': line, hist, scatter, bar, contour, streamline, surface_3d
	# z_axis_label=None,			# Fixed to 'Z': surface_3d
	'data_series_name' : None,		# Active: line, hist, scatter (but not vary), bar
	'font_size' : None,				# {8, 10, 12}; active: all
	'invert_x_axis' : None,			# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	'invert_y_axis' : None,			# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	'invert_z_axis' : None,			# Active: surface_3d
	## Grid lines: active: line, histogram, scatter, bar, contour, streamline
	'grid_line_type' : None,		# {None, 'horizontal', 'vertical', 'both'} -> Primary
	'grid_line_color' : None,		# {colors..., favor black & gray}
	'grid_line_style' : None,		# {'-', '--', '-.', ':'}
	'grid_line_width' : None,		# {0.5, 1}
	## Axis position: active: line, histogram, scatter, bar, imshow, contour, streamline
	'x_axis_position' : None,
	'y_axis_position' : None,
	## Scale: active: line, scatter, contour (both axis); bar (non-base axis); histogram, imshow, streamline, surface_3d (always linear)
	'x_axis_scale' : None,
	'y_axis_scale' : None,
	'polarize' : None,				# Active: line, hist, contour (can be True); scatter, bar, streamline (always False)
}

## This is (approximately) the same as Operator Panel order
SLOTS_NAT_ORDER = [
	'plot_type',
	# ------- Contour -------
	'contour_plot_type',		# {'lined', 'filled'}
	## color_map=None,
	'number_of_levels',			# {6, 10, 15}, natural
	## line_style=None,			# {'-', '--', '-.', ':'}
	## line_width=0,			# {1, 3, 5}
	# ------- Streamline -------
	'density',					# {0.25, 0.5, 0.75}
	## line_color=None,			# {colors..., diff}
	## color_map=None,
	## line_width=None,			# {1, 3, 5, diff}
	'arrow_size',				# {1, 3}
	'arrow_style',				# {lined('->'), solid('-|>')}
	# ------- 3D surface -------
	'surface_color',			# {colors..., diff}
	## color_map=None,
	# ------- Line -------
	'line_style',
	'line_width',
	'line_color',
	'marker_type',
	'marker_size',
	'marker_face_color',
	'marker_edge_width',
	'marker_edge_color',
	'marker_interval',
	# ------- Histogram -------
	'number_of_bins',
	'bar_relative_width',
	'bar_face_color',
	'bar_edge_width',
	'bar_edge_color',
	# ------- Bar -------
	'bar_orientation',
	'bar_width',			# {0.6, 0.8, 1}, already natural
	'bar_height',			# {0.6, 0.8, 1}, already natural
	## bar_edge_width,
	## bar_edge_color,
	## bar_face_color,
	# ------- Scatter -------
	## marker_type,
	## marker_size,
	## marker_edge_width,
	## marker_edge_color,
	## marker_face_color,
	'color_map',
	# ------- Matrix_display (imshow) -------
	## color_map=None,
	# ------- Pie -------
	'explode',					# {None, 0.03, 0.1}
	'precision_digits',			# {0, 1, 2, 3}
	'percentage_distance_from_center',	# {0.45, 0.6, 0.75}
	'label_distance_from_center',		# {1.1, 1.25}
	'radius',					# {0.75, 1, 1.25}
	'section_edge_width',		# {None, 1, 3}
	'section_edge_color',
	# ------- Errorbar -------
	'show_error_bar',
	'error_bar_cap_size',		# {3, 7, 10}, can't be smaller than line_width, otherwise cap invisible
	'error_bar_cap_thickness',	# {1, 3}
	'error_bar_color',
	# ------- Colorbar -------
	'color_bar_orientation',
	'color_bar_length',		# {0.8, 1}, ratio of length vs. plot height/width
	'color_bar_thickness',	# {0.1, 0.05, 0.033}, ratio of width vs. length
	# ------- Shared -------
	# plot_title=None,		# Active: all (plot_type.capitalize() + '\n\n\n')
	# x_axis_label=None,	# Fixed to 'X': line, hist, scatter, bar, contour, streamline, surface_3d
	# y_axis_label=None,	# Fixed to 'Y': line, hist, scatter, bar, contour, streamline, surface_3d
	# z_axis_label=None,	# Fixed to 'Z': surface_3d
	'polarize',				# Active: line, hist, contour (can be True); scatter, bar, streamline (always False)
	## Scale: active: line, scatter, contour (both axis); bar (non-base axis); histogram, imshow, streamline, surface_3d (always linear)
	'x_axis_scale',
	'y_axis_scale',
	## Axis position: active: line, histogram, scatter, bar, imshow, contour, streamline
	'x_axis_position',
	'y_axis_position',
	'data_series_name',		# Active: line, hist, scatter (but not vary), bar
	'font_size',				# {8, 10, 12}; active: all
	'invert_x_axis',			# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	'invert_y_axis',			# Active: line, hist, scatter, bar, contour, streamline, surface_3d
	'invert_z_axis',			# Active: surface_3d
	## Grid lines: active: line, histogram, scatter, bar, contour, streamline
	'grid_line_type',		# {None, 'horizontal', 'vertical', 'both'} -> Primary
	'grid_line_color',		# {colors..., favor black & gray}
	'grid_line_style',		# {'-', '--', '-.', ':'}
	'grid_line_width',		# {0.5, 1}
]

### Auxiliary plotter/sampling functions

def plotter_kwargs_unnaturalize(**kwargs):
	'''
	Carry out plotter() functionality with natural-form values (keys are the same as plotter())
	'''
	call_params = {}
	for k, v in kwargs.items():
		_v = v
		if v is None or v == '':
			_v = None
		elif k.endswith('color'):
			_v = COLOR_S2V[v]
		elif k.endswith('line_width') or k.endswith('edge_width'):
			_v = WIDTH_S2V[v]
		elif k.endswith('line_style'):
			_v = LINE_STYLE_S2V[v]
		elif k in SLOTS_NEED_MAPPING:
			map_name = k.upper() + '_S2V'
			map_dict = eval(map_name)
			if v not in map_dict:
				print(kwargs)
				print(k, v)
			_v = map_dict[v]

		call_params[k] = _v

	return call_params

def plotter_kwargs_type_uniformize(kwargs):
	# Change all numpy types back to python types
	plotter_kwargs = dict(kwargs)
	for k in plotter_kwargs:
		try:
			plotter_kwargs[k] = plotter_kwargs[k].item()
		except:
			pass
	return plotter_kwargs

def plotter_kwargs_validate_polarize_axis(kwargs, hard=False):
	if kwargs is None:
		return None

	plotter_kwargs = dict(kwargs)
	if plotter_kwargs['polarize']:
		# Polar, disable incompatible slots
		plotter_kwargs['show_error_bar'] = None
		plotter_kwargs['x_axis_position'] = None
		plotter_kwargs['y_axis_position'] = None
		plotter_kwargs['x_axis_scale'] = None
		plotter_kwargs['y_axis_scale'] = None
		plotter_kwargs['invert_x_axis'] = None
		plotter_kwargs['invert_y_axis'] = None
		if plotter_kwargs['line_width'] == 'thick':
			# For line: look weird, discard
			return None
	elif hard:
		if (plotter_kwargs['plot_type'] not in {'pie chart'}) and (plotter_kwargs['invert_x_axis'] != True) and (plotter_kwargs['invert_y_axis'] != True):
			# No inverted axis, discard
			return None
		if (plotter_kwargs['plot_type'] in {'line chart', 'scatter plot', 'bar plot', 'contour plot'}) and (plotter_kwargs['x_axis_scale'] != 'log') and (plotter_kwargs['y_axis_scale'] != 'log'):
			# No log-scale, discard
			return None
		if (plotter_kwargs['plot_type'] not in {'pie chart', '3D surface'}) and (plotter_kwargs['x_axis_position'] != 'top') and (plotter_kwargs['y_axis_position'] != 'right'):
			# No axis moving, discard
			return None

	return plotter_kwargs

def plotter_kwargs_validate_grid_line(kwargs, hard=False):
	if kwargs is None:
		return None

	plotter_kwargs = dict(kwargs)
	if plotter_kwargs['plot_type'] == 'contour plot' and plotter_kwargs['grid_line_color'] in plotter_kwargs['color_map']:
		# Grid line color conflicts with cmap, change to gray (for contour, cmap is never None, so no need to check)
		plotter_kwargs['grid_line_color'] = 'black'

	if plotter_kwargs['grid_line_type'] is None:
		plotter_kwargs['grid_line_width'] = None
		plotter_kwargs['grid_line_style'] = None
		plotter_kwargs['grid_line_color'] = None

	return plotter_kwargs

# Sorted slots
# SLOTS = ['fontsize', 'hist_edgecolor', 'hist_edgewidth', 'hist_facecolor', 'hist_nbins', 'hist_rwidth', \
# 	'invert_xaxis', 'invert_yaxis', 'label', 'line_color', 'line_marker', \
# 	'line_markeredgecolor', 'line_markeredgewidth', 'line_markerfacecolor', 'line_markersize', 'line_markevery', \
# 	'line_style', 'line_width', 'plot_type']

### Sampling
# Common kw use the public pool
# Common2 kw use the public pool2
# Special kw use a specialized pool
# (Above) these are used to make pools different for some slots between sample_one from sample_one_hard
# Use None for whatever slots not supported in the plot type or not activated.
# Only data_series_name is included in [type]_label_settings; others, like x_axis_label, y_axis_label, plot_title, are fixed.


LINE_COMMON_KWS = ['line_color', 'line_style', 'line_width', 'marker_type', 'marker_size', \
	'marker_edge_width', 'marker_edge_color', 'marker_face_color', 'marker_interval', \
	'show_error_bar', 'error_bar_cap_size', 'error_bar_cap_thickness', 'error_bar_color', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position', 'x_axis_scale', 'y_axis_scale', 'polarize']
LINE_COMMON2_KWS = []
LINE_SPECIAL_KWS = []

HISTOGRAM_COMMON_KWS = ['number_of_bins', 'bar_relative_width', 'bar_edge_width', 'bar_edge_color', 'bar_face_color', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position']
HISTOGRAM_COMMON2_KWS = []
HISTOGRAM_SPECIAL_KWS = []

SCATTER_COMMON_KWS = ['marker_edge_width', 'marker_edge_color', 'color_map', \
	'color_bar_orientation', 'color_bar_length', 'color_bar_thickness', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position', 'x_axis_scale', 'y_axis_scale', 'polarize']
SCATTER_COMMON2_KWS = ['marker_type', 'marker_size', 'marker_face_color']
SCATTER_SPECIAL_KWS = []

BAR_COMMON_KWS = ['bar_orientation', 'bar_width', 'bar_height', 'bar_edge_width', 'bar_edge_color', 'bar_face_color', \
	'show_error_bar', 'error_bar_cap_size', 'error_bar_cap_thickness', 'error_bar_color', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position', 'x_axis_scale', 'y_axis_scale']
BAR_COMMON2_KWS = []
BAR_SPECIAL_KWS = []

MATRIX_DISPLAY_COMMON_KWS = ['color_map', \
	'color_bar_orientation', 'color_bar_length', 'color_bar_thickness', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'x_axis_position', 'y_axis_position']
MATRIX_DISPLAY_COMMON2_KWS = []
MATRIX_DISPLAY_SPECIAL_KWS = []

CONTOUR_COMMON_KWS = ['contour_plot_type', 'number_of_levels', 'color_map', 'line_width', \
	'color_bar_orientation', 'color_bar_length', 'color_bar_thickness', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position', 'x_axis_scale', 'y_axis_scale', 'polarize']
CONTOUR_COMMON2_KWS = ['line_style']
CONTOUR_SPECIAL_KWS = []

STREAMLINE_COMMON_KWS = ['density', 'color_map', 'arrow_size', 'arrow_style', \
	'color_bar_orientation', 'color_bar_length', 'color_bar_thickness', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'grid_line_type', 'grid_line_style', 'grid_line_width', 'grid_line_color', \
	'x_axis_position', 'y_axis_position']
STREAMLINE_COMMON2_KWS = ['line_color', 'line_width']
STREAMLINE_SPECIAL_KWS = []

SURFACE_3D_COMMON_KWS = ['surface_color', 'color_map', \
	'color_bar_orientation', 'color_bar_length', 'color_bar_thickness', \
	'font_size', 'invert_x_axis', 'invert_y_axis', 'invert_z_axis']
SURFACE_3D_COMMON2_KWS = []
SURFACE_3D_SPECIAL_KWS = []

PIE_COMMON_KWS = ['explode', 'precision_digits', 'percentage_distance_from_center', 'label_distance_from_center', \
	'radius', 'section_edge_width', 'section_edge_color', \
	'font_size']
PIE_COMMON2_KWS = []
PIE_SPECIAL_KWS = []

# With data_series_name: line, bar, hist, scatter


NORMAL_COLOR_POOL = ['red', 'orange', 'green', 'blue', 'magenta', 'gray', 'black']

## Always use natural values outside plotter()!
def Line_data_sampler(**kwargs):
	if kwargs['polarize']:
		data_series_name_pool = ['Position', 'Trajectory']
		data_series_name = np.random.choice(data_series_name_pool)
		label_setting = {'data_series_name': data_series_name}

		x = np.linspace(0, 4*np.pi, 50)
		y = np.linspace(0, 4*np.pi, 50) * (10 + np.random.uniform(-1, 1, 50))
		data = {'x': x, 'y': y, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	elif kwargs['x_axis_scale'] == 'log' or kwargs['y_axis_scale'] == 'log':
		data_series_name_pool = ['Data Size', 'Power']
		data_series_name = np.random.choice(data_series_name_pool)
		# label_setting = {'plot_title': 'Line chart\n\n\n', 'x_axis_label': 'X', 'y_axis_label': 'Y', 'data_series_name': data_series_name}
		label_setting = {'data_series_name': data_series_name}

		x = 10 ** np.linspace(0, 5, 11) if kwargs['x_axis_scale'] == 'log' else np.linspace(0, 5, 11)
		if kwargs['y_axis_scale'] == 'log':
			y = 10 ** (np.linspace(0, 5, 11) + np.random.uniform(0, 2, 11)).clip(0, 6)
			yerr = np.random.uniform(0.5, 0.75, 11) * y
		else:
			y = np.linspace(0, 5, 11) + np.random.uniform(0, 2, 11)
			yerr = np.random.uniform(0.3, 0.6, 11)
		data = {'x': x, 'y': y, 'z': yerr, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	else:
		data_series_name_pool = ['Data', 'Score', 'Energy', 'Stock Price', 'User Ratings']
		data_series_name = np.random.choice(data_series_name_pool)
		# label_setting = {'plot_title': 'Line chart\n\n\n', 'x_axis_label': 'X', 'y_axis_label': 'Y', 'data_series_name': data_series_name}
		label_setting = {'data_series_name': data_series_name}

		x = np.linspace(0, 10, 11)
		y = np.random.uniform(155, 185, 11) + np.random.normal(0, 15, 11)
		yerr = np.random.uniform(2, 4, 11)
		data = {'x': x, 'y': y, 'z': yerr, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
		
	return data, label_setting

def Histogram_data_sampler(**kwargs):
	data_series_name_pool = ['Data', 'Score', 'Energy', 'Stock Price', 'User Ratings']
	data_series_name = np.random.choice(data_series_name_pool)
	# label_setting = {'plot_title': 'Histogram\n\n\n', 'x_axis_label': 'X', 'y_axis_label': 'Y', 'data_series_name': data_series_name}
	label_setting = {'data_series_name': data_series_name}

	x = np.concatenate((np.random.uniform(155, 185, 37) + np.random.normal(0, 15, 37), np.linspace(140, 200, 13)), axis=0)
	hist_range = (140, 200)
	data = {'x': x, 'hist_range': hist_range, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def Scatter_data_sampler(**kwargs):
	if kwargs['polarize']:
		data_series_name_pool = ['Locations', 'Branches', 'Resource Detected']
		data_series_name = np.random.choice(data_series_name_pool)
		label_setting = {'data_series_name': data_series_name}

		x = np.random.uniform(0, 2*np.pi, 50)
		y = (2 + np.sin(x)/2 + np.random.uniform(-1, 1, 50))
		z = np.random.uniform(8, 16, 50)
		data = {'x': x, 'y': y, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	else:
		data_series_name_pool = ['Locations', 'Branches', 'Resource Detected']
		data_series_name = np.random.choice(data_series_name_pool)
		label_setting = {'data_series_name': data_series_name}

		x = 10 ** (np.random.normal(2, 1, 50).clip(0.2, 3.8)) if kwargs['x_axis_scale'] == 'log' else (np.random.normal(2, 1, 50).clip(0.2, 3.8))
		y = 10 ** (np.random.normal(2, 1, 50).clip(0.2, 3.8)) if kwargs['y_axis_scale'] == 'log' else (np.random.normal(2, 1, 50).clip(0.2, 3.8))
		z = np.random.uniform(8, 16, 50)
		data = {'x': x, 'y': y, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	if kwargs['marker_face_color'] == 'different':
		label_setting['data_series_name'] = None
	return data, label_setting

def Bar_data_sampler(**kwargs):
	if kwargs['x_axis_scale'] == 'log' or kwargs['y_axis_scale'] == 'log':
		# For bar, x is pos and y is value, no matter of direction; so only y needs to scale up
		data_series_name_pool = ['Data Size', 'Power']
		data_series_name = np.random.choice(data_series_name_pool)
		label_setting = {'data_series_name': data_series_name}

		x = np.linspace(0, 10, 11)
		y = 10 ** (np.linspace(0, 5, 11) + np.random.uniform(0, 2, 11)).clip(0, 6)
		yerr = np.random.uniform(0.5, 0.75, 11) * y
		bottom = np.min(y + yerr) / 10
		data = {'x': x, 'y': y, 'z': yerr, 'bar_base': bottom, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	else:
		data_series_name_pool = ['Data', 'Score', 'Energy', 'Stock Price', 'User Ratings']
		data_series_name = np.random.choice(data_series_name_pool)
		label_setting = {'data_series_name': data_series_name}

		x = np.linspace(0, 10, 11)
		y = np.random.uniform(155, 185, 11) + np.random.normal(0, 15, 11)
		yerr = np.random.uniform(2, 4, 11)
		bottom = np.min(y + yerr) - 5
		data = {'x': x, 'y': y, 'z': yerr, 'bar_base': bottom, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
		
	return data, label_setting

def Matrix_display_data_sampler(**kwargs):
	label_setting = {}

	x_size_1 = np.random.randint(4, 8)
	x_size_2 = np.random.randint(8, 12)
	if kwargs['color_bar_orientation'] == 'vertical':
		x = np.random.normal(0, 1, (x_size_2, x_size_1)).clip(0, 2)
	else:
		x = np.random.normal(0, 1, (x_size_1, x_size_2)).clip(0, 2)
	data = {'x': x, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def Contour_data_sampler(**kwargs):
	label_setting = {}

	if kwargs['polarize']:
		x, y = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 10, 100))
		coef = 1 / np.random.uniform(1, 5, 4)
		z = ((np.sin(x) + 1)*coef[0] + (np.sin(y - x) + 1)*coef[1] + coef[3] * (np.cos(x) + coef[2]) / (np.sqrt(y) + 1))	# * np.random.uniform(0.8, 1, x.shape)

		data = {'x': x, 'y': y, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}
	else:
		_x, _y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
		coef = 1 / np.random.uniform(1, 5, 4)
		z = ((np.sin(_x) + 1)*coef[0] + (np.sin(_y) + 1)*coef[1] + np.sqrt(_x)*coef[2] + np.sqrt(_y)*coef[3] + 1)	# * np.random.uniform(0.8, 1, x.shape)

		x = 10 ** _x if kwargs['x_axis_scale'] == 'log' else _x
		y = 10 ** _y if kwargs['y_axis_scale'] == 'log' else _y
		data = {'x': x, 'y': y, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def Streamline_data_sampler(**kwargs):
	label_setting = {}

	x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100))
	coef = np.random.uniform(0.3, 0.7, 4)
	u = x*coef[0] + y*coef[1]
	v = x*coef[2] - y*coef[3]
	z = np.sqrt(u ** 2 + v ** 2)
	data = {'x': x, 'y': y, 'u': u, 'v': v, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def Surface_3d_data_sampler(**kwargs):
	label_setting = {}

	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = np.outer(np.cos(u), np.sin(v))
	y = np.outer(np.sin(u), np.sin(v))
	coef = np.random.uniform(0.3, 0.7, 3)
	z = np.outer(np.ones(np.size(u)), np.cos(v)) + (x * x / 2)*coef[0] - x*coef[1] - y*coef[2]
	data = {'x': x, 'y': y, 'z': z, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def Pie_data_sampler(**kwargs):
	label_setting = {}

	x_pool = [
		('Undergraduate\nstudents', 'Graduate\nstudents','Faculty', 'Staff', 'Others'),
		('Math', 'Physics', 'Chemistry', 'Biology', 'EECS', 'Others'),
		('Sleeping', 'Eating', 'Studying', 'Sports', 'Games', 'Thinking\nabout life', 'Others'),
		('Car', 'Bus', 'Bike', 'Walk', 'Skateboard', 'Scooter', 'Others')
	]
	x = np.random.choice(x_pool)
	y = [np.random.uniform(1, 10) for _ in range(len(x))]
	data = {'x': x, 'y': y, 'c_type': PLOT_TYPE_S2V[kwargs['plot_type']]}

	return data, label_setting

def sample_one(plot_type=None):
	line_color_pool = NORMAL_COLOR_POOL
	line_color_pool2 = NORMAL_COLOR_POOL + ['different'] * 7				# streamline
	line_style_pool = [None, 'solid', 'dashed', 'dashed dots', 'dotted']			# None stands for no line
	line_style_pool2 = ['solid', 'dashed', 'dashed dots', 'dotted']			# None stands for no line
	line_width_pool = ['thin', 'medium', 'thick']
	line_width_pool2 = ['thin', 'medium', 'thick', 'different']	# streamline
	marker_type_pool = [None, 'circles', 'down triangles', 'triangles', 'diamonds']	# None stands for no marker
	marker_type_pool2 = ['circles', 'down triangles', 'triangles', 'diamonds']	# scatter
	marker_size_pool = ['small', 'medium', 'large']
	marker_size_pool2 = ['small', 'medium', 'large', 'different']				# scatter
	marker_edge_width_pool = [None, 'thin', 'medium']								# None stands for no edge
	marker_edge_color_pool = NORMAL_COLOR_POOL
	marker_face_color_pool = NORMAL_COLOR_POOL
	marker_face_color_pool2 = NORMAL_COLOR_POOL + ['different'] * 7					# scatter
	marker_interval_pool = [1, 2, 5]

	show_error_bar_pool = [False] + [True]
	error_bar_cap_size_pool = ['small', 'medium', 'large']
	error_bar_cap_thickness_pool = ['thin', 'thick']
	error_bar_color_pool = NORMAL_COLOR_POOL

	font_size_pool = ['small', 'medium', 'large']
	invert_x_axis_pool = [False] + [True]
	invert_y_axis_pool = [False] + [True]
	invert_z_axis_pool = [False] + [True]

	grid_line_type_pool = [None] + ['horizontal', 'vertical', 'both']
	grid_line_color_pool = ['red', 'orange', 'green', 'blue', 'magenta'] + ['gray'] * 3 + ['black'] * 2
	grid_line_style_pool = ['solid', 'dashed', 'dashed dots', 'dotted']
	grid_line_width_pool = ['very thin', 'thin']

	x_axis_position_pool = ['bottom'] + ['top']
	y_axis_position_pool = ['left'] + ['right']
	x_axis_scale_pool = ['linear'] + ['log']
	y_axis_scale_pool = ['linear'] + ['log']
	polarize_pool = [False] * 4 + [True]

	number_of_bins_pool = [6, 8, 10, 12]
	bar_relative_width_pool = [0.6, 0.8, 1]
	bar_edge_width_pool = [None, 'thin', 'medium']
	bar_edge_color_pool = NORMAL_COLOR_POOL
	bar_face_color_pool = NORMAL_COLOR_POOL

	color_map_pool = ['transparent to solid red', 'transparent to solid blue', 'transparent to solid green', \
		'transparent yellow to solid red', 'transparent green to solid blue', 'transparent blue to dark purple', \
		'magenta to yellow', 'red to yellow', 'light cyan to magenta', \
		'purple to white to green', 'red to white to blue', 'red to yellow to green']
	color_bar_orientation_pool = ['vertical'] + ['horizontal']
	color_bar_length_pool = ['short'] + ['long'] * 2
	color_bar_thickness_pool = ['thin', 'medium', 'thick']

	bar_orientation_pool = ['vertical'] + ['horizontal']
	bar_width_pool = [0.6, 0.8, 1]
	bar_height_pool = [0.6, 0.8, 1]

	contour_plot_type_pool = ['lined', 'filled']
	number_of_levels_pool = [6, 10, 15]

	density_pool = ['loose', 'medium', 'dense']
	arrow_size_pool = ['small', 'large']
	arrow_style_pool = ['curve', 'solid']

	surface_color_pool = NORMAL_COLOR_POOL + ['different'] * 7

	explode_pool = [None, 'small', 'large']
	precision_digits_pool = [0, 1, 2, 3]
	percentage_distance_from_center_pool = ['near', 'medium', 'far']
	label_distance_from_center_pool = ['near', 'far']
	radius_pool = ['small', 'medium', 'large']
	section_edge_width_pool = [None, 'thin', 'medium']
	section_edge_color_pool = ['gray', 'black']

	## Total combinations ~= ?

	if plot_type is None:
		plot_type = np.random.choice(['line chart', 'histogram', 'scatter plot', 'bar plot', \
			'matrix display', 'contour plot', 'streamline plot', '3D surface', 'pie chart'])

	if plot_type == 'line chart':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'line chart'
			for kw in LINE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in LINE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in LINE_SPECIAL_KWS:
				pool = eval(kw + '_pool_line')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue
			
			if plotter_kwargs['marker_type'] is None:
				if plotter_kwargs['line_style'] is None:
					# Empty plot. Discard
					continue
				plotter_kwargs['marker_size'] = None
				plotter_kwargs['marker_edge_width'] = None
				plotter_kwargs['marker_edge_color'] = None
				plotter_kwargs['marker_face_color'] = None
				plotter_kwargs['marker_interval'] = None
				# plotter_kwargs['show_error_bar'] = None 	# Can have error bar w/o markers
			else:
				if (plotter_kwargs['marker_edge_width'] is not None) and (WIDTH_S2V[plotter_kwargs['marker_edge_width']] * 2 >= MARKER_SIZE_S2V[plotter_kwargs['marker_size']]):
					# Edge too thick, face not visible. Discard
					continue
				if plotter_kwargs['marker_edge_color'] == plotter_kwargs['marker_face_color']:
					# Edge and face are same color => force no edge
					plotter_kwargs['marker_edge_width'] = None
					plotter_kwargs['marker_edge_color'] = None
				if plotter_kwargs['marker_edge_width'] is None:
					# No edge => force markeredgecolor = None.
					plotter_kwargs['marker_edge_color'] = None
				# Otherwise markeredgecolor != None
			if plotter_kwargs['line_style'] is None:
				# Empty line, force linewidth/linecolor = None. Otherwise linewidth/linecolor != None
				plotter_kwargs['line_width'] = None
				plotter_kwargs['line_color'] = None

			if plotter_kwargs['line_color'] == plotter_kwargs['marker_face_color'] and plotter_kwargs['marker_edge_color'] is None:
				# Marker and line are the same color, hard to describe. Discard
				continue

			if plotter_kwargs['show_error_bar'] != True:
				plotter_kwargs['error_bar_cap_size'] = None
				plotter_kwargs['error_bar_cap_thickness'] = None
				plotter_kwargs['error_bar_color'] = None
			break
		data, label_setting = Line_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'histogram':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'histogram'
			for kw in HISTOGRAM_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in HISTOGRAM_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in HISTOGRAM_SPECIAL_KWS:
				pool = eval(kw + '_pool_histogram')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['bar_edge_color'] == plotter_kwargs['bar_face_color']:
				# Edge and face are same color => force no edge
				plotter_kwargs['bar_edge_width'] = None
				plotter_kwargs['bar_edge_color'] = None
			if plotter_kwargs['bar_edge_width'] is None:
				# No edge => force markeredgecolor = None.
				plotter_kwargs['bar_edge_color'] = None
			break
		data, label_setting = Histogram_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'scatter plot':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'scatter plot'
			for kw in SCATTER_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SCATTER_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SCATTER_SPECIAL_KWS:
				pool = eval(kw + '_pool_scatter')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue
			
			if plotter_kwargs['marker_face_color'] != 'different':
				# Pure color
				plotter_kwargs['color_map'] = None
				plotter_kwargs['color_bar_orientation'] = None
				plotter_kwargs['color_bar_length'] = None
				plotter_kwargs['color_bar_thickness'] = None
				if plotter_kwargs['marker_edge_color'] == plotter_kwargs['marker_face_color']:
					# Edge and face are same color => force no edge
					plotter_kwargs['marker_edge_width'] = None
					plotter_kwargs['marker_edge_color'] = None
			else:
				# Different color
				pass

			if plotter_kwargs['marker_edge_width'] is None:
				# No edge => force markeredgecolor = None.
				plotter_kwargs['marker_edge_color'] = None
				# Otherwise markeredgecolor != None
			if (plotter_kwargs['marker_edge_width'] is not None) and (plotter_kwargs['marker_size'] != 'different') \
					and (WIDTH_S2V[plotter_kwargs['marker_edge_width']] * 2 >= MARKER_SIZE_S2V[plotter_kwargs['marker_size']]):
				# Edge too thick, face not visible. Discard
				continue
			break
		data, label_setting = Scatter_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'bar plot':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'bar plot'
			for kw in BAR_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in BAR_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in BAR_SPECIAL_KWS:
				pool = eval(kw + '_pool_bar')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			if plotter_kwargs['bar_orientation'] == 'vertical':
				plotter_kwargs['x_axis_scale'] = None
				plotter_kwargs['bar_height'] = None
			else:	# horizontal
				plotter_kwargs['y_axis_scale'] = None
				plotter_kwargs['bar_width'] = None

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['show_error_bar'] != True:
				plotter_kwargs['error_bar_cap_size'] = None
				plotter_kwargs['error_bar_cap_thickness'] = None
				plotter_kwargs['error_bar_color'] = None

			if plotter_kwargs['bar_edge_color'] == plotter_kwargs['bar_face_color']:
				# Edge and face are same color => force no edge
				plotter_kwargs['bar_edge_width'] = None
				plotter_kwargs['bar_edge_color'] = None
			if plotter_kwargs['bar_edge_width'] == None:
				# No edge => force markeredgecolor = None.
				plotter_kwargs['bar_edge_color'] = None
			if plotter_kwargs['error_bar_color'] == plotter_kwargs['bar_face_color'] != None:
				# errorbar and bar same color, discard
				continue
			break
		data, label_setting = Bar_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'matrix display':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'matrix display'
			for kw in MATRIX_DISPLAY_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in MATRIX_DISPLAY_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in MATRIX_DISPLAY_SPECIAL_KWS:
				pool = eval(kw + '_pool_matrix_display')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue
			break
		data, label_setting = Matrix_display_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'contour plot':
		# contour_plot_type, color_map, number_of_levels, line_style, line_width, ...(shared)
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'contour plot'
			for kw in CONTOUR_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in CONTOUR_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in CONTOUR_SPECIAL_KWS:
				pool = eval(kw + '_pool_contour')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['polarize'] == True:
				# Polar, force 'filled', because lines can be incontinuous and weird in polar
				plotter_kwargs['contour_plot_type'] = 'filled'
			if plotter_kwargs['contour_plot_type'] == 'filled':
				plotter_kwargs['line_style'] = None
				plotter_kwargs['line_width'] = None
			break
		data, label_setting = Contour_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'streamline plot':
		# density, line_color, color_map, line_width, arrow_size, arrow_style
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'streamline plot'
			for kw in STREAMLINE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in STREAMLINE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in STREAMLINE_SPECIAL_KWS:
				pool = eval(kw + '_pool_streamline')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['line_color'] != 'different':
				# Pure color
				plotter_kwargs['color_map'] = None
				plotter_kwargs['color_bar_orientation'] = None
				plotter_kwargs['color_bar_length'] = None
				plotter_kwargs['color_bar_thickness'] = None
			break
		data, label_setting = Streamline_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == '3D surface':
		# surface_color, color_map
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = '3D surface'
			for kw in SURFACE_3D_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SURFACE_3D_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SURFACE_3D_SPECIAL_KWS:
				pool = eval(kw + '_pool_surface_3d')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			# Surface_3d shouldn't need these
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['surface_color'] != 'different':
				# Pure color
				plotter_kwargs['color_map'] = None
				plotter_kwargs['color_bar_orientation'] = None
				plotter_kwargs['color_bar_length'] = None
				plotter_kwargs['color_bar_thickness'] = None
			break
		data, label_setting = Surface_3d_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'pie chart':
		# explode, precision_digits, percentage_distance_from_center, label_distance_from_center, radius, section_edge_width, section_edge_color
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'pie chart'
			for kw in PIE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in PIE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in PIE_SPECIAL_KWS:
				pool = eval(kw + '_pool_pie')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			# Pie shouldn't need these
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=False)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=False)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['section_edge_width'] is None:
				plotter_kwargs['section_edge_color'] = None
			break

		data, label_setting = Pie_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	else:
		raise ValueError('plot_type = {}'.format(plot_type))

	return data, plotter_kwargs

def sample_one_hard(plot_type=None, polarize=None):
	line_color_pool = NORMAL_COLOR_POOL
	line_color_pool2 = ['different']
	line_style_pool = ['dashed', 'dashed dots', 'dotted']							# None stands for no line
	line_style_pool2 = ['dashed', 'dashed dots', 'dotted']							# None stands for no line
	line_width_pool = ['thin', 'medium', 'thick']
	line_width_pool2 = ['different']
	marker_type_pool = ['circles', 'down triangles', 'triangles', 'diamonds']		# None stands for no marker
	marker_type_pool2 = ['circles', 'down triangles', 'triangles', 'diamonds']
	marker_size_pool = ['small', 'medium', 'large']
	marker_size_pool2 = ['different']
	marker_edge_width_pool = ['thin', 'medium']										# None stands for no edge
	marker_edge_color_pool = NORMAL_COLOR_POOL
	marker_face_color_pool = NORMAL_COLOR_POOL
	# marker_face_color_pool2 = NORMAL_COLOR_POOL + ['different']
	marker_face_color_pool2 = ['different']
	marker_interval_pool = [2, 5]

	show_error_bar_pool = [True]
	error_bar_cap_size_pool = ['medium', 'large']
	error_bar_cap_thickness_pool = ['thick']
	error_bar_color_pool = NORMAL_COLOR_POOL

	font_size_pool = ['small', 'large']
	invert_x_axis_pool = [False] + [True] * 2
	invert_y_axis_pool = [False] + [True] * 2
	invert_z_axis_pool = [False] + [True] * 2

	grid_line_type_pool = ['horizontal', 'vertical', 'both']
	grid_line_color_pool = ['red', 'orange', 'green', 'blue', 'magenta'] + ['gray'] * 3 + ['black'] * 2
	grid_line_style_pool = ['dashed', 'dashed dots', 'dotted']
	grid_line_width_pool = ['very thin', 'thin']

	x_axis_position_pool = ['bottom'] + ['top'] * 2
	y_axis_position_pool = ['left'] + ['right'] * 2
	x_axis_scale_pool = ['linear'] + ['log'] * 2
	y_axis_scale_pool = ['linear'] + ['log'] * 2

	if polarize is None:
		polarize_pool = [False, True]
	elif polarize == True:
		polarize_pool = [True]
	else:
		polarize_pool = [False]					# Do polar whenever we can in tutorial; necessary and easier (fewer plots)

	number_of_bins_pool = [8, 10, 12]
	bar_relative_width_pool = [0.6, 0.8]
	bar_edge_width_pool = ['thin', 'medium']
	bar_edge_color_pool = NORMAL_COLOR_POOL
	bar_face_color_pool = NORMAL_COLOR_POOL

	color_map_pool = ['transparent to solid red', 'transparent to solid blue', 'transparent to solid green', \
		'transparent yellow to solid red', 'transparent green to solid blue', 'transparent blue to dark purple', \
		'magenta to yellow', 'red to yellow', 'light cyan to magenta', \
		'purple to white to green', 'red to white to blue', 'red to yellow to green']
	color_bar_orientation_pool = ['vertical'] + ['horizontal']
	color_bar_length_pool = ['short']
	color_bar_thickness_pool = ['medium', 'thick']

	bar_orientation_pool = ['vertical'] + ['horizontal'] * 2
	bar_width_pool = [0.6, 0.8]
	bar_height_pool = [0.6, 0.8]

	contour_plot_type_pool = ['lined']
	number_of_levels_pool = [10, 15]

	density_pool = ['loose', 'dense']
	arrow_size_pool = ['large']
	arrow_style_pool = ['solid']

	surface_color_pool = ['different']

	explode_pool = ['small', 'large']
	precision_digits_pool = [1, 2, 3]
	percentage_distance_from_center_pool = ['near', 'far']
	label_distance_from_center_pool = ['far']
	radius_pool = ['small', 'large']
	section_edge_width_pool = ['thin', 'medium']
	section_edge_color_pool = ['gray', 'black']

	## Total combinations ~= ?

	if plot_type is None:
		plot_type = np.random.choice(['line chart', 'histogram', 'scatter plot', 'bar plot', \
			'matrix display', 'contour plot', 'streamline plot', '3D surface', 'pie chart'])

	if plot_type == 'line chart':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'line chart'
			for kw in LINE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in LINE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in LINE_SPECIAL_KWS:
				pool = eval(kw + '_pool_line')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue
			
			if (plotter_kwargs['marker_edge_width'] is not None) and (WIDTH_S2V[plotter_kwargs['marker_edge_width']] * 2 >= MARKER_SIZE_S2V[plotter_kwargs['marker_size']]):
				# Edge too thick, face not visible. Discard
				continue
			if plotter_kwargs['marker_edge_color'] == plotter_kwargs['marker_face_color']:
				# Edge and face are same color => force no edge (discard)
				continue

			if plotter_kwargs['line_color'] == plotter_kwargs['marker_face_color'] and plotter_kwargs['marker_edge_color'] is None:
				# Marker and line are the same color, hard to describe. Discard
				continue

			if plotter_kwargs['show_error_bar'] != True:
				plotter_kwargs['error_bar_cap_size'] = None
				plotter_kwargs['error_bar_cap_thickness'] = None
				plotter_kwargs['error_bar_color'] = None
			break
		data, label_setting = Line_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'histogram':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'histogram'
			for kw in HISTOGRAM_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in HISTOGRAM_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in HISTOGRAM_SPECIAL_KWS:
				pool = eval(kw + '_pool_histogram')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['bar_edge_color'] == plotter_kwargs['bar_face_color']:
				# Edge and face are same color => force no edge (discard)
				continue
			break
		data, label_setting = Histogram_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'scatter plot':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'scatter plot'
			for kw in SCATTER_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SCATTER_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SCATTER_SPECIAL_KWS:
				pool = eval(kw + '_pool_scatter')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue
			
			if plotter_kwargs['marker_face_color'] != 'different':
				# Pure color, discard (shouldn't trigger)
				continue

			if plotter_kwargs['marker_edge_width'] is None:
				# No edge => discard (shouldn't trigger)
				continue
			if (plotter_kwargs['marker_edge_width'] is not None) and (plotter_kwargs['marker_size'] != 'different') \
					and (WIDTH_S2V[plotter_kwargs['marker_edge_width']] * 2 >= MARKER_SIZE_S2V[plotter_kwargs['marker_size']]):
				# Edge too thick, face not visible. Discard
				continue
			break
		data, label_setting = Scatter_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'bar plot':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'bar plot'
			for kw in BAR_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in BAR_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in BAR_SPECIAL_KWS:
				pool = eval(kw + '_pool_bar')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			if plotter_kwargs['bar_orientation'] == 'vertical':
				plotter_kwargs['x_axis_scale'] = None
				plotter_kwargs['bar_height'] = None
			else:	# horizontal
				assert plotter_kwargs['bar_orientation'] == 'horizontal', 'Invalid sample: bar_orientation == {}'.format(plotter_kwargs['bar_orientation'])
				plotter_kwargs['y_axis_scale'] = None
				plotter_kwargs['bar_width'] = None

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['bar_edge_color'] == plotter_kwargs['bar_face_color']:
				# Edge and face are same color => force no edge (discard)
				continue
			if plotter_kwargs['error_bar_color'] == plotter_kwargs['bar_face_color'] != None:
				# errorbar and bar same color, discard
				continue
			break
		data, label_setting = Bar_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'matrix display':
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'matrix display'
			for kw in MATRIX_DISPLAY_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in MATRIX_DISPLAY_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in MATRIX_DISPLAY_SPECIAL_KWS:
				pool = eval(kw + '_pool_matrix_display')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue
			break
		data, label_setting = Matrix_display_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'contour plot':
		# contour_plot_type, color_map, number_of_levels, line_style, line_width, ...(shared)
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'contour plot'
			for kw in CONTOUR_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in CONTOUR_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in CONTOUR_SPECIAL_KWS:
				pool = eval(kw + '_pool_contour')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['polarize'] == True:
				# Polar, force 'filled', because lines can be incontinuous and weird in polar
				plotter_kwargs['contour_plot_type'] = 'filled'
			if plotter_kwargs['contour_plot_type'] == 'filled':
				plotter_kwargs['line_style'] = None
				plotter_kwargs['line_width'] = None
			break
		data, label_setting = Contour_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'streamline plot':
		# density, line_color, color_map, line_width, arrow_size, arrow_style
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'streamline plot'
			for kw in STREAMLINE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in STREAMLINE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in STREAMLINE_SPECIAL_KWS:
				pool = eval(kw + '_pool_streamline')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['line_color'] != 'different':
				# Pure color, discard (shouldn't trigger)
				continue
			# 	plotter_kwargs['color_map'] = None
			# 	plotter_kwargs['color_bar_orientation'] = None
			# 	plotter_kwargs['color_bar_length'] = None
			# 	plotter_kwargs['color_bar_thickness'] = None
			break
		data, label_setting = Streamline_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == '3D surface':
		# surface_color, color_map
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = '3D surface'
			for kw in SURFACE_3D_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SURFACE_3D_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in SURFACE_3D_SPECIAL_KWS:
				pool = eval(kw + '_pool_surface_3d')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			# Surface_3d shouldn't need these
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['surface_color'] != 'different':
				# Pure color, discard (shouldn't trigger)
				continue
				# plotter_kwargs['color_map'] = None
				# plotter_kwargs['color_bar_orientation'] = None
				# plotter_kwargs['color_bar_length'] = None
				# plotter_kwargs['color_bar_thickness'] = None
			break
		data, label_setting = Surface_3d_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	elif plot_type == 'pie chart':
		# explode, precision_digits, percentage_distance_from_center, label_distance_from_center, radius, section_edge_width, section_edge_color
		while True:
			plotter_kwargs = dict(DEFAULT_DICT)
			plotter_kwargs['plot_type'] = 'pie chart'
			for kw in PIE_COMMON_KWS:
				pool = eval(kw + '_pool')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in PIE_COMMON2_KWS:
				pool = eval(kw + '_pool2')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val
			for kw in PIE_SPECIAL_KWS:
				pool = eval(kw + '_pool_pie')
				kw_val = np.random.choice(pool)
				plotter_kwargs[kw] = kw_val

			plotter_kwargs = plotter_kwargs_type_uniformize(plotter_kwargs)
			# Pie shouldn't need these
			plotter_kwargs = plotter_kwargs_validate_polarize_axis(plotter_kwargs, hard=True)
			plotter_kwargs = plotter_kwargs_validate_grid_line(plotter_kwargs, hard=True)
			if plotter_kwargs is None:
				continue

			if plotter_kwargs['section_edge_width'] is None:
				# shouldn't trigger
				continue
			break

		data, label_setting = Pie_data_sampler(**plotter_kwargs)
		plotter_kwargs.update(label_setting)

	else:
		raise ValueError('plot_type = {}'.format(plot_type))

	return data, plotter_kwargs

def sample(N, start_id=1, sampler_func=sample_one):
	# N for each type
	data_series_dir = 'data/sample-plots/data-series'
	params_dir = 'data/sample-plots/params'
	plots_dir = 'data/sample-plots/plots'
	os.makedirs(data_series_dir, exist_ok=True)
	os.makedirs(params_dir, exist_ok=True)
	os.makedirs(plots_dir, exist_ok=True)

	plot_types = [
		'line',
		'histogram',
		'scatter',
		'bar',
		'matrix_display',
		'contour',
		'streamline',
		'surface_3d',
		'pie'
	]

	for plot_type in plot_types:
		pbar = tqdm(iterable=range(start_id, start_id + N), desc=plot_type)
		for i in pbar:
			data, plotter_kwargs = sampler_func(PLOT_TYPE_V2S[plot_type])

			out_data_fname = os.path.join(data_series_dir, '{}.{}.dataseries.pkl'.format(plot_type, i))
			out_params_fname = os.path.join(params_dir, '{}.{}.params.json'.format(plot_type, i))
			out_plot_fname = os.path.join(plots_dir, '{}.{}.plot.png'.format(plot_type, i))

			pickle.dump(data, open(out_data_fname, 'wb'))

			plotter_kwargs_ordered = OrderedDict(sorted(plotter_kwargs.items()))
			json.dump(plotter_kwargs_ordered, open(out_params_fname, 'w'), indent=4)
			
			plotter_kwargs_unnat = plotter_kwargs_unnaturalize(**plotter_kwargs)
			fig = plotter(**data, **plotter_kwargs_unnat)
			fig.savefig(out_plot_fname, dpi=100)
			plt.close(fig)
		pbar.close()

def sample_show(N, sampler_func=sample_one):
	for i in range(N):
		data, plotter_kwargs = sampler_func('pie chart')
		print(plotter_kwargs)
		plotter_kwargs_unnat = plotter_kwargs_unnaturalize(**plotter_kwargs)
		fig = plotter(**data, **plotter_kwargs_unnat)
		plt.show()
		plt.close(fig)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--action', dest='action', type=str, choices=['show', 'save'], default='save',
		help='show/save the sampled plots. If "save", the sampled plots will go to "data/sample-plots".')
	parser.add_argument('--n', dest='n', type=int, default=10,
		help='how many plots to sample per type (there are 9 types, therefore 9n plots in total.')
	parser.add_argument('--hard', dest='hard', action='store_true',
		help='samples hard plots, i.e. more slots (components) are activated.')
	args = parser.parse_args()

	sampler_func = sample_one_hard if args.hard else sample_one

	if args.action == 'show':
		sample_show(args.n, sampler_func=sampler_func)
	else:
		sample(args.n, sampler_func=sampler_func)


