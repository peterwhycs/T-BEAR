import datetime
import itertools
import subprocess
import sys
import time
from datetime import date, timedelta
from itertools import chain, product

import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

init_notebook_mode(connected=True)
one_hour = datetime.timedelta(hours=1)
one_day = datetime.timedelta(days=1)


def overview_figs(df_, df_IF, graph_cols, counter_cols):
	plot_figures = {}
	for feature in graph_cols:
		anom_x = df_IF['sourcetime']
		anom_y = df_IF[feature]

		x_data = df_['sourcetime']
		y_data = df_[feature]

		trace0 = Scattergl(x=x_data, y=y_data, mode='lines', name=feature.capitalize())
		trace_anom = Scattergl(x=anom_x, y=anom_y, mode='markers', name="Anomaly",
							   marker=dict(size=5, color='rgba(152, 0, 0, .9)'))
		if feature in counter_cols:
			feature_ct = feature + ' (counter)'
		else:
			feature_ct = feature
		layout = dict(title=feature_ct.capitalize() + ' vs. ' + 'UTC',
					  xaxis=dict(title='UTC', showline=True, showgrid=True, showticklabels=True,
								 linecolor='rgb(0, 0, 0)', linewidth=1, autotick=True, nticks='auto', ticks='outside',
								 tickcolor='rgb(0, 0, 0)', tickwidth=2, ticklen=8,
								 titlefont=dict(family='Open Sans', size=13, color='rgb(0, 0, 0)'), ),
					  yaxis=dict(title=feature_ct.capitalize(), autorange=True, showgrid=True, zeroline=True,
								 showline=True,
								 showticklabels=True, linewidth=1,
								 titlefont=dict(family='Open Sans', size=13, color='rgb(0, 0, 0)'), ), autosize=True,
					  margin=dict(autoexpand=True), showlegend=True)
		fig = dict(data=[trace0, trace_anom], layout=layout)
		plot_figures[feature] = fig
	return plot_figures


def plot_overview_graph(figure):
	selected_fig = plot_figures[figure]
	return iplot(selected_fig)


def split_time_indices(time_indices):
	h, i, time_groups = 0, 0, []
	while i < len(time_indices):
		if i != 0:
			prev, curr = curr, time_indices[i]
			if ((abs(curr - prev) >= one_hour) or (i == len(time_indices) - 1)):
				time_groups.append(time_indices[h:i])
				h, i = i, i + 1
		else:
			curr = time_indices[i]
		i += 1
	return time_groups


def set_time_dict(time_groups):
	time_dict = {}
	for group in time_groups:
		t1, t2 = group[0], group[-1]
		t1_t2_str = t1.strftime("%Y-%m-%d %H:%M:%S") + ' to ' + t2.strftime("%Y-%m-%d %H:%M:%S")
		time_dict[t1_t2_str] = group
	return time_dict


def plot_index_graph(feature, time_dict, ext_min):
	column, anom_timestamps, m = feature, list(time_dict), ext_min
	t1, t2 = anom_timestamps[0] - (m * one_min), anom_timestamps[-1] + (m * one_min)
	u, v = t1 - (one_min * len(anom_timestamps)), t2 + (one_min * len(anom_timestamps))
	if (u < df_['sourcetime'].iloc[0]):
		u = (df_['sourcetime'].iloc[0])
	if v > df_['sourcetime'].iloc[-1]:
		v = df_['sourcetime'].iloc[-1]

	df_anom = df_IF.loc[df_IF['sourcetime'].isin(anom_timestamps)]
	x_anom, y_anom = df_anom['sourcetime'], df_anom[feature]

	mask1, mask2 = (df_['sourcetime'] >= u), (df_['sourcetime'] <= v)
	df_data = df_.loc[mask1 & mask2]
	x_data, y_data = df_data['sourcetime'], df_data[feature]

	trace0 = Scattergl(x=x_data, y=y_data, mode='lines', name=feature.capitalize())
	trace_anom = Scattergl(x=x_anom, y=y_anom, mode='markers', name="Anomaly",
						   marker=dict(size=5, color='rgba(152, 0, 0, .9)'))
	if feature in counter_cols:
		feature_ct = feature + ' (counter)'
	else:
		feature_ct = feature
	layout = dict(title=feature_ct.capitalize() + ' vs. ' + 'UTC',
				  xaxis=dict(title='UTC', showline=True, showgrid=True, showticklabels=True,
							 linecolor='rgb(0, 0, 0)', linewidth=1, autotick=True, nticks='auto', ticks='outside',
							 tickcolor='rgb(0, 0, 0)', tickwidth=2, ticklen=8,
							 titlefont=dict(family='Open Sans', size=13, color='rgb(0, 0, 0)'), ),
				  yaxis=dict(title=feature_ct.capitalize(), autorange=True, showgrid=True, zeroline=True,
							 showline=True,
							 showticklabels=True, linewidth=1,
							 titlefont=dict(family='Open Sans', size=13, color='rgb(0, 0, 0)'), ), autosize=True,
				  margin=dict(autoexpand=True), showlegend=True)
	fig = dict(data=[trace0, trace_anom], layout=layout)
	plot_figures[feature] = fig
	return iplot(fig)
