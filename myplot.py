import matplotlib  # TODO :needed?
import numpy as np # import needed?


# This stuff needs to be set before we import matplotlib.pyplot
matplotlib.use('PDF')  # save plots as PDF
font = {
    'size': 20,
    'family': 'Myriad Pro',
}
#  'serif': 'Times New Roman',
#  'family': 'serif'}
matplotlib.rc('font', **font)

# use type 1 fonts
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True

# use TrueType fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import os
import numpy
from scipy import stats
import textwrap
from matplotlib.colors import Normalize

#plt.xkcd()


##
## STYLES
##

default_style = {
    'colors': ('b', 'g', 'r', 'c', 'm', 'y'),
    'linestyles': ('-', '--', '-.', ':'),
    'marker_edgecolor': 'black',
    'markerstyles': ('o', 'v', '^', 'D', 's', '<', '>', 'h', '8'), # more available
    'hatchstyles': (None, '/', '\\', 'o', '*', '+', '//', '\\\\', '-', 'x', 'O', '.'),
    'textsize_delta': 0,
    'gridalpha':1.0,
    'frame_lines':{'top':True, 'right':True, 'bottom':True, 'left':True},
    'tick_marks':{'top':True, 'right':True, 'bottom':True, 'left':True},
    'bar_edgecolor':'black',
    'errorbar_style':'line',
    'transparent_bg':False,
    'foreground_color':'black',
    'ylabel_rotation':'vertical',
    'ylabel_textwrap_width': 80,
    'ylabel_pad': 0,  # TODO: check this
    'xlabel_font_weight': 'normal',
    'ylabel_font_weight': 'normal',
}


#http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
#http://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.) 

# remove every other color (light version)
tableau10light = [tableau20[i] for i in range(len(tableau20)) if i % 2 == 1]
tableau10 = [tableau20[i] for i in range(len(tableau20)) if i % 2 == 0]


pretty_style = {
    'colors': tableau10,
    'linestyles': ('-', '--', '-.', ':'),
    'markerstyles': ('o', 'v', '^', 'D', 's', '<', '>', 'h', '8'), # more available
    'marker_edgecolor': 'white',
    'hatchstyles': (None, '////', '\\\\\\\\', 'o', '+', '*', '//', '\\\\', '-', 'x', 'O', '.'),
    'textsize_delta': 4,
    'gridalpha':0.3,
    'frame_lines':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'tick_marks':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'bar_edgecolor':'white',
    'errorbar_style':'line',  # 'line' or 'fill'
    'transparent_bg':False,
    'foreground_color':'black',
    'ylabel_rotation':'vertical',
    'ylabel_textwrap_width': 80,
    'ylabel_pad': 20,  # TODO: check this
    'xlabel_font_weight': 'bold',
    'ylabel_font_weight': 'bold',
}

# TODO: text delta 0, set override in mctls plots


# re-arrange tableau10 light for dark bg (brighter colors first
tableau10light2 = [tableau10light[i] for i in (1, 2, 3, 9, 6, 8, 0, 5, 7, 4)]

dark_bg_style = {
    'colors': tableau10light2,
    'linestyles': ('-', '--', '-.', ':'),
    'markerstyles': ('o', 'v', '^', 'D', 's', '<', '>', 'h', '8'), # more available
    'marker_edgecolor': 'None',
    'hatchstyles': (None, '////', '\\\\\\\\', 'o', '+', '*', '//', '\\\\', '-', 'x', 'O', '.'),
    'textsize_delta': 4,
    'gridalpha':0.5,
    'frame_lines':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'tick_marks':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'bar_edgecolor':'white',
    'errorbar_style':'line',  # 'line' or 'fill'
    'transparent_bg':True,
    'foreground_color':'white',
    'ylabel_rotation':'horizontal',
    'ylabel_textwrap_width': 80,
    'ylabel_pad': 70,
    'xlabel_font_weight': 'bold',
    'ylabel_font_weight': 'bold',
}

slide_style = {
    'colors': tableau10,
    'linestyles': ('-', '--', '-.', ':'),
    'markerstyles': ('o', 'v', '^', 'D', 's', '<', '>', 'h', '8'), # more available
    'marker_edgecolor': 'white',
    'hatchstyles': (None, '////', '\\\\\\\\', 'o', '+', '*', '//', '\\\\', '-', 'x', 'O', '.'),
    'textsize_delta': 4,
    'gridalpha':0.3,
    'frame_lines':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'tick_marks':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'bar_edgecolor':'white',
    'errorbar_style':'line',  # 'line' or 'fill'
    'transparent_bg':False,
    'foreground_color':'black',
    'ylabel_rotation':'horizontal',
    'ylabel_textwrap_width': 8,
    'ylabel_pad': 50,
    'xlabel_font_weight': 'bold',
    'ylabel_font_weight': 'bold',

    # overriede kw args to plot function
    'arg_override': {
        'legend_loc': 'out right',
        #'legend_cols': 5,
        'labelspacing': 0.3,
        'guide_lines': [],  # TODO remove?
        'ylim': (0, None),
    },
}
    




##
## HELPER FUNCTIONS
##

# for square legend labels
class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = xdescent + (width - height), ydescent
        p = mpatches.Rectangle(xy=center, width=height,
                               height=height, angle=0.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def cdf_vals_from_data(data, numbins=None, maxbins=None):

    # make sure data is a numpy array
    data = numpy.array(data)
    
    # by default, use numbins equal to number of distinct values
    # TODO: shouldn't this be one per possible x val?
    if numbins == None:
        numbins = numpy.unique(data).size

    if maxbins != None and numbins > maxbins:
        numbins = maxbins
    
    # bin the data and count fraction of points in each bin (for PDF)
    rel_bin_counts, min_bin_x, bin_size, _ =\
        stats.relfreq(data, numbins, (data.min(), data.max()))
    
    # bin the data and count each bin (cumulatively) (for CDF)
    cum_bin_counts, min_bin_x, bin_size, _ =\
        stats.cumfreq(data, numbins, (data.min(), data.max()))

    # normalize bin counts so rightmost count is 1
    cum_bin_counts /= cum_bin_counts.max()

    # make array of x-vals (lower end of each bin)
    x_vals = numpy.linspace(min_bin_x, min_bin_x+bin_size*numbins, numbins)

    # CDF always starts at y=0
    cum_bin_counts = numpy.insert(cum_bin_counts, 0, 0)  # y = 0
    cdf_x_vals = numpy.insert(x_vals, 0, x_vals[0])  # x = min x


    return cum_bin_counts, cdf_x_vals, rel_bin_counts, x_vals

def endpoints_for_stretched_line(endpoints, xlim, ylim):
    e1, e2 = endpoints
    xmin, xmax = xlim
    ymin, ymax = ylim

    if e1[0] == e2[0]:  #vertical line
        return ((e1[0], ymin), (e1[0], ymax))
    else:
        m = float(e2[1]-e1[1])/float(e2[0]-e1[0])  # slope

        y_for_xmin = m*(xmin-e1[0]) + e1[1]
        x_for_ymin = ((ymin-e1[1])/m if m != 0 else 0) + e1[0]
        y_for_xmax = m*(xmax-e1[0]) + e1[1]
        x_for_ymax = ((ymax-e1[1])/m if m != 0 else 0) + e1[0]

        if y_for_xmin < ymin:
            new_e1 = (x_for_ymin, ymin)
        else:
            new_e1 = (xmin, y_for_xmin)

        if y_for_xmax > ymax:
            new_e2 = (x_for_ymax, ymax)
        else:
            new_e2 = (xmax, y_for_xmax)

        return (new_e1, new_e2)

def new_line(endpoints, line_width=1, color='gray', alpha=0.7, label=None,\
        stretch=True):
    return {
        'endpoints':((endpoints[0][0], endpoints[0][1]),
                     (endpoints[1][0], endpoints[1][1])),
        'stretch':stretch,
        'line_args':{
            'linewidth':line_width,
            'color':color,
            'alpha':alpha,
        },

        'label':label,
        'label_args':{
            'color':'gray',
            'alpha':0.7,
            'size':'small',
        },
    }

def confidence_interval_mean(values, degree=0.95):
    return stats.t.interval(0.95, len(values)-1, loc=np.mean(values),\
            scale=stats.sem(values))

def yerr_for_confidence_interval_mean(values, degree=0.95):
    ci = confidence_interval_mean(values, degree)
    mean = numpy.mean(values)
    return ci[1] - mean


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

# for heatmaps where some values are positive and some are negative.
# lets you specify the middle color to be 0
# http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...

        # make sure the absolute values of the extremes match so color
        # intensities are comparable
        # TODO: don't need to do this every time.
        if self.vmin < 0 and self.vmax > 0:
            extreme = max(abs(self.vmin), abs(self.vmax))
            self.vmin = -extreme
            self.vmax = extreme
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def subplots(num_row, num_col, width_scale=1, height_scale=1):
    # TODO: anything helpful we could do here?
    fig, ax_array = plt.subplots(num_row, num_col)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width*width_scale, height*height_scale)
    return fig, ax_array

def save_plot(filename):
    #plt.tight_layout()
    plt.savefig(filename)


##
## PLOT
##
def plot(xs, ys, filename='figure.pdf', builds=[], style=pretty_style, **kwargs):
    if 'arg_override' in style:
        for key, val in style['arg_override'].iteritems():
            kwargs[key] = val

    # save one copy of the fig for each "build" (set of series); e.g., for slides
    for build in builds:
        skip_series = [i for i in range(len(xs)) if i not in build]
        root, ext = os.path.splitext(filename)
        build_filename = '%s%s%s' % (root, build, ext)
        _plot(xs, ys, skip_series=skip_series, style=style,\
            filename=build_filename, **kwargs)

    return _plot(xs, ys, style=style, filename=filename, **kwargs)

def _plot(xs, ys, labels=None, xlabel=None, ylabel=None, title=None,
         type='series', filename=None, yerrs=None,

         # STYLE
         show_markers=True,
         markerstyles=None,
         linestyles=None, 
         hatchstyles=None, 
         linewidths=None, 
         colors=None,           # array of color names, RGB tuples, or integers
         grid='both',           # 'x', 'y', or 'both'
         style=pretty_style,    # dict of style options
         style_override={},     # override specific entries in style dict

         # DATA TRANSFORMATIONS
         xval_transform=lambda x: x,  # transform x values (e.g., divide by 1000)
         yval_transform=lambda y: y,  # transform y values (e.g., divide by 1000)

         # TICK MARKS
         xticks=None,
         xtick_labels=None,
         xtick_label_rotation=0,
         xtick_label_horizontal_alignment='center',
         xtick_frequency=1,  # 1=print every, 2=print every other, etc.
         xtick_label_transform=lambda x: x,  # transform tick label text
         show_x_tick_labels=True,
         show_y_tick_labels=True, 
         master_xticks=None,
         power_limits=(-3, 4),  # scientific notation for nums smaller than 10^-3 or bigger than 10^4
         
         # LEGEND
         legend_loc='best',         # location: 'best', 'upper right', etc.
         legend_bbox=None,
         show_legend=True,
         legend_border=False,
         legend_cols=1, 
         legend_col_spacing=None,
         labelspacing=0.2,      # vertical spacing between labels?
         handletextpad=0.5,     # horizontal space b/w square and label?
         pattern_legend_loc='best',         # location: 'best', 'upper right', etc.
         pattern_legend_bbox=None,
         
         # TEXT SIZE
         xlabel_size=20,
         ylabel_size=20, 
         ticklabel_size=20,
         legend_text_size=20,

         # SIZE & SCALES
         width_scale=1,         # stretch width of plot
         height_scale=1,        # stretch height of plot
         xscale=None,           # 'linear', 'log', or 'symlog'  (None->'linear')
         yscale=None,           # 'linear', 'log', or 'symlog'  (None->'linear')
         xlim=None,             # (min, max)  min & max can be individually set to None for default
         ylim=None,             # (min, max)  min & max can be individually set to None for default
         # TODO : add min or max options

         # OTHER
         bins=10,               # number of bins for histogram
         guide_lines=[],        # list of dicts specifying guide lines  # TODO example
         skip_series=[],        # list of series nums to skip (e.g., to make multi-stage builds)

         # ADDITIONAL Y AXES
         axis_assignments=None, # array; which Y axis should series i be plotted on?
         additional_ylabels=None,
         additional_ylims=None,
         additional_yscales=None,

         # BAR OPTIONS
         label_bars=False,      # set to True to label bars with their Y value
         bar_width=1,
         bar_group_padding=1,   # horizontal padding b/w groups of bars
         stackbar_pattern_labels=None,     # for a second legend
         stackbar_colors_denote='series',  # 'series' or 'segments'

         # PLOT OBJECTS
         axis=None,             # ???
         fig=None,              # pass in existing figure (e.g., for subplots)
         ax=None,               # pass in existing axes (e.g., for subplots)

         **kwargs):
     # TODO: split series and hist into two different functions?
     # TODO: clean up multiple axis stuff 



    #################### CHECK INPUT ####################
    for kw in ('legend', 'marker'):  # deprecated keywords
        if kw in kwargs:
            print '[WARNING]  Deprecated keyword: %s' % kw
    
    if not hasattr(xs[0], '__iter__') or not hasattr(ys[0], '__iter__'):
        print '[WARNING]  xs and ys should be arrays of arrays'

    if len(xs) != len(ys):
        print '[WARNING]  length of xs and ys do not match'
    

    for key, val in style_override.iteritems():
        style[key] = val
    
    
    #################### TRANSFORM DATA ####################
    for i in range(len(xs)):
        xs[i] = numpy.vectorize(xval_transform)(xs[i])

    for i in range(len(ys)):
        ys[i] = numpy.vectorize(yval_transform)(ys[i])
        if yerrs:
            yerrs[i] = numpy.vectorize(yval_transform)(yerrs[i])


    #################### PLOT OBJECTS ####################
    # if we want to do subplots, caller may have passed in an existing figure
    if not fig or not ax:
        fig, ax = plt.subplots()
        width, height = fig.get_size_inches()
        fig.set_size_inches(width*width_scale, height*height_scale)


    #################### SETUP ####################
    if xlabel: ax.set_xlabel(xlabel, color=style['foreground_color'],
        fontsize=xlabel_size + style['textsize_delta'],
        fontweight=style['xlabel_font_weight'])
    if ylabel: ax.set_ylabel(textwrap.fill(ylabel, style['ylabel_textwrap_width']),
        color=style['foreground_color'],
        fontsize=ylabel_size + style['textsize_delta'],
        fontweight=style['ylabel_font_weight'],
        va='center', labelpad=style['ylabel_pad'],
        rotation=style['ylabel_rotation'])
    if not show_x_tick_labels: ax.set_xticklabels([])
    if not show_y_tick_labels: ax.set_yticklabels([])
    if title: ax.set_title(title, color=style['foreground_color'])
    if axis: 
        ax.set_xlim(axis[0:2])
        ax.set_ylim(axis[2:4])
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if isinstance(ax.xaxis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
        ax.xaxis.get_major_formatter().set_powerlimits(power_limits)
    if isinstance(ax.yaxis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
        ax.yaxis.get_major_formatter().set_powerlimits(power_limits)
    lines = [None]*len(ys)
    show_legend = show_legend and labels != None
    if not labels: labels = ['']*len(ys)
    if not linewidths: linewidths = [4]*len(ys)
    if not axis_assignments: axis_assignments = [0]*len(ys)

    if type == 'stackbar' and stackbar_colors_denote == 'segments'\
            and stackbar_pattern_labels != None:
        temp = labels
        labels = list(reversed(stackbar_pattern_labels))
        stackbar_pattern_labels = temp
            
            
    # If X axis points are strings, make a dummy x array for each string x list.
    # not each series might have a data point for each X value, so we need to 
    # make a "master" xtick label list with all of the X values in right order
    # NOTE: for now, this assumes that either all series have numeric X axes or
    # none do.
    master_xnums = None
    if type not in ('stackplot', 'bar', 'barh', 'stackbar'):
        try:
            float(xs[0][0])
        except ValueError:
            # if user didn't pass in a sorted list of xtick labels,
            # make & sort list of all X tick labels used by any series
            if master_xticks == None:
                master_xticks = set()
                for i in range(len(xs)):
                    master_xticks |= set(xs[i])
                master_xticks = sorted(list(master_xticks))
            master_xnums = np.arange(len(master_xticks))

            # replace each old string with its index in master_xticks
            for i in range(len(xs)):
                new_x = []
                for val in xs[i]:
                    new_x.append(master_xticks.index(val))
                xs[i] = new_x



    #################### STYLE ####################

    if not colors:
        colors = []
        if type=='stackbar' and stackbar_colors_denote=='segments':
            colors_needed = len(ys[0])
        else:
            colors_needed = len(ys)
        for i in range(colors_needed):
            colors.append(style['colors'][i%len(style['colors'])])
    else:
        for i in range(len(colors)):
            if isinstance(colors[i], int):
                colors[i] = style['colors'][colors[i]%len(style['colors'])]

    if not linestyles:
        linestyles = []
        for i in range(len(ys)):
            linestyles.append(style['linestyles'][i%len(style['linestyles'])])
    else:
        for i in range(len(linestyles)):
            if isinstance(linestyles[i], int):
                linestyles[i] = style['linestyles'][linestyles[i]%len(style['linestyles'])]
    
    if not markerstyles:
        markerstyles = []
        for i in range(len(ys)):
            markerstyles.append(style['markerstyles'][i%len(style['markerstyles'])])
    else:
        for i in range(len(markerstyles)):
            if isinstance(markerstyles[i], int):
                markerstyles[i] = style['markerstyles'][markerstyles[i]]

    if not hatchstyles:
        hatchstyles = []
        if type=='stackbar' and stackbar_colors_denote=='segments':
            hatchstyles_needed = len(ys)
        else:
            hatchstyles_needed = len(ys[0])
        for i in range(hatchstyles_needed):
            hatchstyles.append(style['hatchstyles'][i%len(style['hatchstyles'])])
    else:
        for i in range(len(hatchstyles)):
            if isinstance(hatchstyles[i], int):
                hatchstyles[i] = style['hatchstyles'][hatchstyles[i]]

    # frame lines
    for side, visible in style['frame_lines'].iteritems():
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(style['foreground_color'])

    # tick marks  (spacing and labels set below)
    ax.tick_params(labelsize=ticklabel_size + style['textsize_delta'],\
        color=style['foreground_color'], **style['tick_marks'])
    
    # show grid lines?
    if grid:
        grid_args = {'zorder':0, 'color':style['foreground_color'], 'alpha':style['gridalpha']}
        if grid == 'x':
            ax.xaxis.grid(**grid_args)
        elif grid == 'y':
            ax.yaxis.grid(**grid_args)
        else:
            plt.grid(**grid_args)



    #################### PLOT ####################
    if type == 'series':
        for i in range(len(ys)):
            if axis_assignments[i] != 0: continue

            marker = markerstyles[i] if show_markers else None
            alpha = 0 if i in skip_series else 1


            # TODO: simplify these two cases? repetitive.
            if yerrs and style['errorbar_style'] == 'line':
                line = ax.errorbar(xs[i], ys[i], linestyle=linestyles[i], marker=marker,\
                    markeredgecolor=style['marker_edgecolor'],\
                    linewidth=linewidths[i], color=colors[i], label=labels[i],\
                    elinewidth=3, alpha=alpha,\
                    yerr=yerrs[i], **kwargs)

                # don't draw error bars in legend
                line = line[0]
            else:
                line, = ax.plot(xs[i], ys[i], linestyle=linestyles[i], marker=marker,\
                    markeredgecolor=style['marker_edgecolor'], zorder=3,\
                    alpha=alpha,\
                    linewidth=linewidths[i], color=colors[i], label=labels[i], **kwargs)

            lines[i] = line

            if yerrs and style['errorbar_style'] == 'fill':
                yerr_upper = numpy.array(ys[i])+numpy.array(yerrs[i])
                yerr_lower = numpy.array(ys[i])-numpy.array(yerrs[i])
                ax.fill_between(xs[i], yerr_lower, yerr_upper, color=colors[i], alpha=0.5)

    elif type == 'bar' or type == 'stackbar':
        if type == 'bar':
            num_groups = max([len(series) for series in ys])  # num clusters of bars
            num_series = len(ys)   # num bars in each cluster
        elif type == 'stackbar':
            num_groups = max([len(series) for x in ys for series in x])  # num clusters of bars
            num_series = len(ys)   # num bars in each cluster

        group_width = bar_width * num_series
        ind = np.arange(bar_group_padding/2.0,\
            num_groups*(bar_width*num_series+bar_group_padding) + bar_group_padding/2.0,\
            group_width + bar_group_padding)

        log = yscale == 'log'
        

        color_squares = []
        for i in range(len(ys)):
            
            if type == 'bar':
                alpha = 0 if i in skip_series else 1
                yerr=yerrs[i] if yerrs else None
                rects = ax.bar(ind + i*bar_width, ys[i], bar_width, log=log,\
                    yerr=yerr, error_kw={'zorder':4, 'ecolor':'gray'},
                    alpha=alpha,\
                    color=colors[i], edgecolor=style['bar_edgecolor'])
                color_squares.append(rects[0])
                if label_bars: autolabel(rects, ax)
            elif type == 'stackbar':
                bottom = [0]*len(ys[i][0])  # keep cumulative sum of height of each bar (bottom of next segment)
                for j in range(len(ys[i])):
                    if stackbar_colors_denote == 'series':
                        color = colors[i]
                        hatchstyle = hatchstyles[j]
                    elif stackbar_colors_denote == 'segments':
                        color = colors[j]
                        hatchstyle = hatchstyles[i]
                    yerr = yerrs[i][j] if yerrs else None
                    
                    rects = ax.bar(ind + i*bar_width, ys[i][j], bar_width, log=log,
                        yerr=yerr, error_kw={'zorder':4, 'ecolor':'gray'},
                        bottom=bottom, color=color, edgecolor=style['bar_edgecolor'],
                        hatch=hatchstyle)
                    bottom = [sum(x) for x in zip(bottom, ys[i][j])]
                    if stackbar_colors_denote == 'series' and j == 0:
                        color_squares.append(rects[0])
                    elif stackbar_colors_denote == 'segments' and i == 0:
                        color_squares.insert(0, rects[0])
                    #if label_bars: autolabel(rects, ax)  TODO: support

                # Add invisible data to add another legend for segment hatchstyles
                if i == 0:  # only do this once   FIXME what if first series doesn't have all segments?
                    if stackbar_colors_denote == 'series':
                        num_segments = len(ys[i])
                    elif stackbar_colors_denote == 'segments':
                        num_segments = len(ys)
                    n=[]
                    for k in range(num_segments):
                        n.append(ax.bar(0,0,color = "gray", hatch=hatchstyles[k],\
                            edgecolor=style['bar_edgecolor'])[0])
                    if stackbar_colors_denote == 'series' and stackbar_pattern_labels != None:
                        pass  # not sure why I reversed these before?
                        #n = reversed(n)
                        #stackbar_pattern_labels = reversed(stackbar_pattern_labels)
                    if stackbar_pattern_labels:
                        # for square legend swatches
                        handlermap = {}
                        for swatch in n:
                            handlermap[swatch] = HandlerSquare()

                        pattern_legend = ax.legend(n, stackbar_pattern_labels,
                            loc=pattern_legend_loc, ncol=legend_cols, frameon=legend_border,
                            columnspacing=legend_col_spacing,
                            bbox_to_anchor=pattern_legend_bbox,
                            labelspacing=labelspacing, handletextpad=handletextpad,
                            handler_map=handlermap,
                            prop={'size':legend_text_size + style['textsize_delta']})
                        ax.add_artist(pattern_legend)
                        
                    

        ax.set_xticks(ind + num_series/2.0*bar_width)
        if xtick_labels == None:
            xtick_labels = xs[0]
        ax.set_xticklabels(xtick_labels, horizontalalignment=xtick_label_horizontal_alignment,\
            rotation=xtick_label_rotation)
        ax.set_xlim(0, ind[-1]+group_width+bar_group_padding/2.0)
        ax.set_axisbelow(True)  # bars on top of grid lines

        # for legend, used below
        lines = color_squares
    elif type == 'hist':
        ax.hist(xs, bins=bins, **kwargs)
    elif type == 'stackplot':
        lines = ax.stackplot(xs, ys)
        ax.set_xticks(np.arange(min(xs), max(xs)+1, 1.0))

    elif type == 'barh':
        # I tend to think of the left-most bar in a normal bar plot as 
        # being the top-most bar in a horizontal bar plot
        ys = ys[::-1]  # reverse list
        xs = xs[::-1]  # reverse list
        colors = colors[::-1]  # reverse list
        if yerrs != None:
            yerrs = yerrs[::-1]
        # DON'T reverse labels, since we're going to add color swatches to legend backwards anyway

        num_groups = max([len(series) for series in ys])  # num clusters of bars
        num_series = len(ys)   # num bars in each cluster
        
        group_width = bar_width * num_series
        ind = np.arange(bar_group_padding/2.0,\
            num_groups*(bar_width*num_series+bar_group_padding) + bar_group_padding/2.0,\
            group_width + bar_group_padding)
        
        log = yscale == 'log'
        
        color_squares = []
        for i in range(len(ys)):
            # I tend to think of the left-most bar group in a normal bar plot as 
            # being the top-most bar group in a horizontal bar plot
            ys[i] = ys[i][::-1]  # reverse list
            xs[i] = xs[i][::-1]  # reverse list

            if type == 'barh':
                alpha = 0 if i in skip_series else 1
                yerr=yerrs[i] if yerrs else None
                rects = ax.barh(ind + i*bar_width, ys[i], bar_width, log=log,\
                    xerr=yerr, error_kw={'zorder':4, 'ecolor':'black'},
                    alpha=alpha,\
                    color=colors[i], edgecolor=style['bar_edgecolor'])
                color_squares.insert(0, rects[0])
                if label_bars: autolabel(rects, ax)

        ax.set_yticks(ind + num_series/2.0*bar_width)
        if xtick_labels == None:
            xtick_labels = xs[0]
        ax.set_yticklabels(xtick_labels)
        ax.set_ylim(0, ind[-1]+group_width+bar_group_padding/2.0)
        ax.tick_params(axis='y', pad=15)
        ax.set_axisbelow(True)  # bars on top of grid lines

        # for legend, used below
        lines = color_squares

    else:
        print '[ERROR]  Unknown plot type: %s' % type
    
    

    #################### AXIS RANGES ####################
    if xlim:
        xmin = xlim[0] if xlim[0] is not None else ax.get_xlim()[0]
        xmax = xlim[1] if xlim[1] is not None else ax.get_xlim()[1]
        ax.set_xlim((xmin, xmax))
    if ylim:
        ymin = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
        ymax = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
        ax.set_ylim((ymin, ymax))


            
    #################### X TICKS ####################
    # xtick frequency (remove extra values if frequency != 1)
    # TODO: minor xticks at frequency 1?
    master_xticks = master_xticks[::xtick_frequency] if master_xticks != None else None
    master_xnums = master_xnums[::xtick_frequency] if master_xnums != None else None
    xticks = xticks[::xtick_frequency] if xticks != None else None
    xtick_labels = xtick_labels[::xtick_frequency] if xtick_labels != None else None

    # transform label text
    if master_xticks != None:
        master_xticks = [xtick_label_transform(l) for l in master_xticks]
    if xtick_labels != None:
        xtick_labels = [xtick_label_transform(l) for l in xtick_labels]

    if xticks:
        ax.set_xticks(xticks)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, color=style['foreground_color'],\
                horizontalalignment=xtick_label_horizontal_alignment,\
                rotation=xtick_label_rotation)
    elif master_xticks:
        ax.set_xticks(master_xnums)
        ax.set_xticklabels(master_xticks, color=style['foreground_color'],\
            horizontalalignment=xtick_label_horizontal_alignment,\
             rotation=xtick_label_rotation)
    if xscale == 'log' and ax.get_xlim()[1] < power_limits[1] and \
            ax.get_xlim()[0] > power_limits[0]:
        #ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        #ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for tl in ax.get_xticklabels():
        tl.set_color(style['foreground_color'])

    
    #################### Y TICKS ####################
    if yscale == 'log' and ax.get_ylim()[1] < power_limits[1] and \
            ax.get_ylim()[0] > power_limits[0]:
        #ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for tl in ax.get_yticklabels():
        tl.set_color(style['foreground_color'])


    #################### ADDITONAL Y AXES ####################
    if additional_ylabels:
        addl_y_axes = []
        for label in additional_ylabels:
            new_ax = ax.twinx()
            addl_y_axes.append(new_ax)
            new_ax.set_ylabel(label, fontsize=ylabel_size + style['textsize_delta'])
            #new_ax.set_yticklabels([]) # temp
            if additional_yscales:
                new_ax.set_yscale(additional_yscales[0])  # TODO: use real index!
            if additional_ylims:
                new_ax.set_ylim(additional_ylims[0])  # TODO: use real index!
                

        # plot the extra series
        for i in range(len(ys)):
            # FIXME: index the correct addl y axis!
            if axis_assignments[i] != 1: continue
            marker = markerstyles[i] if show_markers else None
            line, = addl_y_axes[0].plot(xs[i], ys[i], linestyle=linestyles[i],\
                marker=marker,\
                color=colors[i], label=labels[i], **kwargs)
            lines[i] = line
            if yerrs:
                addl_y_axes[0].fill_between(xs[i], numpy.array(ys[i])+numpy.array(yerrs[i]),\
                numpy.array(ys[i])-numpy.array(yerrs[i]), color=colors[i], alpha=0.5)


    #################### GUIDE LINES ####################
    for line in guide_lines:
        if line['stretch']:
            # compute new endpoints at edges of plot boundaries
            line['endpoints'] = endpoints_for_stretched_line(\
                line['endpoints'], ax.get_xlim(), ax.get_ylim())

        (line_xs, line_ys) = zip(*line['endpoints'])
        ax.add_line(matplotlib.lines.Line2D(line_xs, line_ys, zorder=1, **line['line_args']))
        if 'label' in line and line['label'] != None:
            ax.text(line['endpoints'][1][0]+1, line['endpoints'][1][1]-20, line['label'],\
                ha='right', **line['label_args'])
    
    

    #################### LEGEND ####################
    legend_loc = legend_loc.replace('top', 'upper').replace('bottom', 'lower')
    if show_legend and labels: 
        if type == 'stackplot':
            lines = [matplotlib.patches.Rectangle((0,0), 0,0, facecolor=pol.get_facecolor()[0]) for pol in lines]

        # not real keywords; manually put legend outside plot
        if legend_loc == 'below':  
            legend_bbox=(0, -0.1, 1, 0)
            legend_loc='upper center'
        elif legend_loc == 'above':  
            legend_bbox=(0, 1.1, 1, 0)
            legend_loc='lower center'
        elif legend_loc == 'out right':
            legend_bbox=(1.03, 1.0)
            legend_loc='upper left'


        # for square legend swatches
        handlermap = {}
        if type in ('bar', 'barh', 'stackbar'):
            for swatch in lines:
                handlermap[swatch] = HandlerSquare()


        ax.legend(lines, labels, loc=legend_loc, ncol=legend_cols,
            columnspacing=legend_col_spacing,
            bbox_to_anchor=legend_bbox,
            frameon=legend_border, labelspacing=labelspacing,
            handletextpad=handletextpad,
            handler_map=handlermap,
            prop={'size':legend_text_size + style['textsize_delta']})
    else:
        ax.legend_ = None  # TODO: hacky



    # make sure no text is clipped along the boundaries
    #plt.tight_layout()

    extra_artists = (ax.legend_,) if ax.legend_ else []

    if filename:
        plt.savefig(filename, transparent=style['transparent_bg'], bbox_extra_artists=extra_artists, bbox_inches='tight')
    #plt.show()

    plt.close()  # to save memory

    return lines, labels  # making an overall figure legend






##
## WRAPPERS (for specific types of plots)
##

# TODO: merge this with plot()
# matrix should be a list of lists. element 0, 0 is in the bottom left
def heatmap(matrix, colorbar=True, colorbar_label=None, color_map=plt.cm.Blues,\
        normalize_pos_neg=False,\
        xlabel=None, ylabel=None, filename=None):

    # TODO: if colorbar=False, don't make second axes
    #fig, ax = plt.subplots()
    fig = plt.figure()
    heatmap_ax = plt.subplot2grid((1, 7), (0, 0), colspan=6)
    colorbar_ax = plt.subplot2grid((1, 7), (0, 6))

    norm = MidpointNormalize(midpoint=0) if normalize_pos_neg else None
    heatmap = heatmap_ax.pcolor(np.array(matrix), norm=norm, cmap=color_map)

    if xlabel: heatmap_ax.set_xlabel(xlabel)
    if ylabel: heatmap_ax.set_ylabel(ylabel)
    

    # FIXME hardcoded stuff that shouldn't be
    plt.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom='off',      # ticks along the bottom edge are off
        #top='off',         # ticks along the top edge are off
        #labelbottom='off', # labels along the bottom edge are off
    )
    plt.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #left='off',
        #right='off',
        #labelleft='off',
    )

    
    if colorbar:
        cbar = plt.colorbar(heatmap, cax=colorbar_ax)
        if colorbar_label: cbar.ax.set_ylabel(colorbar_label, rotation=90)
    
    # make sure no text is clipped along the boundaries
    plt.tight_layout()

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)


def distribution_function(data, numbins=None, cdf=False, pdf=False, bars=False, labels=None, **kwargs):
    '''Wrapper for making CDFs (and PDFs)'''
    xs = []
    ys = []
    for d in data:
        cdf_y, cdf_x, pdf_y, pdf_x = cdf_vals_from_data(d, numbins)

        if cdf:
            xs.append(cdf_x)
            ys.append(cdf_y)

        if pdf:
            xs.append(pdf_x)
            ys.append(pdf_y)

    if cdf and pdf:
        if labels:
            # need to duplicate each label and add (CDF) or (PDF)
            new_labels = []
            for label in labels:
                new_labels += ['%s (CDF)' % label, '%s (PDF)' % label]
            labels = new_labels
        else:
            # just label each line CDF or PDF
            labels = ['CDF', 'PDF'] * len(data)

    if bars:
        return bar(xs, ys, labels=labels, show_markers=False, **kwargs)
    else:
        return plot(xs, ys, labels=labels, show_markers=False, **kwargs)


def pdf(data, numbins=None, labels=None, **kwargs):
    return distribution_function(data, numbins=numbins, labels=labels, pdf=True, **kwargs)

def cdf(data, numbins=None, labels=None, **kwargs):
    return distribution_function(data, numbins=numbins, labels=labels, cdf=True, ylabel='CDF', **kwargs)

def bar(xs, ys, xtick_label_rotation=45,\
    xtick_label_horizontal_alignment='right', **kwargs):
    '''Wrapper for bar charts'''

    return plot(xs, ys, type='bar', xtick_label_rotation=xtick_label_rotation,\
        xtick_label_horizontal_alignment=xtick_label_horizontal_alignment,\
        **kwargs)

def stackbar(xs, ys, xtick_label_rotation=45,\
    xtick_label_horizontal_alignment='right', **kwargs):
    '''Wrapper for bar charts'''

    return plot(xs, ys, type='stackbar', xtick_label_rotation=xtick_label_rotation,\
        xtick_label_horizontal_alignment=xtick_label_horizontal_alignment,\
        **kwargs)

def stackplot(ys, sortindex=-1, **kwargs):
    '''Wrapper for making stackplots'''

    # sort, maybe
    if sortindex >= 0:
        ys = zip(*sorted(zip(*ys), key=lambda x: x[sortindex]))

    x = np.arange(len(ys[0]))

    return plot(x, ys, xlim=(min(x), max(x)), type='stackplot', **kwargs)

def barh(xs, ys, **kwargs):
    '''Wrapper for horizontal bar charts'''
    
    return plot(xs, ys, type='barh', **kwargs)
    



def main():

    ## test cdf
    #data = [1, 2, 3, 4, 5, 6]

    #y_vals, x_vals = cdf_vals_from_data(data)
    #plot(y_vals, x_vals, 'cdf')

    ## test heatmap
    #matrix = [[1, 0, 1], [1, 0, 1], [0, 0, 1]]
    #heatmap(matrix, xlabel='App Policies', ylabel='User Policies',\
    #    filename="/Users/dnaylor/Desktop/heatmap.pdf")

    # test stackplot
    y1 = [3, 1, 7, 5]
    y2 = [1, 2, 3, 4]
    y3 = [4, 9, 8, 3]
    stackplot([y1, y2, y3], sortindex=0, filename='/Users/dnaylor/Desktop/stackplot.pdf')

if __name__ == '__main__':
    main()
