import matplotlib  # TODO :needed?
import numpy as np # import needed?


# This stuff needs to be set before we import matplotlib.pyplot
matplotlib.use('PDF')  # save plots as PDF
font = {'size': 20,
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



import matplotlib.pyplot as plt
import numpy
from scipy.stats import cumfreq
from matplotlib.colors import Normalize


##
## STYLES
##

default_style = {
    'colors': ('b', 'g', 'r', 'c', 'm', 'y'),
    'linestyles': ('-', '--', '-.', ':'),
    'hatchstyles': (None, '/', '\\', 'o', '*', '+', '//', '\\\\', '-', 'x', 'O', '.'),
    'gridalpha':1.0,
    'frame_lines':{'top':True, 'right':True, 'bottom':True, 'left':True},
    'tick_marks':{'top':True, 'right':True, 'bottom':True, 'left':True},
    'bar_edgecolor':'black',
    'errorbar_style':'line',
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
tableau20 = [tableau20[i] for i in range(len(tableau20)) if i % 2 == 0]


pretty_style = {
    'colors': tableau20,
    'linestyles': ('-', '--', '-.', ':'),
    'hatchstyles': (None, '/', '\\', 'o', '+', '*', '//', '\\\\', '-', 'x', 'O', '.'),
    'gridalpha':0.3,
    'frame_lines':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'tick_marks':{'top':False, 'right':False, 'bottom':True, 'left':True},
    'bar_edgecolor':'white',
    'errorbar_style':'fill',
}
    




##
## HELPER FUNCTIONS
##
def cdf_vals_from_data(data, numbins=None):

    # make sure data is a numpy array
    data = numpy.array(data)
    
    # by default, use numbins equal to number of distinct values
    if numbins == None:
        numbins = numpy.unique(data).size
    
    # bin the data and count each bin
    result = cumfreq(data, numbins, (data.min(), data.max()))
    cum_bin_counts = result[0]
    min_bin_val = result[1]
    bin_size = result[2]

    # normalize bin counts so rightmost count is 1
    cum_bin_counts /= cum_bin_counts.max()

    # make array of x-vals (lower end of each bin)
    x_vals = numpy.linspace(min_bin_val, min_bin_val+bin_size*numbins, numbins)

    # CDF always starts at (0, 0)
    cum_bin_counts = numpy.insert(cum_bin_counts, 0, 0)
    x_vals = numpy.insert(x_vals, 0, 0)


    return cum_bin_counts, x_vals

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
def plot(xs, ys, labels=None, xlabel=None, ylabel=None, title=None,\
         additional_ylabels=None, num_series_on_addl_y_axis=0,\
         axis_assignments=None, additional_ylims=None,\
         xlabel_size=20, ylabel_size=20, labelspacing=0.2, handletextpad=0.5,\
         marker='o', linestyles=None, legend='best', show_legend=True,\
         legend_cols=1, linewidths=None, legend_border=False,\
         colors=None, axis=None, legend_text_size=20, filename=None,\
         xscale=None, yscale=None, type='series', bins=10, yerrs=None,\
         additional_yscales=None,\
         width_scale=1, height_scale=1, xlim=None, ylim=None,\
         label_bars=False, bar_width=1, bar_group_padding=1,\
         xticks=None, xtick_labels=None, xtick_label_rotation=0,\
         xtick_label_horizontal_alignment='center',\
         show_y_tick_labels=True, show_x_tick_labels=True,\
         hatchstyles=None, stackbar_pattern_labels=None,\
         style=pretty_style, grid=None,\
         fig=None, ax=None,\
         **kwargs):
     # TODO: split series and hist into two different functions?
     # TODO: change label font size back to 20
     # TODO: clean up multiple axis stuff 
     # TODO: legend loc, replace 'bottom' with lower and 'top' with 'upper'
     # TODO: what is the default labelspacing?

    # if we want to do subplots, caller may have passed in an existing figure
    if not fig or not ax:
        fig, ax = plt.subplots()
        width, height = fig.get_size_inches()
        fig.set_size_inches(width*width_scale, height*height_scale)


    #################### SETUP ####################
    if xlabel: ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if ylabel: ax.set_ylabel(ylabel, fontsize=ylabel_size)
    if not show_x_tick_labels: ax.set_xticklabels([])
    if not show_y_tick_labels: ax.set_yticklabels([])
    if title: ax.set_title(title)
    if axis: 
        ax.set_xlim(axis[0:2])
        ax.set_ylim(axis[2:4])
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    lines = [None]*len(ys)
    show_legend = show_legend and labels != None
    if not labels: labels = ['']*len(ys)
    if not linewidths: linewidths = [3]*len(ys)
    if not axis_assignments: axis_assignments = [0]*len(ys)
            
            
    # If X axis points are strings, make a dummy x array for each string x list.
    # not each series might have a data point for each X value, so we need to 
    # make a "master" xtick label list with all of the X values in right order
    # FIXME: for now, assuming they're sortable. should add option for caller to
    # pass the master list in case they're not sortable.
    # NOTE: for now, this assumes that either all series have numeric X axes or
    # none do.
    master_xticks = None
    master_xnums = None
    if type not in ('stackplot', 'bar', 'stackbar'):
        try:
            float(xs[0][0])
        except ValueError:
            # make & sort list of all X tick labels used by any series
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
        for i in range(len(ys)):
            colors.append(style['colors'][i%len(style['colors'])])
    else:
        for i in range(len(colors)):
            if isinstance(colors[i], int):
                colors[i] = style['colors'][colors[i]]

    if not linestyles:
        linestyles = []
        for i in range(len(ys)):
            linestyles.append(style['linestyles'][i%len(style['linestyles'])])
    else:
        for i in range(len(linestyles)):
            if isinstance(linestyles[i], int):
                linestyles[i] = style['linestyles'][linestyles[i]]

    if not hatchstyles:
        hatchstyles = []
        for i in range(len(ys[0])):
            hatchstyles.append(style['hatchstyles'][i%len(style['hatchstyles'])])
    else:
        for i in range(len(hatchstyles)):
            if isinstance(hatchstyles[i], int):
                hatchstyles[i] = style['hatchstyles'][hatchstyles[i]]

    # frame lines
    for side, visible in style['frame_lines'].iteritems():
        ax.spines[side].set_visible(visible)

    # tick marks  (spacing and labels set below)
    ax.tick_params(**style['tick_marks'])
    
    # show grid lines?
    if grid:
        grid_args = {'zorder':0, 'alpha':style['gridalpha']}
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


            # TODO: simplify these two cases? repetitive.
            if yerrs and style['errorbar_style'] == 'line':
                line = ax.errorbar(xs[i], ys[i], linestyle=linestyles[i], marker=marker,\
                    linewidth=linewidths[i], color=colors[i], label=labels[i],\
                    yerr=yerrs[i], **kwargs)
            else:
                line, = ax.plot(xs[i], ys[i], linestyle=linestyles[i], marker=marker,\
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
        

        color_squares = []
        for i in range(len(ys)):

            if type == 'bar':
                rects = ax.bar(ind + i*bar_width, ys[i], bar_width, color=colors[i], zorder=3)
                color_squares.append(rects[0])
                if label_bars: autolabel(rects, ax)
            elif type == 'stackbar':
                bottom = [0]*len(ys[i][0])  # keep cumulative sum of height of each bar (bottom of next segment)
                for j in range(len(ys[i])):
                    rects = ax.bar(ind + i*bar_width, ys[i][j], bar_width,\
                        bottom=bottom, color=colors[i], edgecolor=style['bar_edgecolor'],\
                        hatch=hatchstyles[j], zorder=3)
                    bottom = [sum(x) for x in zip(bottom, ys[i][j])]
                    if j == 0:
                        color_squares.append(rects[0])
                    #if label_bars: autolabel(rects, ax)  TODO: support

                # Add invisible data to add another legend for patterns
                if i == 0:  # only do this once   FIXME what if first series doesn't have all segments?
                    num_segments = len(ys[i])
                    n=[]
                    for k in range(num_segments):
                        n.append(ax.bar(0,0,color = "gray", hatch=hatchstyles[k],\
                            edgecolor=style['bar_edgecolor']))
                    if stackbar_pattern_labels:
                        pattern_legend = ax.legend(reversed(n), reversed(stackbar_pattern_labels),\
                            loc='upper center', ncol=legend_cols, frameon=legend_border,\
                            labelspacing=labelspacing, handletextpad=handletextpad,\
                            prop={'size':legend_text_size})
                        ax.add_artist(pattern_legend)
                        
                    

        ax.set_xticks(ind + num_series/2.0*bar_width)
        ax.set_xticklabels(xs[0], horizontalalignment=xtick_label_horizontal_alignment,\
            rotation=xtick_label_rotation)
        ax.set_xlim(0, ind[-1]+group_width+bar_group_padding/2.0)

        # for legend, used below
        lines = color_squares
    elif type == 'hist':
        ax.hist(xs, bins=bins, **kwargs)
    elif type == 'stackplot':
        lines = ax.stackplot(xs, ys)
        ax.set_xticks(np.arange(min(xs), max(xs)+1, 1.0))

            
    # Set xticks and labels, if non-default values provided        
    if xticks:
        ax.set_xticks(xticks)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, horizontalalignment='right', rotation=xtick_label_rotation)
    elif master_xticks:
        ax.set_xticks(master_xnums)
        ax.set_xticklabels(master_xticks, horizontalalignment='right', rotation=xtick_label_rotation)


    # Additional axes?
    if additional_ylabels:
        addl_y_axes = []
        for label in additional_ylabels:
            new_ax = ax.twinx()
            addl_y_axes.append(new_ax)
            new_ax.set_ylabel(label, fontsize=ylabel_size)
            #new_ax.set_yticklabels([]) # temp
            if additional_yscales:
                new_ax.set_yscale(additional_yscales[0])  # TODO: use real index!
            if additional_ylims:
                new_ax.set_ylim(additional_ylims[0])  # TODO: use real index!
                

        # plot the extra series
        for i in range(len(ys)):
            # FIXME: index the correct addl y axis!
            if axis_assignments[i] != 1: continue
            line, = addl_y_axes[0].plot(xs[i], ys[i], linestyle=linestyles[i], marker=marker,\
                color=colors[i], label=labels[i], **kwargs)
            lines[i] = line
            if yerrs:
                addl_y_axes[0].fill_between(xs[i], numpy.array(ys[i])+numpy.array(yerrs[i]),\
                numpy.array(ys[i])-numpy.array(yerrs[i]), color=colors[i], alpha=0.5)

    if show_legend and labels: 
        if type == 'stackplot':
            lines = [matplotlib.patches.Rectangle((0,0), 0,0, facecolor=pol.get_facecolor()[0]) for pol in lines]
        ax.legend(lines, labels, loc=legend, ncol=legend_cols, frameon=legend_border,\
            labelspacing=labelspacing, handletextpad=handletextpad, prop={'size':legend_text_size})
            
    else:
        ax.legend_ = None  # TODO: hacky


    # make sure no text is clipped along the boundaries
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    #plt.show()

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


def cdf(data, numbins=None, **kwargs):
    '''Wrapper for making CDFs'''
    xs = []
    ys = []
    for d in data:
        y, x = cdf_vals_from_data(d, numbins)
        xs.append(x)
        ys.append(y)
    return plot(xs, ys, ylabel='CDF', marker=None, **kwargs)

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
