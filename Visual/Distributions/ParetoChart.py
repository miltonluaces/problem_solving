from operator import itemgetter
import matplotlib.pyplot as plt

def Pareto(data, labels=[], cumPlot=True, limit=1.0, axes=None):
    assert 0.0 <= limit <= 1.0, 'limit must be a positive scalar between 0.0 and 1.0'
    dataArgs=(); data_kw={}; line_args=('g',); line_kw={}; limit_kw={}
    # re-order the data in descending order
    data = list(data)
    n = len(data)
    if n!=len(labels):
        labels = range(n)
    ordered = sorted(zip(data, labels), key=itemgetter(0), reverse=True)
    ordData = [dat for dat, lab in ordered]
    ordLabels = [lab for dat, lab in ordered]
    
    # create the cumulative line data
    line_data = [0.0]*n
    total_data = float(sum(ordData))
    for i, dat in enumerate(ordData):
        if i==0: line_data[i] = dat/total_data
        else: line_data[i] = sum(ordData[:i+1])/total_data

    # determine where the data will be trimmed based on the limit
    ltcount = 0
    for ld in line_data:
        if ld<limit:
            ltcount += 1
    limLoc = range(ltcount+1)
    
    limData = [ordData[i] for i in limLoc]
    limLabels = [ordLabels[i] for i in limLoc]
    limLine = [line_data[i] for i in limLoc]
    
    # if axes is specified, grab it and focus on its parent figure; otherwise create a new figure
    if axes:
        plt.sca(axes)
        ax1 = axes
        fig = plt.gcf()
    else:
        fig = plt.gcf()
        ax1 = plt.gca()
    
    # Create the second axis
    if cumPlot: ax2 = ax1.twinx()
    
    # Plotting
    if 'align' not in data_kw: data_kw['align'] = 'center'
    if 'width' not in data_kw: data_kw['width'] = 0.9
    ax1.bar(limLoc, limData, *dataArgs, **data_kw)
    if cumPlot: ax2.plot(limLoc, [ld*100 for ld in limLine], *line_args, **line_kw)
    ax1.set_xticks(limLoc)
    ax1.set_xlim(-0.5,len(limLoc)-0.5)
    
    # Formatting
    if cumPlot:
        # since the sum-total value is not likely to be one of the tick marks, let's make it the top-most one, regardless of label closeness
        ax1.set_ylim(0, total_data)
        loc = ax1.get_yticks()
        newloc = [loc[i] for i in range(len(loc)) if loc[i]<=total_data]
        newloc += [total_data]
        ax1.set_yticks(newloc)
        ax2.set_ylim(0, 100)
        if limit<1.0:
            xmin,xmax = ax1.get_xlim()
            if 'linestyle' not in limit_kw:
                limit_kw['linestyle'] = '--'
            if 'color' not in limit_kw:
                limit_kw['color'] = 'r'
            ax2.axhline(limit*100, xmin-1, xmax-1, **limit_kw)
    
    # set the x-axis labels
    ax1.set_xticklabels(limLabels)
    
    # adjust the second axis if cumplot=True
    if cumPlot:
        yt = [str(int(it))+r'%' for it in ax2.get_yticks()]
        ax2.set_yticklabels(yt)

    if cumPlot: return fig,ax1,ax2
    else: return fig,ax1



if __name__=='__main__':



    """
    Plots a `pareto chart`_ of input categorical data. NOTE: The matplotlib
    command ``show()`` will need to be called separately. The default chart
    uses the following styles:
    
    - bars: 
       - color = blue
       - align = center
       - width = 0.9
    - cumulative line:
       - color = blue
       - linestyle = solid
       - markers = None
    - limit line:
       - color = red
       - linestyle = dashed
    
    Parameters
    ----------
    data : array-like
        The categorical data to be plotted (not necessary to put in descending
        order prior).
        
    Optional
    --------
    labels : list
        A list of strings of the same length as ``data`` that provide labels
        to the categorical data. If none provided, a simple integer value is
        used to label the data, relating to the original order as given. If
        a list is provided, but not the same length as ``data``, then it will
        be treated as if no labels had been input at all.
    cumplot : bool
        If ``True``, a cumulative percentage line plot is included in the chart
        (Default: True) and a second axis indicating the percentage is returned.
    axes : axis object(s)
        If valid matplotlib axis object(s) are given, the chart and cumulative
        line plot of placed on the given axis. Otherwise, a new figure is
        created.
    limit : scalar
        The cumulative percentage value at which the input data should be 
        "chopped off" (should be a value between zero and one).
    data_args : tuple
        Any valid ``matplotlib.pyplot.bar`` non-keyword arguments to apply to
        the bar chart.
    data_kw : dict
        Any valid ``matplotlib.pyplot.bar`` keyword arguments to apply to
        the bar chart.
    line_args : tuple
        Any valid ``matplotlib.pyplot.plot`` non-keyword arguments to apply to
        the cumulative line chart.
    line_kw : dict
        Any valid ``matplotlib.pyplot.plot`` keyword arguments to apply to
        the cumulative line chart.
    limit_kw : dict
        Any valid ``matplotlib.axes.axhline`` keyword arguments to apply to
        the limit line.
        
    Returns
    -------
    fig : matplotlib.figure
        The parent figure object of the chart.
    ax1 : matplotlib.axis
        The axis for the categorical data.
    ax2 : matplotlib.axis
        The axis for the cumulative line plot (not returned if 
        ``cumplot=False``).
    
    Examples
    --------
    
    The following code is the same test code if the ``paretoplot.py`` file is
    run with the command-line call ``$ python paretoplot.py``::

        # plot data using the indices as labels
        data = [21, 2, 10, 4, 16]
        
        # define labels
        labels = ['tom', 'betty', 'alyson', 'john', 'bob']
        
        # create a grid of subplots
        fig,axes = plt.subplots(2, 2)
        
        # plot first with just data
        pareto(data, axes=axes[0, 0])
        plt.title('Basic chart without labels', fontsize=10)
        
        # plot data and associate with labels
        pareto(data, labels, axes=axes[0, 1], limit=0.75, line_args=('g',))
        plt.title('Data with labels, green cum. line, limit=0.75', fontsize=10)
        
        # plot data and labels, but remove lineplot
        pareto(data, labels, cumplot=False, axes=axes[1, 0], 
               data_kw={'width': 0.5, 'color': 'g'})
        plt.title('Data without cum. line, bar width=0.5', fontsize=10)
        
        # plot data cut off at 95%
        pareto(data, labels, limit=0.95, axes=axes[1, 1], limit_kw={'color': 'y'})
        plt.title('Data trimmed at 95%, yellow limit line', fontsize=10)
    
        # format the figure and show
        fig.canvas.set_window_title('Pareto Plot Test Figure')
        plt.show()

    .. _pareto chart: http://en.wikipedia.org/wiki/Pareto_chart
    
    """

