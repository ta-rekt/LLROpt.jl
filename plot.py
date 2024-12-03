import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import networkx as nx

def draw_circuit(
        weighted_edges, 
        input_edges, 
        input_labels=None,
        node_labels=None,
        node_size=2000,
        title=None,
        dims=None,
        bottom_set=[1,3],
        edge_cmap=None,
        node_cmap=None,
        node_color="black",
        align=None
    ):

    # create adjacency matrix
    G = nx.DiGraph()

    for u, v, w in weighted_edges:
        # print("edge: ({},{})".format(u, v), "weight: {}".format(w))
        G.add_edge(u, v, weight=w)

    if align is None:
        align = "horizontal"

    pos = nx.bipartite_layout(G, bottom_set, align=align)
    weights = [w for _, _, w in weighted_edges]
    edges = [(u, v) for u, v, _ in weighted_edges]

    if node_cmap is None:
        node_cmap = plt.cm.copper

    if edge_cmap is None:
        edge_cmap = plt.cm.Blues
    elif edge_cmap == 'v':
        edge_cmap = plt.cm.viridis

    if node_labels is None:
        node_labels = {0: r'OFF \ U',
                       1: r'ON \ U',
                       2: r'OFF \ B',
                       3: r'ON \ B'}
    
    if input_labels is None:
        input_labels = dict(zip(input_edges, [r'$c$' for _ in input_edges]))

    plt.style.use('default')

    if dims is None:
        dims = (5,4)

    fig, ax = plt.subplots(1,1,figsize=dims,layout='constrained')

    if title is not None:
        ax.set_title(title)

    nx.draw_networkx_labels(
        G, 
        pos, 
        node_labels, 
        font_size=18,
        font_color='white', 
        verticalalignment='center', 
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=node_size, 
        node_color=node_color, 
        cmap=node_cmap,
        linewidths=None,
        ax=ax
    )
    edges = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=edges,
        arrows=True,
        node_size=node_size,
        arrowstyle="-|>",
        connectionstyle='arc3,rad=0.1',
        arrowsize=20,
        edge_color=weights,
        edge_cmap=edge_cmap,
        width=6,
    )
    nx.draw_networkx_edge_labels(
        G, 
        pos, 
        input_labels, 
        connectionstyle='arc3,rad=0.1', 
        font_size=12, 
        ax=ax
    )

    pc = mpl.collections.PatchCollection(edges, cmap=edge_cmap)
    pc.set_array(weights)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax, label="log rates")
    plt.show()

    return fig, ax, G

def plot_dkl_ep(
        ydata,
        xdata,
        figax=None,
        add_data=False,
        title=None,
        ylabel='',
        xlim=None,
        ylim=None,
        marker=None,
        size=15,
        data_label=None,
        maxvals=None,
        legendloc=None,
        kwargs={}
    ):

    plt.style.use('classic')

    nbins = 20
    nrow, _ = ydata.shape
    means = np.array(range(nrow)) #[:100]
    bin_bounds = np.linspace(0, np.max(means), nbins)

    if figax is None:
        fig, ax = plt.subplots(1,1, layout='constrained')
    else:
        fig, ax = figax

    cmap = plt.colormaps["viridis"]
    plt.rcParams['text.usetex'] = True

    for i in range(len(bin_bounds)-1):
        ub = bin_bounds[i+1]
        lb = bin_bounds[i]
        indices_in_bin = np.where(np.bitwise_and(means >= lb, means < ub))[0]
        means_in_bin = means[indices_in_bin]
        mu = np.mean(means_in_bin) / np.max(means)

        # optimized circuits
        yopt = -ydata[indices_in_bin, 4]
        xopt = xdata[indices_in_bin]

        if i == 0:
            ax.scatter(
                xopt, 
                yopt, 
                s=size, 
                alpha=1, 
                color=cmap(mu), 
                marker=marker, 
                label=data_label,
                **kwargs)
        else:
            ax.scatter(
                xopt, 
                yopt, 
                s=size, 
                alpha=1, 
                color=cmap(mu), 
                marker=marker,
                **kwargs)

    if not add_data:
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,max(means)), cmap=cmap),
                     ax=ax, label="Optimization step")

        ax.set_xlabel('Entropy production rate (bits/second)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

    if maxvals is not None:
        for key, val in maxvals.items():
            ax.axhline(y=val, linestyle='dashed', color='red', label=key)

    ax.legend(loc=legendloc)        

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    return fig, ax