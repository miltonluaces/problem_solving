import metaknowledge as mk
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import community 
import pandas
import plotly.plotly as py
import plotly.graph_objs as go

# install python-louvain for community detection

plt.rc("savefig", dpi=600)
sns.set(font_scale=.75)

#Read Data, Create a RecordCollection, and Generate a Network
rc = mk.RecordCollection('raw_data/imetrics/', cached = True)
rcSp = rc.yearSplit(2010,2014)

coaNet = rc.networkCoAuthor() 
print(mk.graphStats(coaNet))

mk.dropEdges(coaNet, minWeight = 2, dropSelfLoops = True) 
giant_coauth = max(nx.connected_component_subgraphs(coaNet), key=len)
print(mk.graphStats(giant_coauth))

deg = nx.degree_centrality(giant_coauth)
eig = nx.eigenvector_centrality(giant_coauth)

centDf = pandas.DataFrame.from_dict([deg, eig])
centDf = pandas.DataFrame.transpose(centDf)
centDf.columns = ['degree', 'eigenvector']
centDf.sort_values('degree', ascending = False)[:10]

centDf100 = centDf.sort_values('degree', ascending = False)[:100]

trace = go.Bar(x = centDf100.index, y = centDf100['degree'])
data = [trace]

layout = go.Layout(yaxis=dict(title='Degree Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='cent-dist')

# Top 100 betweenness centrality scores
centDfbc = centDf.sort_values('betweenness', ascending = False)[:100]

trace = go.Bar(x = centDfbc.index, y = centDfbc['betweenness'])
data = [trace]

layout = go.Layout(yaxis=dict(title='Betweenness Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='cent-dist-b')


# Top 100 eigenvector centrality scores

centDfec = centDf.sort_values('eigenvector', ascending = False)[:100]

trace = go.Bar(x = centDfec.index, y = centDfec['eigenvector'])
data = [trace]

layout = go.Layout(yaxis=dict(title='Eigenvector Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='cent-dist-e')


# Scatterplots

trace = go.Scatter(x = centDf['degree'], y = centDf['betweenness'], mode = 'markers')
data = [trace]

layout = go.Layout(xaxis=dict(title='Degree Centrality',), yaxis=dict(title='Betweenness Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='centralities-scatter')


# Static Scatterplots
with sns.axes_style('white'):
    sns.jointplot(x='degree', y='eigenvector', data=centDf, xlim = (0, .1), ylim = (0, .7), color = 'gray')
    sns.despine()
plt.savefig('figures/cent_scatterplot.png')
plt.savefig('figures/cent_scatterplot.p')

# Visualizing Networks

eig = nx.eigenvector_centrality(giant_coauth)
size = [2000 * eig[node] for node in giant_coauth]

nx.draw_spring(giant_coauth, node_size = size, with_labels = True, font_size = 5, node_color = "#FFFFFF", edge_color = "#D4D5CE", alpha = .95)
plt.savefig('figures/network_coauthors.png')
plt.savefig('figures/network_coauthors.pdf')

# Community Detection

partition = community.best_partition(giant_coauth)
modularity = community.modularity(partition, giant_coauth) 
print('Modularity:', modularity)
Modularity: 0.8385764300280952

colors = [partition[n] for n in giant_coauth.nodes()] 
palette = plt.cm.Set2 # you can select other color pallettes here: https://matplotlib.org/users/colormaps.html
nx.draw(giant_coauth, node_color=colors, cmap = palette, edge_color = "#D4D5CE")
plt.savefig('figures/coauthors_community.png')
plt.savefig('figures/coauthors_community.pdf')

# Co-Citation Networks
journalCoc = rcSp.networkCoCitation(coreOnly = True) 
mk.dropEdges(journalCoc , minWeight = 3)
print(mk.graphStats(journalCoc))

# visualize the giant component only
giantJournal = max(nx.connected_component_subgraphs(journalCoc), key=len)
nx.draw_spring(giantJournal, with_labels = False, node_size = 75, node_color = "#77787B", edge_color = "#D4D5CE", alpha = .95)
plt.savefig('figures/network_journal_cocite.png')
plt.savefig('figures/network_journal_cocite.pdf')

# Co-Citation Network
partition = community.best_partition(giantJournal) 
modularity = community.modularity(partition, giantJournal) 
print('Modularity:', modularity)

colors = [partition[n] for n in giantJournal.nodes()] 
nx.draw_spring(giantJournal, node_color=colors, with_labels = False, cmap=plt.cm.tab10, node_size = 100, edge_color = "#D4D5CE")
plt.savefig('figures/network_journal_cocite_community.png')
plt.savefig('figures/network_journal_cocite_community.pdf')

nserc_grants = mk.GrantCollection('raw_data/grants/nserc/')
print('There are', len(nserc_grants), 'Grants in this Grant Collection.')
ciNets = nserc_grants.networkCoInvestigator()
print(mk.graphStats(ciNets))

mk.dropEdges(ciNets, minWeight = 4)
giant_ci = max(nx.connected_component_subgraphs(ciNets), key=len)
print(mk.graphStats(giant_ci))

partitionCi = community.best_partition(giant_ci) 
modularity_ci = community.modularity(partitionCi, giant_ci) 
print('Modularity:', modularity_ci)

colorsCi = [partitionCi[n] for n in giant_ci.nodes()] 
nx.draw_spring(giant_ci, node_color=colorsCi, with_labels = False, cmap=plt.cm.tab10, node_size = 100, edge_color = "#D4D5CE")
plt.savefig('figures/network_coinvestigators.png')
plt.savefig('figures/network_coinvestigators.pdf')

# Betweenness centrality scores
bet = nx.betweenness_centrality(giant_ci)
betDf = pandas.DataFrame.from_dict([bet]).transpose()
betDf.columns = ['betweenness']
betDf.sort_values(by = ['betweenness'], ascending = False)[:10]
betweenness

topbet_nserc = betDf.sort_values('betweenness', ascending = False)[:100]

trace = go.Bar(x = topbet_nserc.index, y = topbet_nserc['betweenness'])
data = [trace]

layout = go.Layout(yaxis=dict(title='Betweenness Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='betweenness_nserc')

# Institution-Level Co-Investigator Networks
inst = nserc_grants.networkCoInvestigatorInstitution() 
print(mk.graphStats(inst))

degInst = nx.degree_centrality(inst)
degInstDf = pandas.DataFrame.from_dict([degInst]).transpose()
degInstDf.columns = ['Degree Centrality']
degInstDf.sort_values(by = ['Degree Centrality'], ascending = False)[:15]

# Degree Centrality
instCent = degInstDf.sort_values('Degree Centrality', ascending = False)[:100]
trace = go.Bar(x = instCent.index, y = instCent['Degree Centrality'])
data = [trace]

layout = go.Layout(yaxis=dict(title='Degree Centrality',))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='degree_nserc_inst')

# Eigenvector Centrality
eigInst = nx.eigenvector_centrality(inst)
eigInstDf = pandas.DataFrame.from_dict([eigInst]).transpose()
eigInstDf.columns = ['Eigenvector Centrality']
eigInstDf.sort_values(by = ['Eigenvector Centrality'], ascending = False)[:15]

mk.writeGraph(inst , 'generated_datasets/institutional_collaboration_network/')
nx.write_graphml(inst, 'generated_datasets/institutional_collaboration_network/inst_network.graphml')

