#######################
# Trabalho 2          #
# Grafos e Aplicações #
# Bionda Rozin        #
#######################

# Imports
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

df = pd.read_csv("Data\\plant_pollinator_database.csv")
plants = df["crop"].unique()
pollinators = df["visitor"].unique()

G = nx.Graph()

G.add_nodes_from(plants)
G.add_nodes_from(pollinators)

for _, rows in df.iterrows():
    G.add_edge(rows["crop"], rows["visitor"])

nx.draw_networkx_nodes(G, nodelist=plants, node_color="green", label="Plants", pos=nx.kamada_kawai_layout(G), node_size=50, cmap=plt.get_cmap('jet'))
nx.draw_networkx_nodes(G, nodelist=pollinators, node_color="brown", label="Pollinators", pos=nx.kamada_kawai_layout(G), node_size=50, cmap=plt.get_cmap('jet'))
nx.draw_networkx_edges(G, pos=nx.kamada_kawai_layout(G))
plt.legend(scatterpoints = 1)
plt.draw()
plt.savefig("Figs\\eco network.pdf")