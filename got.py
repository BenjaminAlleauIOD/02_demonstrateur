import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import streamlit as st


def got_func(physics):
  got_net = Network(height="600px", width="100%", font_color="black",heading='Game of Thrones Graph')

# set the physics layout of the network
  got_net.barnes_hut()
  got_data = pd.read_csv("https://www.macalester.edu/~abeverid/data/stormofswords.csv")
  #got_data = pd.read_csv("stormofswords.csv")
  #got_data.rename(index={0: "Source", 1: "Target", 2: "Weight"}) 
  sources = got_data['Source']
  targets = got_data['Target']
  weights = got_data['Weight']

  edge_data = zip(sources, targets, weights)

  for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]

    got_net.add_node(src, src, title=src)
    got_net.add_node(dst, dst, title=dst)
    got_net.add_edge(src, dst, value=w)

  neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
  for node in got_net.nodes:
    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
    node["value"] = len(neighbor_map[node["id"]])
  if physics:
    got_net.show_buttons(filter_=['physics'])
  got_net.show("gameofthrones.html")
  

import streamlit as st
from pyvis.network import Network

def simple_func(df): 
    nx_graph = nx.DiGraph()
    i=1
    for index, row in df.head(6).iterrows():
        node_label = f"Risk Score : {row['Risk Score']}\n Revenue : {row['Revenue']} M€\n D1 : {row['Delivery Time']}%"
        nx_graph.add_node(row['Supplier'], label=row['Supplier'],title=node_label,group=i)
        i = i+1

    
    # Ajout des nœuds (fournisseurs)
    # nx_graph.add_node("f1", label="f1")
    # nx_graph.add_node("f2", label="f2")
    # nx_graph.add_node("f3", label="f3")
    # nx_graph.add_node("f4", label="f4")
    # nx_graph.add_node("f5", label="f5")
    # nx_graph.add_node("f6", label="f6")

    # Ajout des arêtes (relations)
    # nx_graph.add_edge("F 1", "F 2")
    # nx_graph.add_edge("F 1", "F 3")
    # nx_graph.add_edge("F 2", "F 3")
    # nx_graph.add_edge("F 2", "F 4")
    # nx_graph.add_edge("F 3", "F 4")
    # nx_graph.add_edge("F 3", "F 5")
    # nx_graph.add_edge("F 4", "F 5")
    # nx_graph.add_edge("F 5", "F 6")


    # Ajout des arêtes avec les attributs "w" contenant les valeurs correspondantes
    edge_list = [("F 1", "F 2", {'w': 'A1'}),
                ("F 2", "F 3", {'w': 'B'}),
                ("F 3", "F 1", {'w': 'C'}),
                ("F 4", "F 5", {'w': 'D2'}),
                ("F 3", "F 5", {'w': 'F'}),
                ("F 5", "F 4", {'w': 'G'})]

    nx_graph.add_edges_from(edge_list)

    # Calcul du PageRank
    pr = nx.pagerank(nx_graph)

    # Création du DataFrame
    pr_df = pd.DataFrame.from_dict(pr, orient='index', columns=['PageRank'])
    pr_df.index.name = 'Fournisseur'


    nt = Network("500px", "500px",notebook=True,heading='',directed=True)
    nt.from_nx(nx_graph)
    #physics=st.sidebar.checkbox('add physics interactivity?')

    nt.show('test.html')
    # Affichage du DataFrame du PageRank
    return pr_df#st.dataframe(pr_df)



# # Application Streamlit
# supply_chain_network()



def karate_func(physics): 
  G = nx.karate_club_graph()


  nt = Network("500px", "500px",notebook=True,heading='Zachary’s Karate Club graph')
  nt.from_nx(G)
  #physics=st.sidebar.checkbox('add physics interactivity?')
  if physics:
    nt.show_buttons(filter_=['physics'])
  nt.show('karate.html')