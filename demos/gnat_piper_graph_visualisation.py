import networkx as nx
import matplotlib.pyplot as plt


def run_gnat_piper_graph_visualisation():
    print('Running gnat_piper_graph_visualisation.py')

    # Gnat with boundary node
    edges_gnat = [(1,2),(1,3),(1,4),(1,5),(2,7),(2,3),(3,7),(3,4),(4,7),(4,5),(5,6),(6,7),(7,10),(6,10),(10,9),(9,8)]
    labels_gnat = {1:'F',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6',8:'7',9:'8',10:'9'}
    G_gnat = nx.Graph()
    G_gnat.add_edges_from(edges_gnat)

    # Piper with boundary node
    edges_piper = [(1,2),(2,3),(3,4),(4,5),(5,6)]
    labels_piper = {1:'M',2:'1',3:'2',4:'3',5:'4',6:'5'}
    G_piper = nx.Graph()
    G_piper.add_edges_from(edges_piper)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    nx.draw(G_gnat, with_labels=True, labels=labels_gnat, node_color='lightblue', edge_color='gray')
    plt.title('Gnat with boundary')

    plt.subplot(1,2,2)
    nx.draw(G_piper, with_labels=True, labels=labels_piper, node_color='lightgreen', edge_color='gray')
    plt.title('Piper with boundary')

    plt.show()

    # path computation
    paths6 = []
    for target in range(2, 11):
        for path in nx.all_simple_paths(G_gnat, source=1, target=target, cutoff=5):
            if len(path) == 6:
                paths6.append(path)

    print(f'Found {len(paths6)} boundary-including paths of length 5 edges.')

    return {'paths6': paths6}


if __name__ == '__main__':
    run_gnat_piper_graph_visualisation()
