#from sklearn.neighbors import NearestNeighbors
# from evaluate import map_for_dataset
from data_IGN import  read_data
import random

def arr2str(arr):
    result = ""
    for i in arr:
        result += " "+str(i)
    return result

def randomWalk(G, walkSize):
    walkList= []
    curNode = random.choice(G.nodes())

    while(len(walkList) < walkSize):
        walkList.append(G.node[curNode]['nature']) # each node has 5 attributes, I don't use edges so far
        walkList.append(G.node[curNode]['normed_length'])
        walkList.append(G.node[curNode]['curvature_bin1'])
        walkList.append(G.node[curNode]['curvature_bin2'])
        walkList.append(G.node[curNode]['curvature_bin3'])
        # get a new node
        curNode = random.choice(G.neighbors(curNode))
    return walkList

def generateWalkFile(dirName, walkLengthm, graphs_list, walkLength):
    walkFile = open(dirName + '.walk', 'w')
    indexToName = {}

    for graph in graphs_list:
        index = 0
        subgraph = graph # get the graph
        walk = randomWalk(subgraph, walkLength)
        walkFile.write(arr2str(walk) + "\n")
        indexToName[index] = name
        index += 1
    walkFile.close()

    return indexToName

symmetric_dataset = True
if __name__ == '__main__':
    name ='IGN04'
    IGN04 = read_data(name,
        with_classes=False,
        prefer_attr_nodes=True,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=symmetric_dataset, path = 'D:/projects/FGW/data/IGN/2004/')
    name = 'IGN19'
    IGN19 = read_data(name,
                      with_classes=False,
                      prefer_attr_nodes=True,
                      prefer_attr_edges=False,
                      produce_labels_nodes=False,
                      as_graphs=False,
                      is_symmetric=symmetric_dataset, path='D:/projects/FGW/data/IGN/2019/')

    n_samples = 6000
    data_base = IGN04.data[:n_samples]
    data_query = IGN19.data[:n_samples]



    neighborhood_embedding(args)

    #
    # knn_array = []
    # dist_array = []
    #
    # gt_indexes = list(range(len(data_query)))
    # n = 5
    # #calculate now the map@5
    # for i in range(len(query)):
    #     indexes = (-query[i]).argsort()[:n]
    #     knn_array.append(indexes)  # workaround for structure
    # print("total of %d embeddings were processed" % (i + 1))
    # map = map_for_dataset(gt_indexes, knn_array, dist_array)
    # print("Final MAP for the data %s of %d graphs is %f" % ('emb', len(data_base), map))
    #
