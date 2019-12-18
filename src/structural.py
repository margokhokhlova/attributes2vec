import gensim.models.doc2vec as doc
import os
import graphUtils_s
import random
import networkx as nx


def arr2str(arr):
    result = ""
    for i in arr:
        result += " "+str(i)
    return result


def generateDegreeWalk(Graph, walkSize):
    g = Graph
    walk = randomWalkDegreeLabels(g,walkSize)
    #walk = serializeEdge(g,NodeToLables)
    return walk

def randomWalkDegreeLabels(G, walkSize):
    curNode = random.choice(G.nodes())
    walkList= []

    while(len(walkList) < walkSize):
        walkList.append(G.node[curNode]['label'])
        curNode = random.choice(G.neighbors(curNode))
    return walkList

def getDegreeLabelledGraph(G, rangetoLabels, degreeFile):    
    degreeDict = G.degree(G.nodes())
    labelDict = {}
    for node in degreeDict.keys():
        val = degreeDict[node]#/float(nx.number_of_nodes(G))
        degreeFile.write(str(val)+" ")
        labelDict[node] = inRange(rangetoLabels, val)
        nx.set_node_attributes(G, 'label', labelDict)

    return G

def inRange(rangeDict, val):
        for key in rangeDict:
            if key[0] < val and key[1] >= val:
                return rangeDict[key]

def generateWalkFile(dirName, walkLength, alpha):
    walkFile = open(dirName+'.walk', 'w')  
    degreeFile = open(dirName+'.degree_ratio', 'w+')  
    indexToName = {}
    rangetoLabels = {(0, 0.05):'z',(0.05, 0.1):'a', (0.1, 0.15):'b', (0.15, 0.2):'c', (0.2, 0.25):'d', (0.25, 0.5):'e', (0.5, 0.75):'f',(0.75, 1.0):'g'}
    # rangetoLabels = {(0, 1):'z',(1, 2):'a', (2, 3):'b', (3, 4):'c', (4, 5):'d', (6, 7):'e', (7, 8):'f',(8, 100):'g'}
    
    # rangetoLabels = {(0,0.00242131):'a', (0.00242131,0.03090626):'b', (0.03090626,0.05939121) :'c', (0.05939121, 0.08787617):'d',
    #         ( 0.08787617, 0.11636112): 'e',(0.11636112, 0.14484607):'f',  (0.14484607, 0.17333103):'g',  (0.17333103,0.20181598): 'h', (0.20181598, 0.23030093):'i',
    #         (0.23030093, 0.25878589) :'j', (0.25878589, 0.28727084):'k', ( 0.28727084, 0.31575579): 'l', (0.31575579, 0.34424075): 'm', (0.34424075, 0.3727257): 'n',
    #         (0.3727257 , 0.40121065): 'o', (0.40121065, 0.42969561):'p', (0.42969561,0.45818056):'r', (0.45818056, 0.48666551):'s', (0.48666551,0.51515047):'t', 
    #         ( 0.51515047, 0.54363542):'u', ( 0.54363542, 0.57212037):'v', (0.57212037, 0.60060533):'w', (0.60060533,0.62909028):'x', (0.62909028, 1.0):'y' }
    # rangetoLabels = {(0, 0.00724638): 'a', (0.00724638,0.13985507):'b', (0.13985507,0.27246377): 'c', (0.27246377,0.40507246):'d', 
    #         (0.40507246,0.53768116):'e', ( 0.53768116,0.67028986):'f',  (0.67028986, 0.80289855):'g', (0.80289855, 0.93550725):'h',
    #         (0.93550725,1.06811594):'j', (1.06811594, 1.20072464):'l',( 1.20072464, 1.5):'m'}
    
    for  root, dirs, files in os.walk(dirName):
        index = 0
        for name in files:
            print(name)
            subgraph = graphUtils_s.getGraph(os.path.join(root, name))
            degreeGraph = getDegreeLabelledGraph(subgraph, rangetoLabels, degreeFile)
            degreeWalk = generateDegreeWalk(degreeGraph, int(walkLength* (1- alpha)))
            walk = graphUtils_s.randomWalk(subgraph, int(alpha * walkLength))
            walkFile.write(arr2str(walk)+ arr2str(degreeWalk) +"\n")
            indexToName[index] = name
            index += 1
    walkFile.close()
    degreeFile .close()

    return indexToName

def saveVectors(vectors, outputfile, IdToName):
    output = open(outputfile, 'w')

    output.write(str(len(vectors)) +"\n")
    for i in range(len(vectors)):
        output.write(str(IdToName[i]))
        for j in vectors[i]:
            output.write('\t'+ str(j))
        output.write('\n')
    output.close()

def saveVectors_csv(vectors, outputfile, IdToName):
    csv = open(outputfile+'.csv', 'w')
    for i in range(len(vectors)):
        csv.write("{},{}\n".format(IdToName[i],vectors[i:]))
    csv.close()

def structural_embedding(args):

    inputDir = args.input
    print(inputDir)
    outputFile = args.output
    iterations = args.iter
    dimensions = args.d
    window = args.windowSize
    dm = 1 if args.model == 'dm' else 0
    indexToName = generateWalkFile(inputDir, args.walkLength, args.p) # just makes walks 
    sentences = doc.TaggedLineDocument(inputDir+'.walk')

    model = doc.Doc2Vec(sentences, vector_size = dimensions, epochs = iterations, dm = dm, window = window )
    print("Total vects ", len(list(model.docvecs.vectors_docs))) #model.docvecs
    saveVectors(list(model.docvecs.vectors_docs), outputFile, indexToName)



