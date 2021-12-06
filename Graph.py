from my_exception import *

class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}
        self.direction = {}

    def addNeighbor(self,nbr,direction,weight=0):
        self.connectedTo[nbr] = (direction, weight)
        self.direction[direction] = nbr
    
    def delNeighbor(self,nbr):
        _ = self.direction.pop(self.connectedTo[nbr][0])
        _ = self.connectedTo.pop(nbr)

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([str(x) + ' direction: ' + self.connectedTo[x][0] for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]
    
    def getDirection(self):
        return self.direction.keys()
    
    def getNeighbor(self, d):
        return self.direction[d]

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        self.edgeList = []

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,direction,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(t, direction, cost)

    def delEdge(self,f,t):
        try:
            if f not in self.vertList:
                raise NodeNotExistError(f)
            if t not in self.vertList[f].getConnections():
                raise EdgeNotExistError(f,t)
        except NodeNotExistError as e:
            print(e.info)
        except EdgeNotExistError as e:
            print(e.info)
        else:
            self.vertList[f].delNeighbor(t)
        
    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

class Rect_Graph(Graph):
    def __init__(self, l, w):
        super().__init__()
        self.l = l
        self.w = w
        self.gen_node_and_edge()

    def gen_node_and_edge(self):
        for i in range(self.l*self.w):
            if i >= self.l:
                self.addEdge(i, i-self.l, 's', 1)
            if i < (self.w-1)*self.l:
                self.addEdge(i, i+self.l, 'n', 1)
            if i % self.l != 0:
                self.addEdge(i, i-1, 'w', 1)
            if i % self.l != self.l-1:
                self.addEdge(i, i+1, 'e', 1)
    

if __name__ == '__main__':
    # size = 4
    # g = Graph()
    # for i in range(size*size):
    #     if i >= size:
    #         g.addEdge(i, i-4, 1)
    #     if i < 3*size:
    #         g.addEdge(i, i+4, 1)
    #     if i % size != 0:
    #         g.addEdge(i, i-1, 1)
    #     if i % size != size-1:
    #         g.addEdge(i, i+1, 1)
    # g.delEdge(0,4)
    # g.delEdge(4,0)
    # for i in g.getVertices():
    #     print(g.getVertex(i))
    g = Rect_Graph(3,2)
    g.delEdge(3,4)
    for i in g.getVertices():
        print(g.getVertex(i))
    for i in g.getVertices():
        v = g.getVertex(i)
        for j in v.direction.keys():
            print(j + str(v.direction[j]))