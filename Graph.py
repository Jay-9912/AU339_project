from my_exception import *

class Vertex: # 节点类
    def __init__(self,key):
        self.id = key
        self.connectedTo = {} # key: vertex id, value: (方向，边权)
        self.direction = {} # key: 方向， value: vertex id

    def addNeighbor(self,nbr,direction,weight=0): # 添加邻居节点
        self.connectedTo[nbr] = (direction, weight)
        self.direction[direction] = nbr
     
    def delNeighbor(self,nbr):  # 删除邻居节点
        _ = self.direction.pop(self.connectedTo[nbr][0])
        _ = self.connectedTo.pop(nbr)

    def __str__(self): # 打印信息
        return str(self.id) + ' connectedTo: ' + str([str(x) + ' direction: ' + self.connectedTo[x][0] for x in self.connectedTo])

    def getConnections(self): # 获取所有邻居的vertex id
        return self.connectedTo.keys()

    def getId(self): # 获取vertex id
        return self.id

    def getInfo(self,nbr): # 获取邻居节点nbr的信息
        return self.connectedTo[nbr]
    
    def getDirection(self): # 获取当前节点对应的可行方向
        return self.direction.keys()
    
    def getNeighbor(self, d): # 获取某方相对应的邻居的vertex id
        return self.direction[d]

class Graph: # 图类
    def __init__(self):
        self.vertList = {} # 所有节点，key: vertex id , value: 节点类
        self.numVertices = 0 # 总的节点个数
        self.edgeList = [] # 边集

    def addVertex(self,key): # 添加新节点
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n): # 获得节点n
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n): # 查询节点n是否存在
        return n in self.vertList

    def addEdge(self,f,t,direction,cost=0): # 添加有向边(f,t)
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(t, direction, cost)

    def delEdge(self,f,t): # 删除有向边(f,t)
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
        
    def getVertices(self): # 获得所有vertex id
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

class Rect_Graph(Graph): # 矩形图，Graph的派生类
    def __init__(self, l, w):
        super().__init__()
        self.l = l # 长
        self.w = w # 宽
        self.gen_node_and_edge() # 产生网格图

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