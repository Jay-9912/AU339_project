

class NodeNotExistError(RuntimeError): # 节点不在图中
    def __init__(self, id):
        self.info = "Node %d does not exist!" %(id)

class EdgeNotExistError(RuntimeError): # 边不在图中
    def __init__(self, f, t):
        self.info = "Edge from %d to %d does not exist!" %(f, t)