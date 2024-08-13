from PyQt5.QtWidgets import QGraphicsScene


class GraphicScene(QGraphicsScene):
    def __init__(self, parent=None):
        # 在原init函数内增加2行
        super().__init__(parent)
        self.nodes = []  # 存储图元
        self.edges = []  # 存储连线

    def addNode(self, node):
        self.nodes.append(node)
        self.addItem(node)

    def removeNode(self, node):
        # remove all info about this node
        self.nodes.remove(node)
        edges_to_delete = []
        # 删除图元时，遍历与其连接的线，并移除
        for edge in self.edges:
            if edge.edge_wrap.start_item is node or edge.edge_wrap.end_item is node:
                edges_to_delete.append(edge)
        for edge in edges_to_delete:
            self.removeEdge(edge)
        self.removeItem(node)

        # cur_num = storex.getValue('current_dataflow')
        # store_model.deleteItem(cur_num, node.uuid)

    def addEdge(self, edge):
        self.edges.append(edge)
        self.addItem(edge)

    def removeEdge(self, edge):
        self.edges.remove(edge)
        if edge not in self.items():
            return
        self.removeItem(edge)

