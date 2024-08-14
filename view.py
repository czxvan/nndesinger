import math

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsView, QGraphicsItem, QGraphicsPixmapItem, QMessageBox

from edge import Edge
from scene import GraphicScene
from item import getLayer

class DataflowView(QGraphicsView):
    def __init__(self, graphic_scene=None, parent=None):
        super().__init__(parent)
        self.drag_start_item = None
        self.edge_enable = False  # 用来记录目前是否可以画线条
        self.drag_edge = None  # 记录拖拽时的线
        self.pre_position = None  # 用于存储edge更新之前的箭头坐标
        if graphic_scene is not None:
            self.gr_scene = graphic_scene  # 将scene传入此处托管，方便在view中维护
        else:
            self.gr_scene = GraphicScene()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setScene(self.gr_scene)
        # 设置渲染属性
        self.setRenderHints(QPainter.Antialiasing |  # 抗锯齿
                            QPainter.HighQualityAntialiasing |  # 高品质抗锯齿
                            QPainter.TextAntialiasing |  # 文字抗锯齿
                            QPainter.SmoothPixmapTransform |  # 使图元变换更加平滑
                            QPainter.LosslessImageRendering)  # 不失真的图片渲染
        # 视窗更新模式
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        # 设置水平和竖直方向的滚动条不显示
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        # 设置拖拽模式
        self.setDragMode(self.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setStyleSheet("border: none;")

    def keyPressEvent(self, event):
        # 当按下键盘E键时，启动线条功能，再次按下则是关闭
        if event.key() == Qt.Key_E:
            self.edge_enable = ~self.edge_enable
        elif event.key() == Qt.Key_N:
            item = getLayer('linear')
            self.gr_scene.addNode(item)

    def edgeDragStart(self, item):
        self.drag_start_item = item  # 拖拽开始时的图元
        self.drag_edge = Edge(self.gr_scene, self.drag_start_item, None)  # 开始拖拽线条，注意到拖拽终点为None

    def edgeDragEnd(self, item):
        # 拖拽结束
        Edge(self.gr_scene, self.drag_start_item, item)
        # 更新layer的in_list与out_list
        self.drag_start_item.out_list.append(item.uuid)
        item.in_list.append(self.drag_start_item.uuid)
        # 判断drag不为空
        if self.drag_edge is None:
            return
        # 删除拖拽时画的线
        self.drag_edge.remove()
        self.drag_edge = None

    def mousePressEvent(self, event):
        item = self.getItemAtClick(event)
        # 单击鼠标右键删除layer
        if event.button() == Qt.RightButton:
            if isinstance(item, QGraphicsPixmapItem):
                self.gr_scene.removeNode(item)
        # 按住鼠标左键开始拖拽生成edge
        elif self.edge_enable:
            if isinstance(item, QGraphicsPixmapItem):
                # 确认起点是图元后，开始拖拽
                self.edgeDragStart(item)
        else:
            # 如果写到最开头，则线条拖拽功能会不起作用
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.edge_enable:
            try:
                # 拖拽结束后，关闭连线功能
                self.edge_enable = False
                item = self.getItemAtClick(event)
                # 终点图元不能是起点图元
                if isinstance(item, QGraphicsPixmapItem) and item is not self.drag_start_item:
                    self.edgeDragEnd(item)
                    print("edge create success")
                else:
                    if self.drag_edge is not None:
                        self.drag_edge.remove()
                        self.drag_edge = None
                    print("edge create failed")
            except Exception as error:
                print(error)
                QMessageBox.warning(self, '提示', '连接失败，请重试！')
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.edge_enable and self.drag_edge is not None:
            pos = event.pos()
            sc_pos = self.mapToScene(pos)
            self.drag_edge.gr_edge.setEndPosition(sc_pos.x(), sc_pos.y())
            self.drag_edge.gr_edge.update()
        super().mouseMoveEvent(event)

    def getItemsAtRubberSelect(self):
        area = self.rubberBandRect()
        return self.items(area)  # 返回一个所有选中图元的列表，对此操作即可

    def getItemAtClick(self, event: QEvent) -> 'QGraphicsItem':
        """返回我们点击/释放鼠标按钮的对象"""
        pos = event.pos()
        obj = self.itemAt(pos)
        return obj
