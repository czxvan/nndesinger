import math

from PyQt5.QtWidgets import QGraphicsItem, QGraphicsPathItem
from PyQt5.QtGui import QColor, QPen, QPainterPath, QBrush, QPolygon
from PyQt5.QtCore import Qt, QPointF, QPoint
import numpy as np


class GraphicEdge(QGraphicsPathItem):
    def __init__(self, edge_wrap, parent=None):
        super().__init__(parent)
        # 这个参数是GraphicEdge的包装类
        self.edge_wrap = edge_wrap
        self.width = 3.0  # 线条的宽度
        self.pos_src = [0, 0]  # 线条起始位置 x，y坐标
        self.pos_dst = [0, 0]  # 线条结束位置

        self._pen = QPen(QColor("#000"))  # 画线条的
        self._pen.setWidthF(self.width)

        self._pen_dragging = QPen(QColor("#000"))  # 画拖拽线条时线条的
        self._pen_dragging.setStyle(Qt.DashDotLine)
        self._pen_dragging.setWidthF(self.width)

        self.setFlag(QGraphicsItem.ItemIsSelectable)  # 线条可选
        self.setZValue(-1)  # 让线条出现在所有图元的最下层

        # 为连线添加箭头
        self._mark_pen = QPen(Qt.black)
        self._mark_pen.setWidthF(self.width)
        self._mark_brush = QBrush()
        self._mark_brush.setColor(Qt.black)
        self._mark_brush.setStyle(Qt.SolidPattern)

    def setStartPosition(self, x, y):
        self.pos_src = [x, y]

    def setEndPosition(self, x, y):
        self.pos_dst = [x, y]

    # calculate line path
    def calcPath(self):
        path = QPainterPath(QPointF(self.pos_src[0], self.pos_src[1]))  # 起点
        path.lineTo(self.pos_dst[0], self.pos_dst[1])  # 终点
        return path

    # override
    def paint(self, painter, graphics_item, widget=None):
        self.setPath(self.calcPath())  # 设置路径
        path = self.path()
        if self.edge_wrap.end_item is None:
            # 包装类中存储了线条开始和结束位置的图元
            # 刚开始拖拽线条时，并没有结束位置的图元，所以是None
            # 这个线条画的是拖拽路径，点线
            painter.setPen(self._pen_dragging)
            painter.drawPath(path)
        else:
            x1, y1 = self.pos_src
            x2, y2 = self.pos_dst
            length = 0.5  # 圆点距离终点图元的距离
            k = math.atan2(y2 - y1, x2 - x1)  # theta
            new_x = x2 - length * math.cos(k)  # 减去线条自身的宽度
            new_y = y2 - length * math.sin(k)
            new_x1 = new_x - 20 * math.cos(k - np.pi / 6)
            new_y1 = new_y - 20 * math.sin(k - np.pi / 6)
            new_x2 = new_x - 20 * math.cos(k + np.pi / 6)
            new_y2 = new_y - 20 * math.sin(k + np.pi / 6)
            # 先画最终路径
            painter.setPen(self._pen)
            painter.drawPath(path)
            # 再圆点
            painter.setPen(self._mark_pen)
            painter.setBrush(self._mark_brush)
            # 将坐标点转为int
            new_x, new_y = int(new_x), int(new_y)
            new_x1, new_y1 = int(new_x1), int(new_y1)
            new_x2, new_y2 = int(new_x2), int(new_y2)
            point1 = QPoint(new_x, new_y)
            point2 = QPoint(new_x1, new_y1)
            point3 = QPoint(new_x2, new_y2)
            # 连接圆点，形成箭头
            painter.drawPolygon(point1, point2, point3)


class Edge:
    def __init__(self, scene, start_item, end_item):
        # 参数分别为场景、开始图元、结束图元
        super().__init__()
        self.scene = scene
        self.start_item = start_item
        self.end_item = end_item
        # 线条图形在此处创建
        self.gr_edge = GraphicEdge(self)
        # 此类一旦被初始化就在添加进scene
        self.scene.addEdge(self.gr_edge)
        # 开始更新
        if self.start_item is not None:
            self.updatePositions()

    # 最终保存进scene
    def store(self):
        self.scene.addEdge(self.gr_edge)

    # 更新位置
    def updatePositions(self):
        # src_pos 记录的是开始图元的位置，此位置为图元的中心
        src_pos = self.start_item.getCenterPos()
        self.gr_edge.setStartPosition(src_pos.x(), src_pos.y())
        # 如果结束位置图元也存在，则做同样操作
        if self.end_item is not None:
            end_pos = self.end_item.getCenterPos()
            self.gr_edge.setEndPosition(end_pos.x(), end_pos.y())
        else:
            self.gr_edge.setEndPosition(src_pos.x(), src_pos.y())
        self.gr_edge.update()

    def removeFromCurrentItems(self):
        self.end_item = None
        self.start_item = None

    # 移除线条
    def remove(self):
        self.removeFromCurrentItems()
        self.scene.removeEdge(self.gr_edge)
        self.gr_edge = None

