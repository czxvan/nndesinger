import sys
from PyQt5.QtGui import QKeyEvent, QFont, QMouseEvent, QPainter, QPainterPath
from PyQt5.QtGui import  QColor, QLinearGradient, QBrush, QPen
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneHoverEvent, QGraphicsView, QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem, QStyle
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtWidgets import QMainWindow

from edge import Edge
from scene import GraphicScene
from view import DataflowView
 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = GraphicScene(self)
        self.view = DataflowView(self.scene)
        # 设置view可以进行鼠标的拖拽选择
        self.view.setDragMode(self.view.RubberBandDrag)
 
        self.setMinimumHeight(800)
        self.setMinimumWidth(800)
        self.setCentralWidget(self.view)
        self.setWindowTitle("Graphics Demo")
 
def demo_run():
    app = QApplication(sys.argv)
    demo = MainWindow()
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    font = QFont()
    font.setFamily("SimHei")
    font.setPixelSize(14)
    app.setFont(font)
    demo.show()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    import sys
    # sys.setrecursionlimit(40)
    demo_run()
 