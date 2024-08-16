from PyQt5.QtCore import QSize, QRectF, Qt
from PyQt5.QtGui import QPainterPath, QBrush, QPen, QFont, QTextOption
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsTextItem, QStyle
from torch import nn

# from store import property as store_property
# from store import storex

class TextItem(QGraphicsTextItem):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        if len(text) > 9:
            self.setFont(QFont('Arial', 10))
        elif len(text) > 6:
            self.setFont(QFont('Arial', 11))
        else:
            self.setFont(QFont('Arial', 12))
        self.setDefaultTextColor(Qt.white)
        self.setTextWidth(parent.boundingRect().width())

        parent_rect = parent.boundingRect()
        text_rect = self.boundingRect()
        self.setPos(parent.x() + (parent_rect.width() - text_rect.width()) / 2,
                    parent.y() + (parent_rect.height() - text_rect.height()) / 2)

        self.document().setDefaultTextOption(QTextOption(Qt.AlignCenter))

class BaseItem(QGraphicsRectItem):
    """
    set QGraphicsItem size and basic function.
    """

    def __init__(self, text):
        self.text = text
        self.width = 60
        self.height = 60
        self.corner_radius = 10
        self.text_padding = 2
        super(BaseItem, self).__init__(0, 0, self.width, self.height)

        self.uuid = None
        self.in_list = []
        self.out_list = []
        self.layer_property = {}

        self.default_pen = QPen(Qt.darkGray, 1)
        self.selected_pen = QPen(Qt.black, 1, Qt.DashLine)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptedMouseButtons(Qt.LeftButton)

        self.text_item = TextItem(self.text, self)

    def setUuid(self, uuid):
        self.uuid = uuid

    def setLayerName(self, name):
        self.layer_name = name

    def paint(self, painter, option, widget):
        path = QPainterPath()
        path.addRoundedRect(self.boundingRect(), self.corner_radius, self.corner_radius)
        painter.fillPath(path, QBrush(Qt.gray))
        if option.state & QStyle.State_Selected:
            painter.setPen(self.selected_pen)
        else:
            painter.setPen(self.default_pen)
        painter.drawPath(path)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # 如果图元被选中，更新全部的边
        # 可以优化为仅更新在位置移动的图元上的边
        if self.isSelected():
            for gr_edge in self.scene().edges:
                gr_edge.edge_wrap.updatePositions()

    # def mousePressEvent(self, event) -> None:
    #     super().mousePressEvent(event)
    #     storex.getValue('container_widget').property_widget.editProperty(self.layer_property)
    #     storex.setValue('selected_layer', self)
    #     for item in self.layer_property.keys():
    #         store_property.setValue(item, self.layer_property[item])


class DataloaderLayer(BaseItem):
    """
    dataloader layer is used to load dataset through file path. We can load different type dataset, such as image,
    text, data.
    """

    def __init__(self):
        super().__init__(text='特征输入层')
        self.layer_property = {
            'file_path': '',
            'type': '',
        }

"""
Convolution layer, which can be used in CNN or other deep learning model.
"""


class Convolution1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'dilation': '1',
            'groups': '1',
            'bias': 'True',
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.Conv1d(int(self.layer_property['in_channels']),
                         int(self.layer_property['out_channels']),
                         int(self.layer_property['kernel_size']),
                         int(self.layer_property['stride']),
                         int(self.layer_property['padding']),
                         int(self.layer_property['dilation']),
                         int(self.layer_property['groups'],
                             bias),
                         )


class Convolution2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'dilation': '1',
            'groups': '1',
            'bias': 'True',
            'padding_mode': 'zeros'
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.Conv2d(int(self.layer_property['in_channels']),
                         int(self.layer_property['out_channels']),
                         int(self.layer_property['kernel_size']),
                         int(self.layer_property['stride']),
                         int(self.layer_property['padding']),
                         int(self.layer_property['dilation']),
                         int(self.layer_property['groups']),
                         bias,
                         self.layer_property['padding_mode'],
                         )


class Convolution3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'dilation': '1',
            'groups': '1',
            'bias': 'True',
            'padding_mode': 'zeros'
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.Conv3d(int(self.layer_property['in_channels']),
                         int(self.layer_property['out_channels']),
                         int(self.layer_property['kernel_size']),
                         int(self.layer_property['stride']),
                         int(self.layer_property['padding']),
                         int(self.layer_property['dilation']),
                         int(self.layer_property['groups']),
                         bias,
                         self.layer_property['padding_mode'],
                         )


class TransposedConv1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维转置卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'output_padding': '0',
            'groups': '1',
            'bias': 'True',
            'dilation': '1',
            'padding_mode': 'zeros'
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.ConvTranspose1d(int(self.layer_property['in_channels']),
                                  int(self.layer_property['out_channels']),
                                  int(self.layer_property['kernel_size']),
                                  int(self.layer_property['stride']),
                                  int(self.layer_property['padding']),
                                  int(self.layer_property['output_padding']),
                                  int(self.layer_property['groups']),
                                  bias,
                                  int(self.layer_property['dilation']),
                                  self.layer_property['padding_mode'],
                                  )


class TransposedConv2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维转置卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'output_padding': '0',
            'groups': '1',
            'bias': 'True',
            'dilation': '1',
            'padding_mode': 'zeros'
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.ConvTranspose2d(int(self.layer_property['in_channels']),
                                  int(self.layer_property['out_channels']),
                                  int(self.layer_property['kernel_size']),
                                  int(self.layer_property['stride']),
                                  int(self.layer_property['padding']),
                                  int(self.layer_property['output_padding']),
                                  int(self.layer_property['groups']),
                                  bias,
                                  int(self.layer_property['dilation']),
                                  self.layer_property['padding_mode'],
                                  )


class TransposedConv3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维转置卷积层')
        self.layer_property = {
            'in_channels': '',
            'out_channels': '',
            'kernel_size': '',
            'stride': '1',
            'padding': '0',
            'out_padding': '0',
            'groups': '1',
            'bias': 'True',
            'dilation': '1',
            'padding_mode': 'zeros'
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.ConvTranspose3d(int(self.layer_property['in_channels']),
                                  int(self.layer_property['out_channels']),
                                  int(self.layer_property['kernel_size']),
                                  int(self.layer_property['stride']),
                                  int(self.layer_property['padding']),
                                  int(self.layer_property['out_padding']),
                                  int(self.layer_property['groups']),
                                  bias,
                                  int(self.layer_property['dilation']),
                                  self.layer_property['padding_mode'],
                                  )


"""
Sequence Layer.
"""


class RNNLayer(BaseItem):
    def __init__(self):
        super().__init__(text='循环神经网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'nonlinearity': 'tanh',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
        }
        self.bidirectional = False

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.RNN(int(self.layer_property['input_size']),
                      int(self.layer_property['hidden_size']),
                      int(self.layer_property['num_layers']),
                      self.layer_property['nonlinearity'],
                      bias,
                      batch_first,
                      float(self.layer_property['dropout']),
                      self.bidirectional,
                      )


class BiRNNLayer(BaseItem):
    def __init__(self):
        super().__init__(text='双向循环神经网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'nonlinearity': 'tanh',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
        }
        self.bidirectional = True

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.RNN(int(self.layer_property['input_size']),
                      int(self.layer_property['hidden_size']),
                      int(self.layer_property['num_layers']),
                      self.layer_property['nonlinearity'],
                      bias,
                      batch_first,
                      float(self.layer_property['dropout']),
                      self.bidirectional,
                      )


class LSTMLayer(BaseItem):
    def __init__(self):
        super().__init__(text='长短期记忆网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
            'proj_size': '0',
        }
        self.bidirectional = False

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.LSTM(int(self.layer_property['input_size']),
                       int(self.layer_property['hidden_size']),
                       int(self.layer_property['num_layers']),
                       bias,
                       batch_first,
                       float(self.layer_property['dropout']),
                       self.bidirectional,
                       int(self.layer_property['proj_size']),
                       )


class BiLSTMLayer(BaseItem):
    def __init__(self):
        super().__init__(text='双向长短期记忆网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
            'proj_size': '0',
        }
        self.bidirectional = True

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.LSTM(int(self.layer_property['input_size']),
                       int(self.layer_property['hidden_size']),
                       int(self.layer_property['num_layers']),
                       bias,
                       batch_first,
                       float(self.layer_property['dropout']),
                       self.bidirectional,
                       int(self.layer_property['proj_size']),
                       )


class GRULayer(BaseItem):
    def __init__(self):
        super().__init__(text='门控循环单元网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
            'proj_size': '0',
        }
        self.bidirectional = False

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.GRU(int(self.layer_property['input_size']),
                      int(self.layer_property['hidden_size']),
                      int(self.layer_property['num_layers']),
                      bias,
                      batch_first,
                      float(self.layer_property['dropout']),
                      self.bidirectional,
                      )


class BiGRULayer(BaseItem):
    def __init__(self):
        super().__init__(text='双向门控循环单元网络层')
        self.layer_property = {
            'input_size': '',
            'hidden_size': '',
            'num_layers': '1',
            'bias': 'True',
            'batch_first': 'True',
            'dropout': '0',
            'proj_size': '0',
        }
        self.bidirectional = True

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False

        if self.layer_property['batch_first'] == 'True':
            batch_first = True
        else:
            batch_first = False

        return nn.GRU(int(self.layer_property['input_size']),
                      int(self.layer_property['hidden_size']),
                      int(self.layer_property['num_layers']),
                      bias,
                      batch_first,
                      float(self.layer_property['dropout']),
                      self.bidirectional,
                      )


class FlattenLayer(BaseItem):
    def __init__(self):
        super().__init__(text='展平层')
        self.layer_property = {
            'start_dim': '1',
            'end_dim': '-1'
        }

    def generateModule(self):
        return nn.Flatten(int(self.layer_property['start_dim']),
                          int(self.layer_property['end_dim']))


"""
Pooling layer.
"""


class MaxPool1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维最大池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
        }

    def generateModule(self):
        return nn.MaxPool1d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']))


class MaxPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维最大池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
        }

    def generateModule(self):
        return nn.MaxPool2d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']))


class MaxPool3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维最大池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
        }

    def generateModule(self):
        return nn.MaxPool3d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']))


class AvgPool1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维平均池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
            'ceil_mode': 'False',
            'count_include_pad': 'True'
        }

    def generateModule(self):
        if self.layer_property['ceil_mode'] == 'True':
            ceil_mode = True
        else:
            ceil_mode = False

        if self.layer_property['count_include_pad'] == 'True':
            count_include_pad = True
        else:
            count_include_pad = False
        return nn.AvgPool1d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']),
                            ceil_mode,
                            count_include_pad)


class AvgPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维平均池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
            'ceil_mode': 'False',
            'count_include_pad': 'True'
        }

    def generateModule(self):
        if self.layer_property['ceil_mode'] == 'True':
            ceil_mode = True
        else:
            ceil_mode = False

        if self.layer_property['count_include_pad'] == 'True':
            count_include_pad = True
        else:
            count_include_pad = False
        return nn.AvgPool2d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']),
                            ceil_mode,
                            count_include_pad)


class AvgPool3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维平均池化层')
        self.layer_property = {
            'kernel_size': '',
            'stride': '',
            'padding': '0',
            'ceil_mode': 'False',
            'count_include_pad': 'True'
        }

    def generateModule(self):
        if self.layer_property['ceil_mode'] == 'True':
            ceil_mode = True
        else:
            ceil_mode = False

        if self.layer_property['count_include_pad'] == 'True':
            count_include_pad = True
        else:
            count_include_pad = False
        return nn.AvgPool3d(int(self.layer_property['kernel_size']),
                            int(self.layer_property['stride']),
                            int(self.layer_property['padding']),
                            ceil_mode,
                            count_include_pad)


class AdaptiveMaxPool1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应一维最大池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveMaxPool1d(int(self.layer_property['output_size']))


class AdaptiveMaxPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应二维最大池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveMaxPool2d(int(self.layer_property['output_size']))


class AdaptiveMaxPool3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应三维最大池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveMaxPool3d(int(self.layer_property['output_size']))


class AdaptiveAvgPool1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应一维平均池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveAvgPool1d(int(self.layer_property['output_size']))


class AdaptiveAvgPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应二维平均池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveAvgPool2d(int(self.layer_property['output_size']))


class AdaptiveAvgPool3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='自适应三维平均池化层')
        self.layer_property = {
            'output_size': '',
        }

    def generateModule(self):
        return nn.AdaptiveAvgPool3d(int(self.layer_property['output_size']))


class FractionalMaxPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维分数最大池化层')
        self.layer_property = {
            'kernel_size': '',
            'output_size': '',
            'output_ratio': '',
        }

    def generateModule(self):
        return nn.FractionalMaxPool2d(kernel_size=int(self.layer_property['kernel_size']),
                                      output_size=int(self.layer_property['output_size']),
                                      output_ratio=float(self.layer_property['output_ratio']))


class FractionalMaxPool3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维分数最大池化层')
        self.layer_property = {
            'kernel_size': '',
            'output_size': '',
            'output_ratio': '',
        }

    def generateModule(self):
        return nn.FractionalMaxPool3d(kernel_size=int(self.layer_property['kernel_size']),
                                      output_size=int(self.layer_property['output_size']),
                                      output_ratio=float(self.layer_property['output_ratio']))


class LpPool1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维Lp池化层')
        self.layer_property = {
            'norm_type': '',
            'kernel_size': '',
            'stride': '',
            'ceil_mode': 'False',
        }

    def generateModule(self):
        if self.layer_property['ceil_mode'] == "True":
            ceil_mode = True
        else:
            ceil_mode = False

        return nn.LPPool1d(int(self.layer_property['norm_type']),
                           int(self.layer_property['kernel_size']),
                           int(self.layer_property['stride']),
                           ceil_mode)


class LpPool2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维Lp池化层')
        self.layer_property = {
            'norm_type': '',
            'kernel_size': '',
            'stride': '',
            'ceil_mode': 'False',
        }

    def generateModule(self):
        if self.layer_property['ceil_mode'] == "True":
            ceil_mode = True
        else:
            ceil_mode = False
        return nn.LPPool2d(int(self.layer_property['norm_type']),
                           int(self.layer_property['kernel_size']),
                           int(self.layer_property['stride']),
                           ceil_mode)


"""
Normalization layer. These layers can normalize data, then data can through activation function.
"""


class BatchNorm1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维批标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.BatchNorm1d(int(self.layer_property['num_features']),
                              int(self.layer_property['eps']),
                              int(self.layer_property['momentum']),
                              affine)


class BatchNorm2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维批标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.BatchNorm2d(int(self.layer_property['num_features']),
                              int(self.layer_property['eps']),
                              int(self.layer_property['momentum']),
                              affine)


class BatchNorm3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维批标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.BatchNorm3d(int(self.layer_property['num_features']),
                              int(self.layer_property['eps']),
                              int(self.layer_property['momentum']),
                              affine)


class LazyBatchNorm1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维延迟批标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.LazyBatchNorm1d(float(self.layer_property['eps']),
                                  float(self.layer_property['momentum']),
                                  affine)


class LazyBatchNorm2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维延迟批标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.LazyBatchNorm2d(float(self.layer_property['eps']),
                                  float(self.layer_property['momentum']),
                                  affine)


class LazyBatchNorm3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维延迟批标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False
        return nn.LazyBatchNorm3d(float(self.layer_property['eps']),
                                  float(self.layer_property['momentum']),
                                  affine)


class InstanceNorm1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维实例标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.InstanceNorm1d(int(self.layer_property['num_features']),
                                 float(self.layer_property['eps']),
                                 float(self.layer_property['momentum']),
                                 affine)


class InstanceNorm2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维实例标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.InstanceNorm2d(int(self.layer_property['num_features']),
                                 float(self.layer_property['eps']),
                                 float(self.layer_property['momentum']),
                                 affine)


class InstanceNorm3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维实例标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.InstanceNorm3d(int(self.layer_property['num_features']),
                                 float(self.layer_property['eps']),
                                 float(self.layer_property['momentum']),
                                 affine)


class LazyInstanceNorm1dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='一维延迟实例标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.LazyInstanceNorm1d(float(self.layer_property['eps']),
                                     float(self.layer_property['momentum']),
                                     affine)


class LazyInstanceNorm2dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='二维延迟实例标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.LazyInstanceNorm2d(float(self.layer_property['eps']),
                                     float(self.layer_property['momentum']),
                                     affine)


class LazyInstanceNorm3dLayer(BaseItem):
    def __init__(self):
        super().__init__(text='三维延迟实例标准化层')
        self.layer_property = {
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'False',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.LazyInstanceNorm3d(float(self.layer_property['eps']),
                                     float(self.layer_property['momentum']),
                                     affine)


class LayerNormLayer(BaseItem):
    def __init__(self):
        super().__init__(text='层标准化层')
        self.layer_property = {
            'normalized_shape': '',
            'eps': '0.00001',
            'elementwise_affine': 'True',
            'bias': 'True',
        }

    def generateModule(self):
        if self.layer_property['elementwise_affine'] == 'True':
            elementwise_affine = True
        else:
            elementwise_affine = False

        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.LayerNorm(int(self.layer_property['normalized_shape']),
                            float(self.layer_property['eps']),
                            elementwise_affine,
                            bias)


class GroupNormLayer(BaseItem):
    def __init__(self):
        super().__init__(text='组标准化层')
        self.layer_property = {
            'num_groups': '',
            'num_channels': '',
            'eps': '0.00001',
            'affine': 'True',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.GroupNorm(int(self.layer_property['num_groups']),
                            int(self.layer_property['num_channels']),
                            float(self.layer_property['eps']),
                            affine)


class LocalResponseNormLayer(BaseItem):
    def __init__(self):
        super().__init__(text='局部响应标准化层')
        self.layer_property = {
            'size': '',
            'alpha': '0.0001',
            'beta': '0.75',
            'k': '1.0',
        }

    def generateModule(self):
        return nn.LocalResponseNorm(int(self.layer_property['size']),
                                    float(self.layer_property['alpha']),
                                    float(self.layer_property['beta']),
                                    float(self.layer_property['k']))


class SyncBatchNormLayer(BaseItem):
    def __init__(self):
        super().__init__(text='同步批标准化层')
        self.layer_property = {
            'num_features': '',
            'eps': '0.00001',
            'momentum': '0.1',
            'affine': 'True',
        }

    def generateModule(self):
        if self.layer_property['affine'] == 'True':
            affine = True
        else:
            affine = False

        return nn.SyncBatchNorm(int(self.layer_property['num_features']),
                                float(self.layer_property['eps']),
                                float(self.layer_property['momentum']),
                                affine)


"""
Linear layer
"""


class LinearLayer(BaseItem):
    def __init__(self):
        super().__init__(text='全连接层')
        self.layer_property = {
            'in_features': '1',
            'out_features': '1',
            'bias': 'True',
        }

    def generateModule(self):
        if self.layer_property['bias'] == 'True':
            bias = True
        else:
            bias = False
        return nn.Linear(int(self.layer_property['in_features']),
                         int(self.layer_property['out_features']),
                         bias)


"""
Activation function, deep learning model need these layer 
to activate value.
"""


class ReluLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Relu')
        self.layer_property = {
            'inplace': 'True'
        }

    def generateModule(self):
        if self.layer_property['inplace'] == 'True':
            inplace = True
        else:
            inplace = False
        return nn.ReLU(inplace)


class LeakyReluLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Leaky\nRelu')
        self.layer_property = {
            'negative_slope': '0.01',
            'inplace': 'False'
        }

    def generateModule(self):
        if self.layer_property['inplace'] == 'True':
            inplace = True
        else:
            inplace = False
        return nn.LeakyReLU(float(self.layer_property['negative_slope']),
                            inplace)


class PreluLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Prelu')
        self.layer_property = {
            'num_parameters': '1',
            'init': '0.25'
        }

    def generateModule(self):
        return nn.PReLU(int(self.layer_property['num_parameters']),
                        float(self.layer_property['init']))


class EluLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Elu')
        self.layer_property = {
            'alpha': '1.0',
            'inplace': 'False'
        }

    def generateModule(self):
        if self.layer_property['inplace'] == 'True':
            inplace = True
        else:
            inplace = False
        return nn.ELU(float(self.layer_property['alpha']),
                      inplace)


class SeluLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Selu')
        self.layer_property = {
            'inplace': 'False'
        }

    def generateModule(self):
        if self.layer_property['inplace'] == 'True':
            inplace = True
        else:
            inplace = False
        return nn.SELU(inplace)


class SoftplusLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Softplus')
        self.layer_property = {
            'beta': '1',
            'threshold': '20',
        }

    def generateModule(self):
        return nn.Softplus(int(self.layer_property['beta']),
                           int(self.layer_property['threshold']))


class SoftmaxLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Softmax')
        self.layer_property = {
            'dim': '0'
        }

    def generateModule(self):
        return nn.Softmax(int(self.layer_property['dim']))


class SigmoidLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Sigmoid')
        self.layer_property = {
        }

    def generateModule(self):
        return nn.Sigmoid()


class TanhLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Tanh')
        self.layer_property = {
        }

    def generateModule(self):
        return nn.Tanh()


class SoftsignLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Softsign')
        self.layer_property = {
        }

    def generateModule(self):
        return nn.Softsign()


"""
Other function Layer.
"""

class AddModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

class AddLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Add')
        self.layer_property = {
        }

    def generateModule(self):
        return AddModule()

class MulModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x * y

class MulLayer(BaseItem):
    def __init__(self):
        super().__init__(text='Mul')
        self.layer_property = {
        }

    def generateModule(self):
        return MulModule()

def getLayer(name):
    NAME_TO_LAYER = {
        'featureInputLayer': DataloaderLayer,
        'convolution1dLayer': Convolution1dLayer,
        'convolution2dLayer': Convolution2dLayer,
        'convolution3dLayer': Convolution3dLayer,
        'transposedConv1dLayer': TransposedConv1dLayer,
        'transposedConv2dLayer': TransposedConv2dLayer,
        'transposedConv3dLayer': TransposedConv3dLayer,
        'rnnLayer': RNNLayer,
        'birnnLayer': BiRNNLayer,
        'lstmLayer': LSTMLayer,
        'bilstmLayer': BiLSTMLayer,
        'gruLayer': GRULayer,
        'bigruLayer': BiGRULayer,
        'flattenLayer': FlattenLayer,
        'maxPool1dLayer': MaxPool1dLayer,
        'maxPool2dLayer': MaxPool2dLayer,
        'maxPool3dLayer': MaxPool3dLayer,
        'avgPool1dLayer': AvgPool1dLayer,
        'avgPool2dLayer': AvgPool2dLayer,
        'avgPool3dLayer': AvgPool3dLayer,
        'adaptiveMaxPool1dLayer': AdaptiveMaxPool1dLayer,
        'adaptiveMaxPool2dLayer': AdaptiveMaxPool2dLayer,
        'adaptiveMaxPool3dLayer': AdaptiveMaxPool3dLayer,
        'adaptiveAvgPool1dLayer': AdaptiveAvgPool1dLayer,
        'adaptiveAvgPool2dLayer': AdaptiveAvgPool2dLayer,
        'adaptiveAvgPool3dLayer': AdaptiveAvgPool3dLayer,
        'fractionalMaxPool2dLayer': FractionalMaxPool2dLayer,
        'fractionalMaxPool3dLayer': FractionalMaxPool3dLayer,
        'lpPool1dLayer': LpPool1dLayer,
        'lpPool2dLayer': LpPool2dLayer,
        'batchNorm1dLayer': BatchNorm1dLayer,
        'batchNorm2dLayer': BatchNorm2dLayer,
        'batchNorm3dLayer': BatchNorm3dLayer,
        'lazyBatchNorm1dLayer': LazyBatchNorm1dLayer,
        'lazyBatchNorm2dLayer': LazyBatchNorm2dLayer,
        'lazyBatchNorm3dLayer': LazyBatchNorm3dLayer,
        'instanceNorm1dLayer': InstanceNorm1dLayer,
        'instanceNorm2dLayer': InstanceNorm2dLayer,
        'instanceNorm3dLayer': InstanceNorm3dLayer,
        'lazyInstanceNorm1dLayer': LazyInstanceNorm1dLayer,
        'lazyInstanceNorm2dLayer': LazyInstanceNorm2dLayer,
        'lazyInstanceNorm3dLayer': LazyInstanceNorm3dLayer,
        'layerNormLayer': LayerNormLayer,
        'groupNormLayer': GroupNormLayer,
        'localResponseNormLayer': LocalResponseNormLayer,
        'syncBatchNormLayer': SyncBatchNormLayer,
        'linear': LinearLayer,
        'Relu': ReluLayer,
        'LeakyRelu': LeakyReluLayer,
        'Prelu': PreluLayer,
        'Elu': EluLayer,
        'Selu': SeluLayer,
        'Softplus': SoftplusLayer,
        'Softmax': SoftmaxLayer,
        'Sigmoid': SigmoidLayer,
        'Tanh': TanhLayer,
        'Softsign': SoftsignLayer,
        'Add': AddLayer,
        'Mul': MulLayer,
    }
    if name in NAME_TO_LAYER.keys():
        return NAME_TO_LAYER[name]()

    return None
