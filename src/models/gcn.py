import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    Реализация графовой свёрточной нейронной сети (Graph Convolutional Network - GCN).
    
    GCN использует спектральный подход к свёртке на графах, основанный на аппроксимации 
    Чебышева первого порядка. Это одна из наиболее фундаментальных и широко используемых
    архитектур для обучения на графовых данных.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        """
        Инициализация модели GCN.
        
        Аргументы:
            in_channels (int): Размерность входных признаков узлов.
            hidden_channels (int): Размерность скрытого представления.
            out_channels (int): Размерность выходных признаков (количество классов).
            dropout (float, optional): Вероятность дропаута для регуляризации. По умолчанию 0.5.
        """
        super().__init__()
        
        # Первый свёрточный слой: входные признаки -> скрытое представление
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Второй свёрточный слой: скрытое представление -> выходные логиты
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Сохраняем вероятность дропаута для использования в forward
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Прямой проход через модель GCN.
        
        Аргументы:
            x (torch.Tensor): Тензор признаков узлов размерности (num_nodes, in_channels).
            edge_index (torch.Tensor): Тензор рёбер графа размерности (2, num_edges).
        
        Возвращает:
            torch.Tensor: Выходные логиты модели размерности (num_nodes, out_channels).
        
        Примечание:
            Классическая архитектура GCN использует ReLU активацию после первого слоя
            и дропаут для регуляризации. Выходной слой не имеет активации,
            так как обычно используется с CrossEntropyLoss.
        """
        # Первый свёрточный слой
        x = self.conv1(x, edge_index)   # Применяем первую GCN свёртку
        x = F.relu(x)                   # ReLU (Rectified Linear Unit) активация
        
        # Применяем дропаут для регуляризации (только во время обучения)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Второй свёрточный слой (выходной)
        x = self.conv2(x, edge_index)   # Применяем вторую GCN свёртку
        
        # Возвращаем логиты (без Softmax, так как CrossEntropyLoss сама применяет LogSoftmax)
        return x