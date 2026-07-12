import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """
    Реализация графовой нейронной сети на основе механизма внимания (Graph Attention Network - GAT).
    
    GAT использует механизм самовнимания для вычисления важности соседних узлов,
    что позволяет модели адаптивно учитывать вклад различных соседей при агрегации.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5, heads: int = 4):
        """
        Инициализация модели GAT.
        
        Аргументы:
            in_channels (int): Размерность входных признаков узлов.
            hidden_channels (int): Размерность скрытого представления.
            out_channels (int): Размерность выходных признаков (количество классов).
            dropout (float, optional): Вероятность дропаута для регуляризации. По умолчанию 0.5.
            heads (int, optional): Количество голов внимания в первом слое.
            Во втором слое используется 1 голова для получения итогового выхода.
            По умолчанию 4.
        """
        super().__init__()
        
        # Первый свёрточный слой: входные признаки -> скрытое представление
        # Использует несколько голов внимания (heads) для захвата различных паттернов
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads,        # Количество голов внимания
            dropout=dropout     # Дропаут для внимания (применяется внутри слоя)
        )
        
        # Второй свёрточный слой: скрытое представление -> выходные логиты
        # Использует одну голову внимания (concat=False) для получения финального выхода
        self.conv2 = GATConv(
            hidden_channels * heads,  # Размерность увеличивается в heads раз из-за конкатенации
            out_channels, 
            heads=1,                  # Одна голова для выхода
            concat=False,             # Не конкатенируем головы, а усредняем
            dropout=dropout
        )
        
        self.dropout = dropout  # Сохраняем вероятность дропаута для использования в forward

    def forward(self, x, edge_index):
        """
        Прямой проход через модель GAT.
        
        Аргументы:
            x (torch.Tensor): Тензор признаков узлов размерности (num_nodes, in_channels).
            edge_index (torch.Tensor): Тензор рёбер графа размерности (2, num_edges).
        
        Возвращает:
            torch.Tensor: Выходные логиты модели размерности (num_nodes, out_channels).
        """
        # Первый слой GAT с функцией активации ELU
        x = self.conv1(x, edge_index)   # Применяем первый слой внимания
        x = F.elu(x)                    # ELU (Exponential Linear Unit) активация
        
        # Применяем дропаут для регуляризации (только во время обучения)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Второй слой GAT (выходной слой)
        x = self.conv2(x, edge_index)   # Применяем второй слой внимания
        
        # Возвращаем логиты (без Softmax, так как CrossEntropyLoss сама применяет LogSoftmax)
        return x