import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.attacks import apply_label_flip, scaled_update
from src.data.partition import ClientGraph


def train_local(model: torch.nn.Module, data: Data, train_mask: torch.Tensor, device: torch.device, epochs: int, lr: float, weight_decay: float) -> dict[str, torch.Tensor]:
    """
    Выполняет локальное обучение модели на данных клиента.
    
    Аргументы:
        model (torch.nn.Module): Глобальная модель, которая будет обучена локально.
        data (Data): Данные графа клиента (признаки, рёбра, метки).
        train_mask (torch.Tensor): Маска тренировочных узлов клиента.
        device (torch.device): Устройство для вычислений (CPU или GPU).
        epochs (int): Количество эпох локального обучения.
        lr (float): Скорость обучения (learning rate).
        weight_decay (float): Коэффициент регуляризации L2.
    
    Возвращает:
        dict[str, torch.Tensor]: Словарь с весами локально обученной модели.
    """
    # Создаём глубокую копию модели и переносим на целевое устройство
    model = copy.deepcopy(model).to(device)
    model.train()  # Переводим модель в режим обучения
    
    # Настраиваем оптимизатор Adam с заданными параметрами
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Переносим данные на целевое устройство
    x = data.x.to(device)                     # Признаки узлов
    edge_index = data.edge_index.to(device)   # Рёбра графа
    y = data.y.to(device)                     # Метки узлов
    mask = train_mask.to(device)              # Маска тренировочных узлов

    # Цикл локального обучения
    for _ in range(epochs):
        optimizer.zero_grad()                 # Обнуляем градиенты
        out = model(x, edge_index)            # Прямой проход через модель
        loss = F.cross_entropy(out[mask], y[mask])  # Вычисляем функцию потерь
        loss.backward()                       # Обратный проход (вычисление градиентов)
        optimizer.step()                      # Обновление весов модели

    # Возвращаем веса модели, перенесённые на CPU и отделённые от графа вычислений
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class FederatedClient:
    """
    Класс, представляющий клиента в федеративном обучении.
    Поддерживает как честное поведение, так и различные виды атак.
    """
    
    def __init__(self, client_graph: ClientGraph, device: torch.device, local_epochs: int, lr: float, weight_decay: float, is_malicious: bool = False, attack: str = "none", flip_ratio: float = 1.0, scale_factor: float = 10.0, num_classes: int = 7, seed: int = 42):
        """
        Инициализация клиента федеративного обучения.
        
        Аргументы:
            client_graph (ClientGraph): Объект с графовыми данными клиента.
            device (torch.device): Устройство для вычислений.
            local_epochs (int): Количество эпох локального обучения.
            lr (float): Скорость обучения.
            weight_decay (float): Коэффициент регуляризации L2.
            is_malicious (bool, optional): Является ли клиент вредоносным. По умолчанию False.
            attack (str, optional): Тип атаки ("none", "label_flip", "scaled_update").
            По умолчанию "none".
            flip_ratio (float, optional): Доля меток для подмены (при атаке label_flip).
            По умолчанию 1.0.
            scale_factor (float, optional): Коэффициент масштабирования обновления 
            (при атаке scaled_update). По умолчанию 10.0.
            num_classes (int, optional): Количество классов в задаче. По умолчанию 7.
            seed (int, optional): Базовое зерно для воспроизводимости. 
            К нему прибавляется client_id для уникальности.
            По умолчанию 42.
        """
        self.client_id = client_graph.client_id
        self.client_graph = client_graph
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_malicious = is_malicious
        self.attack = attack
        self.flip_ratio = flip_ratio
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        # Уникальное зерно для каждого клиента для воспроизводимости атак
        self.seed = seed + client_graph.client_id

    def local_update(self, global_model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """
        Выполняет локальное обновление модели с возможным применением атак.
        
        Аргументы:
            global_model (torch.nn.Module): Текущая глобальная модель.
        
        Возвращает:
            dict[str, torch.Tensor]: Обновлённые веса модели (возможно, модифицированные атакой).
        """
        # Получаем графовые данные клиента
        graph = self.client_graph
        data = graph.data

        # Применяем атаку подмены меток, если клиент вредоносный и выбрана соответствующая атака
        if self.is_malicious and self.attack == "label_flip":
            # Изменяем метки в тренировочной выборке клиента
            graph = apply_label_flip(graph, self.num_classes, self.flip_ratio, self.seed)
            data = graph.data  # Обновляем данные с отравленными метками

        # Выполняем локальное обучение на данных клиента
        local_weights = train_local(
            global_model, 
            data, 
            graph.train_mask, 
            self.device, 
            self.local_epochs, 
            self.lr, 
            self.weight_decay
        )

        # Применяем атаку масштабированного обновления, если клиент вредоносный
        if self.is_malicious and self.attack == "scaled_update":
            # Сохраняем состояние глобальной модели на CPU
            global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
            # Масштабируем локальное обновление относительно глобальной модели
            local_weights = scaled_update(local_weights, global_state, self.scale_factor)

        # Возвращаем веса модели (возможно, модифицированные атакой)
        return local_weights