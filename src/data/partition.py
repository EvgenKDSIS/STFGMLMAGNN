from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


@dataclass
class ClientGraph:
    """
    Класс данных для представления графа отдельного клиента в федеративном обучении.
    
    Атрибуты:
        client_id (int): Уникальный идентификатор клиента.
        data (Data): Объект PyTorch Geometric, содержащий признаки узлов и рёбра графа.
        train_mask (torch.Tensor): Маска тренировочных узлов клиента.
        val_mask (torch.Tensor): Маска валидационных узлов клиента.
        test_mask (torch.Tensor): Маска тестовых узлов клиента.
        num_nodes (int): Количество узлов в подграфе клиента.
    """
    client_id: int
    data: Data
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int


def load_dataset(name: str = "Cora", root: str = "data") -> Data:
    """
    Загружает датасет Planetoid (Cora, CiteSeer или PubMed).
    
    Аргументы:
        name (str, optional): Название датасета. По умолчанию "Cora".
        root (str, optional): Корневая директория для хранения данных. По умолчанию "data".
    
    Возвращает:
        Data: Объект данных PyTorch Geometric с полным графом.
    """
    dataset = Planetoid(root=root, name=name)
    return dataset[0]


def _build_client_graph(client_id: int, node_ids: np.ndarray, full_data: Data) -> ClientGraph:
    """
    Вспомогательная функция для построения подграфа клиента на основе списка узлов.
    
    Аргументы:
        client_id (int): Идентификатор клиента.
        node_ids (np.ndarray): Массив идентификаторов узлов, принадлежащих клиенту.
        full_data (Data): Полный граф датасета.
    
    Возвращает:
        ClientGraph: Объект с подграфом и соответствующими масками для клиента.
    """
    # Сортируем узлы и создаём отображение старых индексов на новые (локальные)
    node_ids = np.sort(node_ids.astype(np.int64))
    id_map = {int(old): new for new, old in enumerate(node_ids)}

    # Извлекаем рёбра из полного графа
    edge_index = full_data.edge_index.cpu().numpy()
    src, dst = edge_index[0], edge_index[1]

    # Фильтруем рёбра, оставляя только те, где оба узла принадлежат клиенту
    local_edges = []
    for u, v in zip(src, dst):
        if u in id_map and v in id_map:
            local_edges.append([id_map[u], id_map[v]])

    # Формируем тензор рёбер для локального графа
    if local_edges:
        local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
    else:
        local_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Извлекаем признаки, метки и маски для узлов клиента
    local_x = full_data.x[node_ids]                           # Признаки узлов
    local_y = full_data.y[node_ids]                           # Метки узлов
    local_train = full_data.train_mask[node_ids]              # Тренировочная маска
    local_val = full_data.val_mask[node_ids]                  # Валидационная маска
    local_test = full_data.test_mask[node_ids]                # Тестовая маска

    # Создаём объект данных PyTorch Geometric для клиента
    data = Data(x=local_x, edge_index=local_edge_index, y=local_y)
    data.num_nodes = len(node_ids)

    return ClientGraph(
        client_id=client_id,
        data=data,
        train_mask=local_train,
        val_mask=local_val,
        test_mask=local_test,
        num_nodes=len(node_ids)
    )


def partition_iid(data: Data, num_clients: int, seed: int = 42) -> list[ClientGraph]:
    """
    Выполняет IID-разбиение данных между клиентами (одинаковое распределение классов).
    
    Аргументы:
        data (Data): Полный граф датасета.
        num_clients (int): Количество клиентов.
        seed (int, optional): Зерно для воспроизводимости. По умолчанию 42.
    
    Возвращает:
        list[ClientGraph]: Список объектов ClientGraph для каждого клиента.
    """
    rng = np.random.default_rng(seed)
    
    # Получаем индексы тренировочных узлов и перемешиваем их
    train_nodes = torch.where(data.train_mask)[0].cpu().numpy()
    rng.shuffle(train_nodes)

    # Равномерно распределяем тренировочные узлы между клиентами
    splits = np.array_split(train_nodes, num_clients)
    
    all_nodes = np.arange(data.num_nodes)
    assigned = np.zeros(data.num_nodes, dtype=bool)
    client_node_lists: list[np.ndarray] = []

    # Для каждого клиента добавляем соседей (до 3) к его тренировочным узлам
    for split in splits:
        nodes = set(split.tolist())
        for node in split:
            # Находим соседей узла и добавляем их (до 3 первых)
            neighbors = data.edge_index[1, data.edge_index[0] == node].cpu().numpy()
            nodes.update(int(n) for n in neighbors[:3])
        node_array = np.array(sorted(nodes), dtype=np.int64)
        assigned[node_array] = True
        client_node_lists.append(node_array)

    # Распределяем оставшиеся узлы (не назначенные ни одному клиенту) между клиентами
    remaining = all_nodes[~assigned]
    for idx, node in enumerate(remaining):
        client_node_lists[idx % num_clients] = np.unique(
            np.concatenate([client_node_lists[idx % num_clients], [node]])
        )

    # Строим графы для каждого клиента
    return [_build_client_graph(i, nodes, data) for i, nodes in enumerate(client_node_lists)]


def partition_dirichlet(data: Data, num_clients: int, alpha: float = 0.5, seed: int = 42) -> list[ClientGraph]:
    """
    Выполняет Non-IID разбиение данных между клиентами с использованием распределения Дирихле.
    Это позволяет моделировать различные степени гетерогенности данных.
    
    Аргументы:
        data (Data): Полный граф датасета.
        num_clients (int): Количество клиентов.
        alpha (float, optional): Параметр концентрации Дирихле.
        Чем меньше alpha, тем более гетерогенное распределение. По умолчанию 0.5.
        seed (int, optional): Зерно для воспроизводимости. По умолчанию 42.
    
    Возвращает:
        list[ClientGraph]: Список объектов ClientGraph для каждого клиента.
    """
    rng = np.random.default_rng(seed)
    
    # Получаем тренировочные узлы и их метки
    train_nodes = torch.where(data.train_mask)[0].cpu().numpy()
    labels = data.y[train_nodes].cpu().numpy()
    num_classes = int(data.y.max().item()) + 1

    # Группируем узлы по классам
    class_indices = [train_nodes[labels == c] for c in range(num_classes)]
    
    # Инициализируем списки узлов для каждого клиента
    client_nodes: list[list[int]] = [[] for _ in range(num_clients)]

    # Для каждого класса распределяем его узлы между клиентами согласно распределению Дирихле
    for class_nodes in class_indices:
        rng.shuffle(class_nodes)
        # Генерируем пропорции для клиентов из распределения Дирихле
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Вычисляем точки разделения для распределения узлов
        proportions = (np.cumsum(proportions) * len(class_nodes)).astype(int)[:-1]
        # Разбиваем узлы класса между клиентами
        splits = np.split(class_nodes, proportions)
        for client_id, split in enumerate(splits):
            client_nodes[client_id].extend(split.tolist())

    assigned = set()
    client_node_lists = []
    
    # Добавляем соседей (до 2) к узлам каждого клиента
    for client_id, nodes in enumerate(client_nodes):
        node_set = set(nodes)
        for node in nodes:
            neighbors = data.edge_index[1, data.edge_index[0] == node].cpu().numpy()
            node_set.update(int(n) for n in neighbors[:2])
        node_array = np.unique(np.array(sorted(node_set), dtype=np.int64))
        assigned.update(node_array.tolist())
        client_node_lists.append(node_array)

    # Распределяем оставшиеся узлы (не назначенные ни одному клиенту)
    all_nodes = np.arange(data.num_nodes)
    remaining = [n for n in all_nodes if n not in assigned]
    for idx, node in enumerate(remaining):
        client_node_lists[idx % num_clients] = np.unique(
            np.concatenate([client_node_lists[idx % num_clients], [node]])
        )

    # Строим графы для каждого клиента
    return [_build_client_graph(i, nodes, data) for i, nodes in enumerate(client_node_lists)]


def build_clients(data: Data, num_clients: int, iid: bool, alpha: float, seed: int) -> list[ClientGraph]:
    """
    Главная функция для построения списка клиентов с заданным типом разбиения данных.
    
    Аргументы:
        data (Data): Полный граф датасета.
        num_clients (int): Количество клиентов.
        iid (bool): Флаг, определяющий тип разбиения:
                    True - IID разбиение,
                    False - Non-IID разбиение Дирихле.
        alpha (float): Параметр концентрации Дирихле (используется при Non-IID).
        seed (int): Зерно для воспроизводимости.
    
    Возвращает:
        list[ClientGraph]: Список объектов ClientGraph для каждого клиента.
    """
    if iid:
        return partition_iid(data, num_clients, seed)
    return partition_dirichlet(data, num_clients, alpha, seed)