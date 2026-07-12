import copy
import torch


def _stack_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Вспомогательная функция для стекования тензоров из словарей состояний моделей.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
    
    Возвращает:
        dict[str, torch.Tensor]: Словарь, где для каждого ключа тензоры из всех моделей
                                 объединены в один тензор размерности (num_models, *shape).
    """
    keys = state_dicts[0].keys()
    stacked = {}
    for key in keys:
        # Преобразуем все тензоры к типу float и стековываем по первому измерению
        stacked[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
    return stacked


def fedavg(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Стандартный алгоритм усреднения Федеративного обучения (FedAvg).
    Вычисляет среднее арифметическое весов всех клиентов.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
    
    Возвращает:
        dict[str, torch.Tensor]: Усреднённые веса модели.
    """
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        # Стекуем тензоры всех клиентов и вычисляем среднее по первому измерению
        avg[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg


def coordinate_median(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Агрегация с использованием медианы по каждому координатному измерению.
    Это робастный метод, устойчивый к выбросам.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
    
    Возвращает:
        dict[str, torch.Tensor]: Медианные веса модели.
    """
    stacked = _stack_state_dicts(state_dicts)
    result = {}
    for key, values in stacked.items():
        # Вычисляем медиану по первому измерению (между клиентами)
        result[key] = values.median(dim=0).values
    return result


def trimmed_mean(state_dicts: list[dict[str, torch.Tensor]], trim_ratio: float = 0.2) -> dict[str, torch.Tensor]:
    """
    Агрегация с использованием усечённого среднего (Trimmed Mean).
    Отбрасывает определённую долю самых больших и самых маленьких значений
    и вычисляет среднее на оставшихся. Повышает устойчивость к аномалиям.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
        trim_ratio (float, optional): Доля значений, отсекаемых с каждого края.
        Должна быть в диапазоне [0, 0.5). По умолчанию 0.2.
    
    Возвращает:
        dict[str, torch.Tensor]: Усечённые средние веса модели.
    """
    n = len(state_dicts)
    trim_count = int(n * trim_ratio)
    
    # Если отсекать нечего или отсечение слишком сильное, используем обычное усреднение
    if trim_count == 0 or 2 * trim_count >= n:
        return fedavg(state_dicts)

    stacked = _stack_state_dicts(state_dicts)
    result = {}
    for key, values in stacked.items():
        # Преобразуем тензор весов в плоский вид для сортировки
        flat = values.view(n, -1)
        # Сортируем значения вдоль оси клиентов
        sorted_vals, _ = torch.sort(flat, dim=0)
        # Отбрасываем trim_count самых малых и trim_count самых больших значений
        trimmed = sorted_vals[trim_count : n - trim_count]
        # Вычисляем среднее и восстанавливаем исходную форму тензора
        result[key] = trimmed.mean(dim=0).view_as(state_dicts[0][key])
    return result


def _pairwise_distances(state_dicts: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Вычисляет попарные квадратичные расстояния между весами моделей клиентов.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
    
    Возвращает:
        torch.Tensor: Матрица попарных расстояний размерности (n, n).
    """
    n = len(state_dicts)
    dist = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            sq = 0.0
            # Вычисляем сумму квадратов разностей по всем параметрам
            for key in state_dicts[0]:
                diff = state_dicts[i][key].float() - state_dicts[j][key].float()
                sq += (diff * diff).sum().item()
            dist[i, j] = dist[j, i] = sq
    return dist


def krum(state_dicts: list[dict[str, torch.Tensor]], num_byzantine: int = 1) -> dict[str, torch.Tensor]:
    """
    Алгоритм Krum для робастной агрегации в федеративном обучении.
    Выбирает модель клиента, которая наиболее близка к другим моделям,
    игнорируя потенциально вредоносные обновления.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей.
        num_byzantine (int, optional): Ожидаемое количество византийских (вредоносных) клиентов.
        По умолчанию 1.
    
    Возвращает:
        dict[str, torch.Tensor]: Веса модели, выбранной алгоритмом Krum.
    """
    n = len(state_dicts)
    # m = n - num_byzantine - 2 (количество ближайших соседей для рассмотрения)
    m = max(1, n - num_byzantine - 2)
    
    # Вычисляем попарные расстояния между моделями
    dist = _pairwise_distances(state_dicts)

    # Для каждого клиента вычисляем сумму расстояний до m ближайших соседей
    scores = []
    for i in range(n):
        # Находим m+1 ближайших соседей (включая самого себя) и исключаем себя
        nearest = torch.topk(dist[i], k=m + 1, largest=False).values[1:]
        scores.append(nearest.sum().item())

    # Выбираем клиента с минимальной суммой расстояний
    chosen = int(torch.tensor(scores).argmin().item())
    return copy.deepcopy(state_dicts[chosen])


def aggregate(state_dicts: list[dict[str, torch.Tensor]], method: str = "fedavg", trim_ratio: float = 0.2, num_byzantine: int = 1) -> dict[str, torch.Tensor]:
    """
    Основная функция агрегации весов моделей клиентов.
    Поддерживает различные методы агрегации, включая робастные к атакам.
    
    Аргументы:
        state_dicts (list[dict[str, torch.Tensor]]): Список словарей с весами моделей клиентов.
        method (str, optional): Метод агрегации.
                                Доступные варианты:
                                - "fedavg": Стандартное усреднение
                                - "median": Покоординатная медиана
                                - "trimmed_mean": Усечённое среднее
                                - "krum": Алгоритм Krum
                                По умолчанию "fedavg".
        trim_ratio (float, optional): Доля отсекаемых значений для trimmed_mean.
        По умолчанию 0.2.
        num_byzantine (int, optional): Ожидаемое количество вредоносных клиентов для Krum.
        По умолчанию 1.
    
    Возвращает:
        dict[str, torch.Tensor]: Агрегированные веса модели.
    
    Raises:
        ValueError: Если указан неизвестный метод агрегации.
    """
    method = method.lower()
    if method == "fedavg":
        return fedavg(state_dicts)
    if method == "median":
        return coordinate_median(state_dicts)
    if method == "trimmed_mean":
        return trimmed_mean(state_dicts, trim_ratio)
    if method == "krum":
        return krum(state_dicts, num_byzantine)
    raise ValueError(f"Неизвестный метод агрегирования: {method}")