import copy
import torch


def flip_labels(labels: torch.Tensor, train_mask: torch.Tensor, num_classes: int, flip_ratio: float = 1.0, seed: int = 42) -> torch.Tensor:
    """
    Выполняет атаку подмены меток (label flipping) для указанной доли тренировочных образцов.
    
    Аргументы:
        labels (torch.Tensor): Исходные метки всех узлов графа.
        train_mask (torch.Tensor): Маска, указывающая, какие узлы принадлежат тренировочной выборке.
        num_classes (int): Общее количество классов в задаче классификации.
        flip_ratio (float, optional): Доля тренировочных узлов, у которых будут изменены метки.
        Значение должно быть в диапазоне [0, 1]. По умолчанию 1.0.
        seed (int, optional): Зерно для генератора случайных чисел, обеспечивающее воспроизводимость.
        По умолчанию 42.
    
    Возвращает:
        torch.Tensor: Тензор с изменёнными метками (отравленные данные).
    """
    # Создаём копию исходных меток, чтобы не изменять их напрямую
    poisoned = labels.clone()
    
    # Получаем индексы узлов, принадлежащих тренировочной выборке
    train_ids = torch.where(train_mask)[0]
    
    # Если тренировочных узлов нет, возвращаем неизменённые метки
    if len(train_ids) == 0:
        return poisoned

    # Инициализируем генератор случайных чисел с заданным зерном для воспроизводимости
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Вычисляем количество узлов, метки которых будут изменены (минимум 1)
    num_flip = max(1, int(len(train_ids) * flip_ratio))
    
    # Случайным образом выбираем узлы для атаки без повторений
    perm = train_ids[torch.randperm(len(train_ids), generator=generator)[:num_flip]]

    # Для каждого выбранного узла изменяем метку на случайную, отличную от исходной
    for node in perm:
        original = int(labels[node].item())  # Сохраняем исходную метку
        # Генерируем новую метку: сдвигаем на 1 и добавляем случайное число от 0 до num_classes-2,
        # затем берём по модулю num_classes, чтобы гарантировать, что новая метка != исходной
        poisoned[node] = (original + 1 + torch.randint(num_classes - 1, (1,), generator=generator)) % num_classes
    
    return poisoned


def apply_label_flip(client_data, num_classes: int, flip_ratio: float, seed: int):
    """
    Применяет атаку подмены меток к данным клиента в федеративном обучении.
    
    Аргументы:
        client_data: Объект данных клиента (предполагается, что содержит поля data.y и data.train_mask).
        num_classes (int): Общее количество классов.
        flip_ratio (float): Доля тренировочных образцов для подмены меток.
        seed (int): Зерно для воспроизводимости атаки.
    
    Возвращает:
        Изменённый объект данных клиента с отравленными метками.
    """
    # Создаём глубокую копию данных клиента, чтобы не изменять исходные данные
    client_data = copy.deepcopy(client_data)
    
    # Применяем подмену меток к тренировочной выборке клиента
    client_data.data.y = flip_labels(
        client_data.data.y,      # Исходные метки
        client_data.train_mask,  # Маска тренировочных узлов
        num_classes,            # Количество классов
        flip_ratio,             # Доля подменяемых меток
        seed,                   # Зерно случайности
    )
    
    return client_data


def scaled_update(local_state: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor], scale_factor: float) -> dict[str, torch.Tensor]:
    """
    Выполняет масштабированное обновление весов модели в федеративном обучении.
    Позволяет усилить или ослабить вклад локального обновления при агрегации.
    
    Аргументы:
        local_state (dict[str, torch.Tensor]): Словарь с весами локальной модели клиента.
        global_state (dict[str, torch.Tensor]): Словарь с весами глобальной модели.
        scale_factor (float): Коэффициент масштабирования локального обновления.
        > 1.0 - усиление вклада локального обновления
        < 1.0 - ослабление вклада локального обновления
        = 0.0 - полное игнорирование локального обновления
    
    Возвращает:
        dict[str, torch.Tensor]: Словарь с обновлёнными весами модели.
    """
    scaled = {}
    
    # Проходим по всем слоям модели (ключам в словаре состояний)
    for key in local_state:
        # Вычисляем разницу между локальными и глобальными весами (локальное обновление)
        delta = local_state[key].float() - global_state[key].float()
        
        # Применяем масштабирование: global + scale_factor * (local - global)
        # При scale_factor = 1.0 получаем стандартное локальное обновление
        # При scale_factor = 0.0 получаем глобальные весы без изменений
        scaled[key] = global_state[key].float() + scale_factor * delta
    
    return scaled