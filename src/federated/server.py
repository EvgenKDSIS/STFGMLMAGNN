from src.aggregation import aggregate


class FederatedServer:
    """
    Класс, представляющий сервер в федеративном обучении.
    Отвечает за агрегацию весов моделей, полученных от клиентов.
    """
    
    def __init__(self, aggregation: str, trim_ratio: float = 0.2, num_byzantine: int = 1):
        """
        Инициализация сервера федеративного обучения.
        
        Аргументы:
            aggregation (str): Метод агрегации весов моделей.
            Возможные значения: "fedavg", "trimmed_mean", "krum", "median" и др.
            trim_ratio (float, optional): Доля экстремальных значений, отсекаемых при агрегации
            trimmed_mean (обычно от 0.0 до 0.5). По умолчанию 0.2.
            num_byzantine (int, optional): Ожидаемое количество византийских (вредоносных) клиентов.
            Используется в некоторых робастных методах агрегации,
            например, Krum. По умолчанию 1.
        """
        self.aggregation = aggregation
        self.trim_ratio = trim_ratio
        self.num_byzantine = num_byzantine

    def aggregate(self, client_weights: list[dict]) -> dict:
        """
        Выполняет агрегацию весов моделей от всех клиентов.
        
        Аргументы:
            client_weights (list[dict]): Список словарей с весами моделей от каждого клиента.
                                        Каждый словарь содержит параметры модели в формате
                                        {имя_слоя: тензор_весов}.
        
        Возвращает:
            dict: Агрегированные веса модели, которые станут новой глобальной моделью.
                Структура словаря соответствует структуре весов моделей клиентов.
        """
        # Вызываем функцию агрегации из модуля src.aggregation
        # Передаём веса клиентов и параметры агрегации
        return aggregate(client_weights, method=self.aggregation, trim_ratio=self.trim_ratio, num_byzantine=self.num_byzantine)