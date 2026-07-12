import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.aggregation import aggregate
from src.config import ExperimentConfig
from src.data.partition import build_clients, load_dataset
from src.federated import FederatedClient, FederatedServer
from src.models import build_model
from src.utils import evaluate_global, set_seed


def parse_args() -> tuple[ExperimentConfig, str]:
    """
    Разбор аргументов командной строки и создание конфигурации эксперимента.
    
    Возвращает:
        tuple[ExperimentConfig, str]: Объект конфигурации и путь для сохранения результатов.
    """
    parser = argparse.ArgumentParser(description="Эксперименты с защищёнными федеративными GNN")
    
    # Аргументы датасета
    parser.add_argument("--dataset", default="Cora", help="Название датасета")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat"], help="Тип модели")
    
    # Аргументы клиентов
    parser.add_argument("--num-clients", type=int, default=5, help="Количество клиентов")
    parser.add_argument("--num-malicious", type=int, default=1, help="Количество вредоносных клиентов")
    parser.add_argument("--iid", action="store_true", help="Использовать IID разбиение данных")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, 
                       help="Параметр концентрации Дирихле для Non-IID")
    
    # Аргументы модели
    parser.add_argument("--hidden-dim", type=int, default=64, help="Размерность скрытого слоя")
    parser.add_argument("--dropout", type=float, default=0.5, help="Вероятность дропаута")
    
    # Аргументы обучения
    parser.add_argument("--rounds", type=int, default=50, help="Количество раундов федеративного обучения")
    parser.add_argument("--local-epochs", type=int, default=3, help="Количество эпох локального обучения")
    parser.add_argument("--lr", type=float, default=0.01, help="Скорость обучения")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Коэффициент регуляризации L2")
    
    # Аргументы агрегации
    parser.add_argument("--aggregation", default="fedavg", 
                       choices=["fedavg", "median", "trimmed_mean", "krum"],
                       help="Метод агрегации")
    parser.add_argument("--trim-ratio", type=float, default=0.2, 
                       help="Доля отсекаемых значений для trimmed_mean")
    
    # Аргументы атак
    parser.add_argument("--attack", default="none", 
                       choices=["none", "label_flip", "scaled_update"],
                       help="Тип атаки вредоносных клиентов")
    parser.add_argument("--flip-ratio", type=float, default=1.0, 
                       help="Доля подменяемых меток при атаке label_flip")
    parser.add_argument("--scale-factor", type=float, default=10.0, 
                       help="Коэффициент масштабирования при атаке scaled_update")
    
    # Общие аргументы
    parser.add_argument("--seed", type=int, default=42, help="Зерно для воспроизводимости")
    parser.add_argument("--device", default="auto", help="Устройство для вычислений (cpu/cuda/auto)")
    parser.add_argument("--log-every", type=int, default=5, 
                       help="Интервал логирования в раундах")
    parser.add_argument("--output-dir", default="results", 
                       help="Директория для сохранения результатов")
    
    args = parser.parse_args()

    # Автоматический выбор устройства
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        num_clients=args.num_clients,
        num_malicious=args.num_malicious,
        iid=args.iid,
        dirichlet_alpha=args.dirichlet_alpha,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aggregation=args.aggregation,
        trim_ratio=args.trim_ratio,
        attack=args.attack,
        flip_ratio=args.flip_ratio,
        scale_factor=args.scale_factor,
        seed=args.seed,
        device=device,
        log_every=args.log_every,
    ), args.output_dir


def select_malicious_clients(num_clients: int, num_malicious: int, seed: int) -> list[int]:
    """
    Выбирает случайных клиентов, которые будут вредоносными.
    
    Аргументы:
        num_clients (int): Общее количество клиентов.
        num_malicious (int): Количество вредоносных клиентов.
        seed (int): Зерно для воспроизводимости.
    
    Возвращает:
        list[int]: Список ID вредоносных клиентов.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(num_clients, generator=rng).tolist()
    return perm[:num_malicious]


def run_experiment(cfg: ExperimentConfig) -> dict:
    """
    Запускает эксперимент по федеративному обучению с заданной конфигурацией.
    
    Аргументы:
        cfg (ExperimentConfig): Конфигурация эксперимента.
    
    Возвращает:
        dict: Словарь с результатами эксперимента (конфигурация, история, финальные метрики).
    """
    # Устанавливаем зерно для воспроизводимости
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Загружаем датасет
    data = load_dataset(cfg.dataset)
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    # Разбиваем данные между клиентами
    clients_data = build_clients(
        data,
        num_clients=cfg.num_clients,
        iid=cfg.iid,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )

    # Выбираем вредоносных клиентов
    malicious_ids = select_malicious_clients(cfg.num_clients, cfg.num_malicious, cfg.seed)
    cfg.malicious_client_ids = malicious_ids

    # Инициализируем глобальную модель
    global_model = build_model(
        cfg.model, num_features, cfg.hidden_dim, num_classes, cfg.dropout
    ).to(device)
    
    # Создаём сервер
    server = FederatedServer(cfg.aggregation, cfg.trim_ratio, cfg.num_malicious)

    # Создаём клиентов
    clients = [
        FederatedClient(
            client_graph=cg,
            device=device,
            local_epochs=cfg.local_epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            is_malicious=cg.client_id in malicious_ids,
            attack=cfg.attack,
            flip_ratio=cfg.flip_ratio,
            scale_factor=cfg.scale_factor,
            num_classes=num_classes,
            seed=cfg.seed,
        )
        for cg in clients_data
    ]

    # История обучения
    history = []
    
    # Выводим информацию об эксперименте
    print(f"Датасет: {cfg.dataset} | Модель: {cfg.model.upper()} | Клиентов: {cfg.num_clients} | Вредоносные: {malicious_ids}")
    print(f"Агрегация: {cfg.aggregation} | Атака: {cfg.attack} | Устройство: {cfg.device}")

    # Основной цикл федеративного обучения
    for round_idx in tqdm(range(1, cfg.rounds + 1), desc="Раунды FL"):
        # Локальное обновление всех клиентов
        client_weights = [client.local_update(global_model) for client in clients]
        
        # Агрегация весов на сервере
        aggregated = server.aggregate(client_weights)
        global_model.load_state_dict(aggregated)

        # Логирование результатов
        if round_idx % cfg.log_every == 0 or round_idx == cfg.rounds:
            metrics = evaluate_global(global_model, data, device)
            metrics["round"] = round_idx
            history.append(metrics)
            print(
                f"Раунд {round_idx:03d} | "
                f"обучение={metrics.get('train_acc', 0):.3f} "
                f"валидация={metrics.get('val_acc', 0):.3f} "
                f"тест={metrics.get('test_acc', 0):.3f}"
            )

    # Финальная оценка
    final_metrics = evaluate_global(global_model, data, device)
    
    return {
        "config": cfg.__dict__,
        "history": history,
        "final_metrics": final_metrics,
    }


def main():
    """
    Основная функция запуска эксперимента.
    """
    # Парсим аргументы и получаем конфигурацию
    cfg, output_dir = parse_args()
    
    # Запускаем эксперимент
    results = run_experiment(cfg)

    # Сохраняем результаты в JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{cfg.dataset}_{cfg.model}_{cfg.aggregation}_{cfg.attack}_"
        f"m{cfg.num_malicious}_c{cfg.num_clients}.json"
    )
    with open(out_path / filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Выводим итоговую точность
    print("\nИтоговая точность на тесте:", f"{results['final_metrics'].get('test_acc', 0):.4f}")
    print(f"Сохранено: {out_path / filename}")


if __name__ == "__main__":
    main()