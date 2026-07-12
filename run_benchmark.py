"""
Полный бенчмарк: GCN и GAT, 50 раундов федеративного обучения,
робастная агрегация при атаке подмены меток (label-flip).
"""

import json
import subprocess
import sys
from pathlib import Path


# Список моделей для тестирования
MODELS = ["gcn", "gat"]

# Список экспериментов: комбинации методов агрегации и типов атак
EXPERIMENTS = [
    {"aggregation": "fedavg", "attack": "none"},          # Базовый случай без атак
    {"aggregation": "fedavg", "attack": "label_flip"},    # FedAvg + атака
    {"aggregation": "median", "attack": "label_flip"},    # Медиана + атака
    {"aggregation": "trimmed_mean", "attack": "label_flip"},  # Усечённое среднее + атака
    {"aggregation": "krum", "attack": "label_flip"},      # Krum + атака
]


def main():
    """
    Основная функция запуска бенчмарка.
    Выполняет все эксперименты, собирает результаты и сохраняет сводку.
    """
    # Создаём директорию для сохранения результатов
    output_dir = Path("results/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # Запускаем эксперименты для каждой модели и каждой конфигурации
    for model in MODELS:
        for exp in EXPERIMENTS:
            # Формируем команду для запуска обучения
            cmd = [
                sys.executable,              # Интерпретатор Python
                "train.py",                  # Скрипт обучения
                "--model", model,            # Тип модели
                "--aggregation", exp["aggregation"],  # Метод агрегации
                "--attack", exp["attack"],   # Тип атаки
                "--num-malicious", "1",      # Количество вредоносных клиентов
                "--rounds", "50",            # Количество раундов
                "--log-every", "10",         # Интервал логирования
                "--output-dir", str(output_dir)  # Директория для сохранения
            ]
            print("\n>>> Запуск:", " ".join(cmd))
            # Запускаем процесс обучения и ждём его завершения
            subprocess.run(cmd, check=True)

    # Сбор всех результатов из JSON-файлов
    for path in sorted(output_dir.glob("Cora_*.json")):
        if path.name == "summary.json":
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = data["config"]
        # Добавляем результат в сводку
        summary.append(
            {
                "file": path.name,           # Имя файла
                "model": cfg.get("model", "gcn"),  # Тип модели
                "aggregation": cfg["aggregation"], # Метод агрегации
                "attack": cfg["attack"],     # Тип атаки
                "test_acc": data["final_metrics"].get("test_acc"),  # Точность на тесте
                "val_acc": data["final_metrics"].get("val_acc"),    # Точность на валидации
            }
        )

    # Сохраняем сводку в JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Выводим сводку в консоль
    print("\n" + "=" * 70)
    print("Сводка результатов бенчмарка (50 раундов, Non-IID, 1 вредоносный клиент)")
    print("=" * 70)
    for row in sorted(summary, key=lambda r: (r["model"], r["aggregation"], r["attack"])):
        print(
            f"{row['model'].upper():4s} | {row['aggregation']:14s} | "
            f"атака={row['attack']:12s} | тест={row['test_acc']:.4f} | валидация={row['val_acc']:.4f}"
        )
    print(f"Сохранено: {summary_path}")


if __name__ == "__main__":
    main()