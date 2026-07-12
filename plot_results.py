import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_histories(results_dir: Path, model: str | None = None) -> list[dict]:
    """
    Загружает историю обучения из JSON-файлов результатов.
    
    Аргументы:
        results_dir (Path): Директория с результатами экспериментов.
        model (str | None): Фильтр по типу модели (GCN или GAT).
    
    Возвращает:
        list[dict]: Список словарей с историей обучения и метриками.
    """
    rows = []
    for path in sorted(results_dir.glob("Cora_*.json")):
        if path.name == "summary.json":
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = data["config"]
        cfg_model = cfg.get("model", "gcn")
        if model and cfg_model != model:
            continue
        rows.append(
            {
                "label": f"{cfg['aggregation']} / {cfg['attack']}",
                "history": data.get("history", []),
                "final_test": data["final_metrics"].get("test_acc", 0),
            }
        )
    return rows


def plot_learning_curves(rows: list[dict], output: Path, title: str) -> None:
    """
    Строит графики кривых обучения (точность на тесте от раунда).
    
    Аргументы:
        rows (list[dict]): Данные истории обучения.
        output (Path): Путь для сохранения графика.
        title (str): Заголовок графика.
    """
    plt.figure(figsize=(9, 5))
    for row in rows:
        rounds = [h["round"] for h in row["history"]]
        test_acc = [h.get("test_acc", 0) for h in row["history"]]
        if rounds:
            plt.plot(rounds, test_acc, marker="o", label=row["label"])

    plt.xlabel("Раунд федеративного обучения")  # Подпись оси X
    plt.ylabel("Точность на тестовой выборке")  # Подпись оси Y
    plt.title(title)  # Заголовок
    plt.grid(True, alpha=0.3)  # Сетка
    plt.legend(fontsize=8)  # Легенда
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_bar_summary(summary: list[dict], output: Path, title: str, model: str | None = None) -> None:
    """
    Строит столбчатую диаграмму сравнения итоговой точности.
    
    Аргументы:
        summary (list[dict]): Сводные данные по всем экспериментам.
        output (Path): Путь для сохранения графика.
        title (str): Заголовок графика.
        model (str | None): Фильтр по типу модели.
    """
    if model:
        summary = [r for r in summary if r.get("model", "gcn") == model]
    if not summary:
        return

    # Подписи для оси X: агрегация / атака
    labels = [f"{r['aggregation']}\n{r['attack']}" for r in summary]
    values = [r["test_acc"] for r in summary]

    plt.figure(figsize=(10, 5))
    # Цвета для столбцов: синий, оранжевый, зелёный, красный, фиолетовый
    colors = ["blue", "orange", "green", "red", "purple"]
    bars = plt.bar(range(len(values)), values, color=colors[: len(values)])
    
    plt.xticks(range(len(labels)), labels, rotation=0, fontsize=8)
    plt.ylabel("Итоговая точность на тесте")  # Подпись оси Y
    plt.title(title)  # Заголовок
    plt.ylim(0, 1)  # Ограничение оси Y от 0 до 1
    
    # Подписи значений над столбцами
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", 
                ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_model_comparison(summary: list[dict], output: Path) -> None:
    """
    Строит сравнительные графики GCN vs GAT для различных агрегаций и атак.
    
    Аргументы:
        summary (list[dict]): Сводные данные по всем экспериментам.
        output (Path): Путь для сохранения графика.
    """
    attacks = sorted({r["attack"] for r in summary})
    aggregations = sorted({r["aggregation"] for r in summary})
    models = sorted({r.get("model", "gcn") for r in summary})

    x = range(len(aggregations))
    width = 0.35
    plt.figure(figsize=(12, 5))

    for attack_idx, attack in enumerate(attacks):
        plt.subplot(1, len(attacks), attack_idx + 1)
        for i, model in enumerate(models):
            values = []
            for agg in aggregations:
                match = [r for r in summary if r.get("model") == model and r["aggregation"] == agg and r["attack"] == attack]
                values.append(match[0]["test_acc"] if match else 0)
            offset = (i - (len(models) - 1) / 2) * width
            bars = plt.bar([xi + offset for xi in x], values, width, label=model.upper())
            for bar, val in zip(bars, values):
                if val > 0:
                    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
        
        plt.xticks(list(x), aggregations, rotation=30, ha="right", fontsize=8)
        plt.ylabel("Точность на тесте")  # Подпись оси Y
        plt.title(f"Тип атаки: {attack}")  # Заголовок
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Сравнение GCN и GAT при различных методах агрегации")  # Общий заголовок
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main():
    """
    Основная функция для построения всех графиков сравнения.
    """
    parser = argparse.ArgumentParser(description="Построение графиков сравнения результатов бенчмарка")
    parser.add_argument("--results-dir", default="results/benchmark", help="Директория с результатами экспериментов")
    parser.add_argument("--output-dir", default="results/figures", help="Директория для сохранения графиков")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка сводных данных
    summary_path = results_dir / "summary.json"
    summary = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    # Построение графиков для каждой модели
    for model in sorted({r.get("model", "gcn") for r in summary} or ["gcn", "gat"]):
        rows = load_histories(results_dir, model=model)
        if rows:
            plot_learning_curves(rows, output_dir / f"learning_curves_{model}.png", title=f"Федеративное обучение {model.upper()}: робастная агрегация при атаках")
            print(f"Сохранён график: {output_dir / f'learning_curves_{model}.png'}")

        if summary:
            plot_bar_summary(summary, output_dir / f"final_comparison_{model}.png", title=f"Итоговая точность на тесте ({model.upper()})", model=model)
            print(f"Сохранён график: {output_dir / f'final_comparison_{model}.png'}")

    # Сравнение моделей
    if summary:
        plot_model_comparison(summary, output_dir / "gcn_vs_gat.png")
        print(f"Сохранён график: {output_dir / 'gcn_vs_gat.png'}")


if __name__ == "__main__":
    main()