# STFGMLMAGNN

Исследование устойчивости **федеративных графовых нейронных сетей** к отравлению данных и сравнение робастных методов серверной агрегации.

## Что реализовано

- **Модели:** GCN и GAT (PyTorch Geometric)
- **FL:** FedAvg + Median + Trimmed Mean + Krum
- **Датасет:** Cora (Planetoid), разбиение на клиентов (IID / non-IID Dirichlet)
- **Атаки:** label flip, scaled model update
- **Оценка:** глобальная accuracy на train / val / test

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

При наличии GPU замените первую строку установки torch на версию с CUDA.

## Быстрый запуск

Baseline без атаки:

```bash
python train.py --aggregation fedavg --attack none --rounds 30
```

Атака label-flip + робастная агрегация:

```bash
python train.py --aggregation trimmed_mean --attack label_flip --num-malicious 1 --rounds 30
```

Non-IID разбиение (по умолчанию):

```bash
python train.py --model gat --aggregation median --attack label_flip --dirichlet-alpha 0.3
```

IID разбиение:

```bash
python train.py --aggregation fedavg --attack label_flip --iid
```

## Полный бенчмарк

Сравнивает FedAvg / Median / Trimmed Mean / Krum при атаке и без:

```bash
python run_benchmark.py
```

Результаты сохраняются в `results/`.

## Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--model` | `gcn` или `gat` | `gcn` |
| `--num-clients` | Число клиентов | 5 |
| `--num-malicious` | Число злых клиентов | 1 |
| `--aggregation` | `fedavg`, `median`, `trimmed_mean`, `krum` | `fedavg` |
| `--attack` | `none`, `label_flip`, `scaled_update` | `none` |
| `--flip-ratio` | Доля перевёрнутых меток | 1.0 |
| `--trim-ratio` | Доля отбрасываемых клиентов (Trimmed Mean) | 0.2 |
| `--rounds` | Раундов FL | 50 |
| `--local-epochs` | Локальных эпох на раунд | 3 |

## Структура проекта

```
src/
  models/gcn.py          # GCN
  models/gat.py          # GAT
  data/partition.py      # разбиение Cora на клиентов
  federated/             # клиент и сервер FL
  aggregation.py         # FedAvg, Median, Trimmed Mean, Krum
  attacks/               # label flip, scaled update
train.py                 # один эксперимент
run_benchmark.py         # серия экспериментов для курсовой (GCN + GAT, 50 раундов)
plot_results.py          # графики из results/benchmark/
```

## Результаты бенчмарка (50 раундов, non-IID)

| Модель | FedAvg | FedAvg + flip | Median + flip | Trimmed Mean + flip | Krum + flip |
|--------|--------|---------------|---------------|---------------------|-------------|
| GCN    | 0.777  | 0.647         | 0.402         | 0.572               | 0.360       |
| GAT    | 0.771  | 0.637         | 0.535         | 0.547               | 0.371       |

Графики: `results/figures/`

## Для курсовой

Рекомендуемая последовательность экспериментов:

1. FedAvg, без атаки — baseline
2. FedAvg + label_flip — демонстрация уязвимости
3. Median / Trimmed Mean + label_flip — защита
4. Повтор п. 2–3 с `--iid` и без (non-IID по умолчанию)
5. Варьирование `--num-malicious` (1, 2) и `--flip-ratio` (0.5, 1.0)
