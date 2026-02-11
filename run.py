Запуск одной командой, добавить остальные агенты 

run_pipeline.py — единая команда запуска

python run_pipeline.py

import subprocess
import sys

def run(step_name, cmd):
    print(f"\n=== {step_name} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Ошибка на шаге: {step_name}")
        sys.exit(1)

if __name__ == "__main__":
    run(
        "A1: сбор данных и изображений",
        "python -m agents.a1_collect.main"
    )

    run(
        "A2: генерация признаков",
        "python -m agents.a2_features.main --radius_m 800"
    )

    run(
        "A3: алерты и агрегации",
        "python -m agents.a3_alerts.main"
    )

    run(
        "V1: обучение модели",
        "python -m agents.v1_train.main --model rf"
    )
    print("\nПайплайн успешно выполнен")
   
   Проект реализован в виде последовательного пайплайна, состоящего из нескольких агентов. Для упрощения воспроизводимости был реализован единый скрипт запуска, который последовательно выполняет все этапы обработки данных и обучения модели одной командой.
