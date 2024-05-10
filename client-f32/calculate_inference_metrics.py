import json
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла JSON
with open('measurements.json', 'r') as f:
    prediction_times = json.load(f)

prediction_times = [float(time) / 1000 for time in prediction_times]
# Вычисление статистики
mean_time = np.mean(prediction_times)
median_time = np.median(prediction_times)
std_dev = np.std(prediction_times)

# Создание списка номеров запросов
requests = list(range(1, len(prediction_times) + 1))

# Создание графика времени предсказания для каждого запроса
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(requests, prediction_times, marker='o', linestyle='-', markersize=1, linewidth=1)
plt.xlabel('Номер вызова')
plt.ylabel('Время (секунды)')
plt.title('Время предсказания для каждого вызова\nСреднее: {:.4f} s, Медиана: {:.4f} s, Std Dev: {:.4f} s'.format(mean_time, median_time, std_dev))
plt.grid(True)

# Создание гистограммы распределения времени предсказания относительно медианного значения
plt.subplot(1, 2, 2)
plt.hist(np.array(prediction_times) - median_time, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Отклонение (секунды)')
plt.ylabel('Частота')
plt.title('Распределение отклонений от медианного значения'.format(np.mean(np.array(prediction_times) - median_time), np.std(np.array(prediction_times) - median_time)))
plt.grid(True)


plt.savefig('inference_analysis.png')
plt.tight_layout()
plt.show()


