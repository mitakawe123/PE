#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, InterpolatedUnivariateSpline
from scipy.stats import skew, kurtosis

def main():
    # Данни: дни от 1 до 30
    days = np.arange(1, 31)

    # Температурни данни – липсващи стойности обозначени с np.nan
    temperature = np.array([
        20.30, 22.30, 23.67, 25.98, np.nan, 28.46, 29.21, 30.05, 29.80, 29.56, 
        28.96, np.nan, 25.68, 24.22, 21.98, 20.05, 17.72, 16.23, np.nan, 12.72, 
        11.24, 10.69, 9.80, 10.15, 10.34, np.nan, 12.62, 14.02, np.nan, 18.12
    ])

    # Енергопотребление
    energy = np.array([
        110.20, 104.20, 104.66, 97.54, 96.64, 91.08, 92.08, 92.10, 89.40, 91.88,
        91.58, 93.34, 99.14, 100.76, 107.24, 109.40, 114.86, 116.04, 122.66, 123.56,
        129.02, 127.92, 130.70, 128.50, 130.12, 126.42, 125.76, 121.26, 119.94, 112.26
    ])

    ### 1. Импутация на липсващите стойности с Лагранжов полином и кубичен сплайн

    # Извличане на дните с налични данни
    valid_indices = ~np.isnan(temperature)
    days_valid = days[valid_indices]
    temp_valid = temperature[valid_indices]

    # Лагранжова интерполация
    lagrange_poly = lagrange(days_valid, temp_valid)
    temperature_lagrange = lagrange_poly(days)

    # Сплайн интерполация (кубичен сплайн, k=3)
    spline = InterpolatedUnivariateSpline(days_valid, temp_valid, k=3)
    temperature_spline = spline(days)

    # Показване на дните с липсващи стойности и интерполираните им стойности:
    missing_days = days[~valid_indices]
    print("Дни с липсващи температурни стойности:", missing_days)
    print("Липсващи стойности според Лагранжова интерполация:", temperature_lagrange[~valid_indices])
    print("Липсващи стойности според Сплайн интерполация:", temperature_spline[~valid_indices])
    print("-" * 60)

    ### 2. Сравнение между Лагранжов полином и Сплайн функция
    # (Визуално сравнение чрез графики)
    
    plt.figure(figsize=(10,6))
    plt.plot(days, temperature_spline, label='Сплайн интерполация', color='blue', lw=2)
    plt.plot(days, temperature_lagrange, label='Лагранжов полином', color='red', linestyle='--', lw=2)
    plt.scatter(days_valid, temp_valid, label='Наблюдавани данни', color='black', zorder=5)
    plt.xlabel("Ден")
    plt.ylabel("Температура (°C)")
    plt.title("Интерполация на температура: Сплайн vs. Лагранж")
    plt.legend()
    plt.grid(True)
    plt.savefig("temperature_interpolation.png")
    plt.show()

    ### 3. Статистически анализ за температурните данни и енергопотреблението

    # За температурите използваме само наличните (наблюдавани) стойности:
    temp_mean = np.mean(temp_valid)
    temp_variance = np.var(temp_valid, ddof=1)
    temp_std = np.std(temp_valid, ddof=1)
    temp_skew = skew(temp_valid)
    temp_kurt = kurtosis(temp_valid)

    # За енергопотреблението (данни за всички дни)
    energy_mean = np.mean(energy)
    energy_variance = np.var(energy, ddof=1)
    energy_std = np.std(energy, ddof=1)
    energy_skew = skew(energy)
    energy_kurt = kurtosis(energy)

    print("Статистика за температура (наблюдавани):")
    print("Средна стойност: {:.2f}, Дисперсия: {:.2f}, Стандартно отклонение: {:.2f}, Асиметрия: {:.2f}, Ексцес: {:.2f}"
          .format(temp_mean, temp_variance, temp_std, temp_skew, temp_kurt))
    print("\nСтатистика за енергопотреблението:")
    print("Средна стойност: {:.2f}, Дисперсия: {:.2f}, Стандартно отклонение: {:.2f}, Асиметрия: {:.2f}, Ексцес: {:.2f}"
          .format(energy_mean, energy_variance, energy_std, energy_skew, energy_kurt))
    print("-" * 60)

    ### 4. Ковариация и корелация между интерполираната (сплайн) температура и енергопотреблението
    cov_matrix = np.cov(temperature_spline, energy)
    covariance = cov_matrix[0, 1]
    correlation = np.corrcoef(temperature_spline, energy)[0, 1]

    print("Ковариация между сплайн температура и енергопотребление: {:.2f}".format(covariance))
    print("Корелация между сплайн температура и енергопотребление: {:.2f}".format(correlation))
    print("-" * 60)

    ### 5. Допълнителна визуализация: Енергопотребление през 30 дни
    plt.figure(figsize=(10,6))
    plt.plot(days, energy, label='Енергопотребление', color='green', marker='o', lw=2)
    plt.xlabel("Ден")
    plt.ylabel("Енергопотребление (единици)")
    plt.title("Енергопотребление през 30 дни")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_consumption.png")
    plt.show()

if __name__ == "__main__":
    main()
