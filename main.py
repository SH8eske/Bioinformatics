import pandas as pd
import numpy as np
from scipy import stats


def load_data(file_path):
    """
    Загружает данные из CSV-файла
    Вход:
    - file_path: str - путь к CSV-файлу с данными
    Выход:
    - pd.DataFrame - загруженные данные
    """
    return pd.read_csv(file_path)


def split_data(df, split_index):
    """
    Разделяет данные на нормальные и опухолевые образцы.
    Вход:
    - df: pd.DataFrame - исходные данные
    - split_index: int - индекс столбца, после которого начинаются опухолевые данные
    Выход:
    - tuple (normal_data, tumor_data) - два DataFrame с нормальными и опухолевыми данными
    """
    normal_data = df.iloc[:, 2:split_index]
    tumor_data = df.iloc[:, split_index:split_index + (df.shape[1] - split_index) // 2]
    return normal_data, tumor_data


def calculate_medians(norm_vals, tumor_vals):
    """
    Вычисляет медианы для нормальных и опухолевых значений.
    Вход:
    - norm_vals: np.array - массив нормальных значений
    - tumor_vals: np.array - массив опухолевых значений
    Выход:
    - tuple (median_norm, median_tumor) - медианы для нормальных и опухолевых значений
    """
    return np.median(norm_vals), np.median(tumor_vals)


def perform_shapiro_tests(norm_vals, tumor_vals):
    """
    Выполняет тесты Шапиро-Уилка на нормальность распределения.
    Вход:
    - norm_vals: np.array - массив нормальных значений
    - tumor_vals: np.array - массив опухолевых значений
    Выход:
    - tuple (shapiro_norm, shapiro_tumor, conj_shapiro):
        - shapiro_norm: результат теста для нормальных значений
        - shapiro_tumor: результат теста для опухолевых значений
        - conj_shapiro: bool - True если оба распределения нормальны (p-value > 0.05)
    """
    shapiro_norm = stats.shapiro(norm_vals)
    shapiro_tumor = stats.shapiro(tumor_vals)
    conj_shapiro = (shapiro_norm.pvalue > 0.05) and (shapiro_tumor.pvalue > 0.05)
    return shapiro_norm, shapiro_tumor, conj_shapiro


def perform_mannwhitney_test(norm_vals, tumor_vals):
    """
    Выполняет тест Манна-Уитни для сравнения распределений.
    Вход:
    - norm_vals: np.array - массив нормальных значений
    - tumor_vals: np.array - массив опухолевых значений
    Выход:
    - tuple (mannwhitney, medians_equal_mw):
        - mannwhitney: результат теста
        - medians_equal_mw: bool - True если медианы равны (p-value > 0.05)
    """
    mannwhitney = stats.mannwhitneyu(norm_vals, tumor_vals)
    medians_equal_mw = mannwhitney.pvalue > 0.05
    return mannwhitney, medians_equal_mw


def perform_wilcoxon_test(norm_vals, tumor_vals):
    """
    Выполняет тест Вилкоксона для парных сравнений.
    Вход:
    - norm_vals: np.array - массив нормальных значений
    - tumor_vals: np.array - массив опухолевых значений
    Выход:
    - tuple (wilcoxon_pval, medians_equal_wilcoxon):
        - wilcoxon_pval: p-value теста (nan если размеры массивов не совпадают)
        - medians_equal_wilcoxon: bool - True если медианы равны (p-value > 0.05)
    """
    if len(norm_vals) == len(tumor_vals):
        wilcoxon = stats.wilcoxon(norm_vals, tumor_vals)
        wilcoxon_pval = wilcoxon.pvalue
    else:
        wilcoxon_pval = np.nan
    medians_equal_wilcoxon = wilcoxon_pval > 0.05 if not np.isnan(wilcoxon_pval) else False
    return wilcoxon_pval, medians_equal_wilcoxon


def analyze_gene(gene_id, norm_vals, tumor_vals):
    """
    Анализирует данные для одного гена, выполняя все статистические тесты.
    Вход:
    - gene_id: str - идентификатор гена
    - norm_vals: np.array - массив нормальных значений
    - tumor_vals: np.array - массив опухолевых значений
    Выход:
    - dict - словарь с результатами всех тестов для гена
    """
    # Проверка наличия данных
    if len(norm_vals) == 0 or len(tumor_vals) == 0:
        return None

    # Вычисление медиан
    median_norm, median_tumor = calculate_medians(norm_vals, tumor_vals)

    # Тесты Шапиро-Уилка
    shapiro_norm, shapiro_tumor, conj_shapiro = perform_shapiro_tests(norm_vals, tumor_vals)

    # Тест Манна-Уитни
    mannwhitney, medians_equal_mw = perform_mannwhitney_test(norm_vals, tumor_vals)

    # Тест Вилкоксона
    wilcoxon_pval, medians_equal_wilcoxon = perform_wilcoxon_test(norm_vals, tumor_vals)

    # Конъюнкция критериев
    conj_tests = medians_equal_mw and medians_equal_wilcoxon

    return {
        'gene_id': gene_id,
        'median_normal': median_norm,
        'median_tumor': median_tumor,
        'shapiro_norm_pval': shapiro_norm.pvalue,
        'shapiro_tumor_pval': shapiro_tumor.pvalue,
        'conj_shapiro': conj_shapiro,
        'mannwhitney_pval': mannwhitney.pvalue,
        'medians_equal_mw': medians_equal_mw,
        'wilcoxon_pval': wilcoxon_pval,
        'medians_equal_wilcoxon': medians_equal_wilcoxon,
        'conj_tests': conj_tests
    }


def save_results(results, output_file):
    """
    Сохраняет результаты анализа в CSV-файл.

    Вход:
    - results: list[dict] - список словарей с результатами
    - output_file: str - путь к файлу для сохранения результатов
    """
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)


def main():
    """Основная функция выполнения анализа."""
    # Загрузка данных
    file_path = input("Введите путь к файлу с данными: ")
    df = load_data(file_path)

    # Разделение данных
    split_index = int(input("Введите индекс столбца, после которого начинаются опухолевые данные: "))
    normal_data, tumor_data = split_data(df, split_index)

    # Обработка каждого гена
    results = []
    for i in range(len(df)):
        gene_id = df.iloc[i, 1]
        norm_vals = normal_data.iloc[i].dropna().astype(float).values
        tumor_vals = tumor_data.iloc[i].dropna().astype(float).values

        # Анализ гена
        gene_result = analyze_gene(gene_id, norm_vals, tumor_vals)
        if gene_result:
            results.append(gene_result)

    # Сохранение результатов
    save_results(results, 'results.csv')
    print("Анализ завершен. Результаты сохранены в results.csv")


if __name__ == "__main__":
    main()