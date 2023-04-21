import os
import numpy as np
from comtrade import Comtrade

test_directory = 'test_data'

def create_data_on_directory(directory):
    # Получение списка файлов Comtrade из директории
    files = [f for f in os.listdir(directory) if f.endswith('.cfg')]

    # Инициализация списков для хранения данных
    x_train_voltage_phase_a = []
    x_train_voltage_phase_b = []
    x_train_voltage_phase_c = []
    x_train_current_phase_a = []
    x_train_current_phase_b = []
    x_train_current_phase_c = []
    x_train_channel = []

    # Проход по каждому файлу и извлечение данных
    for file in files:
        # Загрузка файла Comtrade
        rec = Comtrade()
        rec.load(os.path.join(
            directory, file[:-4] + '.cfg'), os.path.join(directory, file[:-4] + '.dat'))

        # Извлечение данных о напряжении и токе
        current_phase_a = normalize_data(rec.analog[0])
        # Значений тока фазы B нет на данных сигналах, поэтому зануляем
        current_phase_b = np.zeros(rec.total_samples)
        current_phase_c = normalize_data(rec.analog[1])
        voltage_phase_a = normalize_data(rec.analog[2])
        voltage_phase_b = normalize_data(rec.analog[3])
        voltage_phase_c = normalize_data(rec.analog[4])

        #ml_signal_2_1_1 = get_digital_signal_by_name(rec, "MLsignal_2_1_1")
        ml_signal_1_1 = get_digital_signal_by_name(rec, "MLsignal_1_1")
        #ml_signal_1_2 = get_digital_signal_by_name(rec, "MLsignal_1_2")
        #ml_signal_2_1_3 = get_digital_signal_by_name(rec, "MLsignal_2_1_3")
        #ml_signal_3 = get_digital_signal_by_name(rec, "MLsignal_3")
        #start_osc = get_digital_signal_by_name(rec, "Пуск осциллографа")

        # Выбор канала для обучения
        train_channel = ml_signal_1_1

        # Если нет значений
        if (len(train_channel) == 0):
            # Значит аномалий по данному каналу нет
            train_channel = np.zeros(rec.total_samples)

        # Преобразование данных в формат, который можно использовать для обучения
        for i in range(len(current_phase_a)):
            x_train_voltage_phase_a.append([current_phase_a[i]])
            x_train_voltage_phase_b.append([current_phase_b[i]])
            x_train_voltage_phase_c.append([current_phase_c[i]])
            x_train_current_phase_a.append([voltage_phase_a[i]])
            x_train_current_phase_b.append([voltage_phase_b[i]])
            x_train_current_phase_c.append([voltage_phase_c[i]])
            x_train_channel.append([train_channel[i]])

    # Упаковка подготовленных данных
    data = [x_train_voltage_phase_a,
            x_train_voltage_phase_b,
            x_train_voltage_phase_c,
            x_train_current_phase_a,
            x_train_current_phase_b,
            x_train_current_phase_c,
            x_train_channel]

    return data


def normalize_data(data):
    max_value = max(data)
    min_value = min(data)
    normalized_data = []
    for value in data:
        normalized_value = (value - min_value) / \
            (max_value - min_value) * 2 - 1
        normalized_data.append(normalized_value)
    return normalized_data


def get_digital_signal_by_name(rec, name):
    index = get_channel_index_by_name(rec.digital_channel_ids, name)
    if index != -1:
        return rec.digital[index]
    else:
        return []


def get_analog_signal_by_name(rec, name, normalize=True):
    index = get_channel_index_by_name(rec.analog_channel_ids, name)
    if index != -1:
        data = rec.analog[index]
        if (normalize):
            return normalize_data(data)
        else:
            return data
    else:
        return []


def get_channel_index_by_name(string_list, name):
    for i, string in enumerate(string_list):
        if name in string:
            return i
    return -1


create_data_on_directory(test_directory)