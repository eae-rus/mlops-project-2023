import os
import numpy as np
import pandas as pd
from comtrade import Comtrade

filepath = os.path.abspath("..\..\data")



def create_data_on_directory(directory):
    # Получение списка файлов Comtrade из директории
    raw_filepath = directory + "\\raw"
    interim_filepath = directory + "\\interim"
    
    files = [f for f in os.listdir(raw_filepath) if f.endswith('.cfg')]

    # Инициализация общего датафрейма
    columns_name = ['filename',
                    'bus',
                    'current_phase_a', 'current_phase_c', 
                    'voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c', 'voltage_3U0', 
                    'ml_signal_1', 'ml_signal_1_1', 'ml_signal_1_2', 
                    'ml_signal_2', 'ml_signal_2_1', 'ml_signal_2_1_1', 'ml_signal_2_1_2', 'ml_signal_2_1_3', 
                    'ml_signal_3', 'ml_signal_3_1', 'ml_signal_3_2', 
                    'ml_anomaly']
    df = pd.DataFrame(columns=columns_name)


    # Проход по каждому файлу и извлечение данных
    for file in files:
        # Загрузка файла Comtrade
        rec = Comtrade()
        rec.load(os.path.join(
            raw_filepath, file[:-4] + '.cfg'), os.path.join(raw_filepath, file[:-4] + '.dat'), encoding='utf-8')

        # FIXME: оформить в нормальный список / массив
        current_phase_a = np.zeros(rec.total_samples)
        current_phase_c = np.zeros(rec.total_samples)
        voltage_phase_a = np.zeros(rec.total_samples)
        voltage_phase_b = np.zeros(rec.total_samples)
        voltage_phase_c = np.zeros(rec.total_samples)
        voltage_3U0     = np.zeros(rec.total_samples)
        ml_signal_1     = np.zeros(rec.total_samples)
        ml_signal_1_1   = np.zeros(rec.total_samples)
        ml_signal_1_2   = np.zeros(rec.total_samples)
        ml_signal_2     = np.zeros(rec.total_samples)
        ml_signal_2_1   = np.zeros(rec.total_samples)
        ml_signal_2_1_1 = np.zeros(rec.total_samples)
        ml_signal_2_1_2 = np.zeros(rec.total_samples)
        ml_signal_2_1_3 = np.zeros(rec.total_samples)
        ml_signal_3     = np.zeros(rec.total_samples)
        ml_signal_3_1   = np.zeros(rec.total_samples)
        ml_signal_3_2   = np.zeros(rec.total_samples)
        ml_anomaly      = np.zeros(rec.total_samples)

        for bus in range(1,3):
            # Извлечение данных о напряжении и токе
            name = f"IA {bus}ВВ"
            current_phase_a = get_analog_signal_by_name(rec, f"IA {bus}ВВ", True, 5, name_two = f"IA{bus}") # FIXME: исправить для любых названий
            current_phase_c = get_analog_signal_by_name(rec, f"IC {bus}ВВ", True, 5, name_two = f"IC{bus}") # FIXME: исправить для любых названий
            voltage_phase_a = get_analog_signal_by_name(rec, f"UA{bus}СШ",  True, 100)
            voltage_phase_b = get_analog_signal_by_name(rec, f"UB{bus}СШ",  True, 100)
            voltage_phase_c = get_analog_signal_by_name(rec, f"UC{bus}СШ",  True, 100)
            voltage_3U0     = get_analog_signal_by_name(rec, f"UC{bus}СШ",  True, 100)

            # FIXME: Нормально оформить список всех событий, вероятно - сделать извлекаемым не вручну.
            ml_signal_1     = get_digital_signal_by_name(rec, f"MLsignal_{bus}_1")
            ml_signal_1_1   = get_digital_signal_by_name(rec, f"MLsignal_{bus}_1_1")
            ml_signal_1_2   = get_digital_signal_by_name(rec, f"MLsignal_{bus}_1_2")
            ml_signal_2     = get_digital_signal_by_name(rec, f"MLsignal_{bus}_2")
            ml_signal_2_1   = get_digital_signal_by_name(rec, f"MLsignal_{bus}_2_1")
            ml_signal_2_1_1 = get_digital_signal_by_name(rec, f"MLsignal_{bus}_2_1_1")
            ml_signal_2_1_2 = get_digital_signal_by_name(rec, f"MLsignal_{bus}_2_1_1")
            ml_signal_2_1_3 = get_digital_signal_by_name(rec, f"MLsignal_{bus}_2_1_2")
            ml_signal_3     = get_digital_signal_by_name(rec, f"MLsignal_{bus}_3")
            ml_signal_3_1   = get_digital_signal_by_name(rec, f"MLsignal_{bus}_3_1")
            ml_signal_3_2   = get_digital_signal_by_name(rec, f"MLsignal_{bus}_3_2")

            # FIXME: Сделать функцию объединения каналов в единую переменую нормальной
            for i in range(rec.total_samples):
                if     ((ml_signal_2[i] == 1) or (ml_signal_2_1[i] == 1) or (ml_signal_2_1_1[i] == 1) or (ml_signal_2_1_2[i] == 1) or (ml_signal_2_1_3[i] == 1) or
                        (ml_signal_3[i] == 1) or (ml_signal_3_1[i] == 1) or (ml_signal_3_2[i] == 1)):
                    ml_anomaly[i] = 1

            # Преобразование данных в формат, который можно использовать для обучения
            # FIXME: оформить нормально объединение
            file_name_array = np.array([file[:-4]]*rec.total_samples)
            bus_name_array  = np.array([bus]*rec.total_samples)
            
            new_row = {'filename': file_name_array,
                       'bus': bus_name_array,
                       'current_phase_a': current_phase_a, 'current_phase_c': current_phase_c, 
                       'voltage_phase_a': voltage_phase_a, 'voltage_phase_b': voltage_phase_b, 'voltage_phase_c': voltage_phase_c, 'voltage_3U0': voltage_3U0,
                       'ml_signal_1': ml_signal_1, 'ml_signal_1_1': ml_signal_1_1, 'ml_signal_1_2': ml_signal_1_2, 
                       'ml_signal_2': ml_signal_2, 'ml_signal_2_1': ml_signal_2_1, 'ml_signal_2_1_1': ml_signal_2_1_1, 'ml_signal_2_1_2': ml_signal_2_1_2, 'ml_signal_2_1_3': ml_signal_2_1_3,
                       'ml_signal_3': ml_signal_3, 'ml_signal_3_1': ml_signal_3_1, 'ml_signal_3_2': ml_signal_3_2, 
                       'ml_anomaly': ml_anomaly}
            
            df_local = pd.DataFrame(new_row)
            
            
            df = pd.concat([df, df_local], ignore_index=True)

    # Сохранение данных в файл .csv через pandas
    df.to_csv(os.path.join(interim_filepath, 'data.csv'))

    return df


def normalize_data(data, normalize_level=100):
    # FIXME: normalize_level заменить на нормальный список типов (ток=5/напряжение=100)
    return data / normalize_level


def get_digital_signal_by_name(rec, name):
    index = get_channel_index_by_name(rec.digital_channel_ids, name)
    if index != -1:
        return np.array(rec.digital[index])
    else:
        return np.zeros(rec.total_samples)


def get_analog_signal_by_name(rec, name, is_normalize=True, normalize_level=100, name_two=""):
    index = get_channel_index_by_name(rec.analog_channel_ids, name)
    if index == -1: # FIXME: временный костыль (кривые данные...)
        index = get_channel_index_by_name(rec.analog_channel_ids, name_two)
    
    if index != -1:
        data = np.array(rec.analog[index])
        if (is_normalize):
            return normalize_data(data, normalize_level)
        else:
            return data
    else:
        return np.zeros(rec.total_samples)


def get_channel_index_by_name(string_list, name):
    for i, string in enumerate(string_list):
        if name in string:
            return i
    return -1


create_data_on_directory(filepath)
