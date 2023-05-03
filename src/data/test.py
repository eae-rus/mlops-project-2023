import os
import numpy as np
import pandas as pd

filepath = os.path.abspath("..\..\data")
interim_filepath = filepath + "\\interim"

# FIXME: очень вероятно, стоит уменьшить числов гармоник до примерно 5
HARMONIC_NUMBER = 6  # изначально была половина спектра 32/2=16, согласно достаточности выборок

analog_signal_name = ['current_phase_a', 'current_phase_c',
                      'voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c', 'voltage_3U0']


def data_preparation_fft(df):
    '''
    Первичный спектральный анализ производится с окном равным периоду (32 выборки)
    На первичном этапе решено не извлекать частоты ниже 50Гц (промышленная частота),
    и 50Гц будет считаться первой.
    '''
    length_df = df.shape[0]

    fft_analog_signal_name = []
    for signal_name in analog_signal_name:
        for harmonic in range(HARMONIC_NUMBER):
            fft_analog_signal_name.append(f'{signal_name}_h{harmonic}')

    # создаём новый пустой массив требуемой размерности для ускорения расчёта
    # first_row = np.zeros((length_df, len(fft_analog_signal_name)), dtype=float)
    first_row = np.full((length_df, len(fft_analog_signal_name)), np.nan, dtype=np.float32)
    new_df = pd.DataFrame(first_row, columns=fft_analog_signal_name, dtype=np.float32)

    fft_coefficient = np.sqrt(2) / 32

    window = df[analog_signal_name].rolling(window=32)
    for i, win_data in enumerate(window):
        if i < 32:
            continue
        f_amp_row = np.zeros(0, dtype=np.float32)
        if ((df['filename'][i - 32] == df['filename'][i]) and
                (df['bus'][i - 32] == df['bus'][i])):
            for analog_name in analog_signal_name:
                # FIXME: вероятно добавить принудительное зануление (присвоение NaN) сигналам ниже какого-то порога
                # для того чтобы удалить "шумы" и уменьшить вес файлов.
                x = win_data[analog_name]
                f_amp = np.abs(np.fft.fft(x)[0:HARMONIC_NUMBER]) * fft_coefficient
                f_amp_row = np.concatenate((f_amp_row, f_amp), axis=0)
        else:
            f_amp_row = np.zeros(len(fft_analog_signal_name), dtype=np.float32)
        new_row = dict(zip(fft_analog_signal_name, f_amp_row))
        new_df.loc[i] = new_row
        # if i % 10000 == 0: # проверка на скорость, ориентировочно 9с на 10к точек
        #     print(i)

    return new_df


df = pd.read_csv('D:/MLOps_project/data/raw/test/data.csv')
fft_df_value = data_preparation_fft(df)
fft_df_value.to_csv('D:/MLOps_project/data/raw/test/data_fft_value.csv')
