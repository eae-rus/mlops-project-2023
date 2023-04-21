import os
import numpy as np
import pandas as pd

test_directory = 'test_data'
HARMONIC_NUMBER = 16 # половина спектра согласно достаточности выборок

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

    first_row = np.zeros((31, len(fft_analog_signal_name)))        
    new_df = pd.DataFrame(first_row, columns=fft_analog_signal_name)
    
    fft_coefficient = np.sqrt(2) / 32
    
    # FIXME: считает ОЧЕНЬ не рационально, требуется оптимизировать
    for i in range(32,length_df):
        f_amp_row = np.zeros(0)
        if ((df['filename'][i-32] == df['filename'][i]) and
            (df['bus'][i-32] == df['bus'][i])):
            for analog_name in analog_signal_name:
                x = df[i-32:i][analog_name]
                f_amp = np.abs(np.fft.fft(x)[0:HARMONIC_NUMBER])*fft_coefficient # пока решено использовать только амплитуды, в будущем, будут комплексные числа
                f_amp_row = np.concatenate((f_amp_row, f_amp), axis=0)
        else:
            f_amp_row = np.zeros(len(fft_analog_signal_name))
        new_row = dict(zip(fft_analog_signal_name, f_amp_row))
        new_df = new_df.append(new_row, ignore_index=True)
        # if i % 10000 == 0: # проверка на скорость
        #     print(i)
    
    return new_df


df = pd.read_csv(f'{test_directory}/data.csv')
fft_df_value = data_preparation_fft(df)
fft_df_value.to_csv(os.path.join(test_directory, 'data_fft_value.csv'))
