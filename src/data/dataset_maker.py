import numpy as np
import pandas as pd


class FeatureMaker:

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.harmonic_number = 6
        self.analog_signal_name = ['current_phase_a', 'current_phase_c',
                                   'voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c', 'voltage_3U0']

    # # FIXME: очень вероятно, стоит уменьшить числов гармоник до примерно 5

    def data_preparation_fft(self) -> pd.DataFrame:
        """
        Первичный спектральный анализ производится с окном равным периоду (32 выборки)
        На первичном этапе решено не извлекать частоты ниже 50Гц (промышленная частота),
        и 50Гц будет считаться первой.
        """
        length_df = self.dataset.shape[0]

        fft_analog_signal_name = []
        for signal_name in self.analog_signal_name:
            for harmonic in range(self.harmonic_number):
                fft_analog_signal_name.append(f'{signal_name}_h{harmonic}')

        # создаём новый пустой массив требуемой размерности для ускорения расчёта
        # first_row = np.zeros((length_df, len(fft_analog_signal_name)), dtype=float)
        first_row = np.full((length_df, len(fft_analog_signal_name)), np.nan, dtype=np.float32)
        new_df = pd.DataFrame(first_row, columns=fft_analog_signal_name, dtype=np.float32)

        fft_coefficient = np.sqrt(2) / 32

        window = self.dataset[self.analog_signal_name].rolling(window=32)
        for i, win_data in enumerate(window):
            if i < 32:
                continue
            f_amp_row = np.zeros(0, dtype=np.float32)
            # hostelCandidates1.equals(hostelCandidates2)
            # print(self.dataset['filename'][i - 32])
            # print(self.dataset['filename'][i])
            if self.dataset['filename'].values[i - 32] == self.dataset['filename'].values[i] and\
                    self.dataset['bus'].values[i - 32] == self.dataset['bus'].values[i]:
                for analog_name in self.analog_signal_name:
                    # FIXME: вероятно добавить принудительное зануление (присвоение NaN) сигналам ниже какого-то порога
                    # для того чтобы удалить "шумы" и уменьшить вес файлов.
                    x = win_data[analog_name]
                    f_amp = np.abs(np.fft.fft(x)[0:self.harmonic_number]) * fft_coefficient
                    f_amp_row = np.concatenate((f_amp_row, f_amp), axis=0)
            else:
                f_amp_row = np.zeros(len(fft_analog_signal_name))
            new_row = dict(zip(fft_analog_signal_name, f_amp_row))
            new_df.loc[i] = new_row

        return new_df
