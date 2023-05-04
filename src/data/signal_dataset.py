import numpy as np
import pandas as pd
from comtrade import Comtrade


class SignalDataset:
    def __init__(self, filename: str, bus: int, rec: Comtrade) -> None:
        self.filename = filename
        self.bus = bus
        self.rec = rec
        self.analog_signals = [f"IA{bus}ВВ", f"IC{bus}ВВ",
                               f'UA{bus}СШ', f'UB{bus}СШ', f'UC{bus}СШ']
        self.analog_signals_alternative = [f"IA {bus}ВВ", f"IC {bus}ВВ",
                                           f'UA {bus}СШ', f'UB {bus}СШ', f'UC {bus}СШ']
        self.analog_signals_names = ['current_phase_a', 'current_phase_c',
                                     'voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c']
        self.digital_signals = [f'MLsignal_{bus}_1', f'MLsignal_{bus}_1_1', f'MLsignal_{bus}_1_2',
                                f'MLsignal_{bus}_2', f'MLsignal_{bus}_2_1', f'MLsignal_{bus}_2_1_1',
                                f'MLsignal_{bus}_2_1_2', f'MLsignal_{bus}_2_1_3', f'MLsignal_{bus}_3',
                                f'MLsignal_{bus}_3_1', f'MLsignal_{bus}_3_2']
        self.digital_signals_add = ['MLsignal_12_1', 'MLsignal_12_1_1', 'MLsignal_12_1_2',
                                    'MLsignal_12_2', 'MLsignal_12_2_1', 'MLsignal_12_2_1_1', 'MLsignal_12_2_1_2',
                                    'MLsignal_12_2_1_3', 'MLsignal_12_3', 'MLsignal_12_3_1', 'MLsignal_12_3_2']
        self.digital_signals_names = ['ml_signal_1', 'ml_signal_1_1', 'ml_signal_1_2',
                                      'ml_signal_2', 'ml_signal_2_1', 'ml_signal_2_1_1', 'ml_signal_2_1_2',
                                      'ml_signal_2_1_3',
                                      'ml_signal_3', 'ml_signal_3_1', 'ml_signal_3_2']

    def get_features(self) -> pd.DataFrame:
        # TODO: для уменьшения веса создаваемых файлов стоит задать чётки типы данных,
        # для аналоговых достаточно float32, для дискретных bool/uint8/float16/float32...
        # Но данная корректировка может требовать иных правок в коде.
        dataset = pd.DataFrame()
        dataset['filename'] = self.filename
        dataset['bus'] = self.bus

        analog_features = self.make_analog_features(self.rec, self.analog_signals_names,
                                                    self.analog_signals, self.analog_signals_alternative)
        for k, v in analog_features.items():
            dataset[k] = v

        # TODO: 1) В моделях не стоит использовать совместно с фазными сигналами
        # 2) В будущих разработках "voltage_3U0" стоит извлекать из осциллограммы, для выявления повреждения
        # цепей измерения напряжения за счёт различия расчётного и реального сигналов.
        dataset['voltage_3U0'] = dataset[['voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c']].sum(axis=1)

        digital_features = self.make_digital_features(self.rec, self.digital_signals_names, self.digital_signals,
                                                      self.digital_signals_add)
        for k, v in digital_features.items():
            dataset[k] = v

        dataset['ml_anomaly'] = np.where(dataset[self.digital_signals_names[3:]].sum(axis=1) > 0, 1, 0)
        is_use_for_ml = np.where(dataset[self.digital_signals_names[:]].sum(axis=1) > 0, 1, 0)
        expansion_zone = 10 * self.rec.cfg.sample_rates[0][0] / self.rec.frequency
        dataset['is_use_for_ml'] = self.expand_useful_areas(is_use_for_ml, expansion_zone)
        # Не удалять filename и bus, чтобы поля не были пустыми
        dataset['filename'] = self.filename
        dataset['bus'] = self.bus

        return dataset

    def make_analog_features(self, rec: Comtrade, signals_names: list, signals: list, signals_add: list) -> dict:
        features = dict()
        # FIXME: исправить для любых названий, а не только "второго"
        for i in range(len(signals_names)):
            try:
                if 'current_phase' in signals_names[i]:
                    features[signals_names[i]] = self.get_analog_signal_by_name(rec, signals[i], True, 5,
                                                                                name_two=signals_add[i])
                elif 'voltage' in signals_names[i]:
                    features[signals_names[i]] = self.get_analog_signal_by_name(rec, signals[i], True, 100,
                                                                                name_two=signals_add[i])
            except Exception as e:
                # print("Неизвестное название аналогового сигнала. Пожалуйста, скорректируйте список типов.")
                print(e)
                # TODO: в будущем появятся In с номиналом 1А, и Пояса Роговского (скорее всего 0,1В)
        return features

    def make_digital_features(self, rec: Comtrade, signals_names: list,
                              signals: list, signals_add: list) -> dict:
        digital_features = dict()
        for i in range(len(signals_names)):
            digital_features[signals_names[i]] = self.get_digital_signal_by_name(rec, signals[i], signals_add[i])
        return digital_features

    def normalize_data(self, data: np.ndarray, normalize_level=100) -> np.ndarray:
        # FIXME: normalize_level заменить на нормальный список типов (ток=5/напряжение=100)
        return np.array(data / normalize_level)

    def get_digital_signal_by_name(self, rec: Comtrade, name: str, name_two='') -> np.ndarray:
        index = self.get_channel_index_by_name(rec.status_channel_ids, name)
        if index == -1:  # FIXME: временный костыль для добавления сигналов межсекционных (12)

            index = self.get_channel_index_by_name(rec.status_channel_ids, name_two)
        if index != -1:
            return np.array(rec.status[index])
        else:
            return np.zeros(rec.total_samples)

    def get_analog_signal_by_name(self, rec: Comtrade, name: str,
                                  is_normalize=True, normalize_level=100,
                                  name_two=None) -> np.ndarray:
        index = self.get_channel_index_by_name(rec.analog_channel_ids, name)
        if index == -1:  # FIXME: временный костыль (кривые данные...)
            index = self.get_channel_index_by_name(rec.analog_channel_ids, name_two)

        if index != -1:
            data = np.array(rec.analog[index])
            if is_normalize:
                return self.normalize_data(data, normalize_level)
            return data
        else:
            return np.zeros(rec.total_samples)

    def get_channel_index_by_name(self, string_list: list, name: str) -> int:
        if name in string_list:
            return string_list.index(name)
        return -1

    def expand_useful_areas(self, is_use_for_ml: np.array, expansion_zone: int) -> np.ndarray:
        # FIXME: Сделать функцию более оптимальной по вычислительной нагрузке
        len_signal = len(is_use_for_ml)

        if is_use_for_ml.sum() == 0:  # в осциллограмме нет интересных событий
            if len_signal > 2 * expansion_zone + 2:
                middle = len_signal / 2
                is_use_for_ml[int(middle - expansion_zone): int(middle + expansion_zone)] = 1
            else:  # для слишком маленьких осциллограмм
                is_use_for_ml[:] = 1
        else:  # в осциллограмме есть интересные события
            # левая граница
            for i in range(1, len_signal):
                if (is_use_for_ml[i - 1] == 0) and (is_use_for_ml[i] == 1):
                    left_border = int(i - expansion_zone)
                    if left_border < 0:
                        left_border = 0
                    is_use_for_ml[left_border:i] = 1
            # правая граница
            for i in range(len_signal - 1, 0, -1):
                if (is_use_for_ml[i] == 0) and (is_use_for_ml[i - 1] == 1):
                    right_border = int(i + expansion_zone)
                    if right_border > len_signal - 1:
                        right_border = len_signal
                    is_use_for_ml[i:right_border] = 1

        return is_use_for_ml
