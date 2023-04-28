import numpy as np
import pandas as pd


class SignalDataset:
    def __init__(self, filename, bus, rec) -> None:
        self.filename = filename
        self.bus = bus
        self.rec = rec
        self.analog_signals = [f'UA{bus}СШ', f'UB{bus}СШ', f'UC{bus}СШ', f'UC{bus}СШ']
        self.analog_signals_names = ['voltage_phase_a', 'voltage_phase_b', 'voltage_phase_c', 'voltage_3U0']
        self.digital_signals = [f'MLsignal_{bus}_1', f'MLsignal_{bus}_1_1', f'MLsignal_{bus}_1_2', f'MLsignal_{bus}_2',
                                f'MLsignal_{bus}_2_1', f'MLsignal_{bus}_2_1_1',
                                f'MLsignal_{bus}_2_1_2', f'MLsignal_{bus}_2_1_3', f'MLsignal_{bus}_3',
                                f'MLsignal_{bus}_3_1', f'MLsignal_{bus}_3_2']
        self.digital_signals_add = ['MLsignal_12_1', 'MLsignal_12_1_1', 'MLsignal_12_1_2', 'MLsignal_12_2',
                                    'MLsignal_12_2_1', 'MLsignal_12_2_1_1', 'MLsignal_12_2_1_2',
                                    'MLsignal_12_2_1_3', 'MLsignal_12_3', 'MLsignal_12_3_1', 'MLsignal_12_3_2']
        self.digital_signals_names = ['ml_signal_1', 'ml_signal_1_1', 'ml_signal_1_2', 'ml_signal_2', 'ml_signal_2_1',
                                      'ml_signal_2_1_1', 'ml_signal_2_1_2',
                                      'ml_signal_2_1_3', 'ml_signal_3', 'ml_signal_3_1', 'ml_signal_3_2']

    def get_features(self) -> pd.DataFrame:
        dataset = pd.DataFrame()

        analog_features = self.make_analog_features(self.rec, self.analog_signals_names, self.analog_signals)
        for k, v in analog_features.items():
            dataset[k] = v

        digital_features = self.make_digital_features(self.rec, self.digital_signals_names, self.digital_signals,
                                                      self.digital_signals_add)
        for k, v in digital_features.items():
            dataset[k] = v

        dataset['ml_anomaly'] = np.where(dataset[self.digital_signals_names[3:]].sum(axis=1) > 0, 1, 0)

        dataset['filename'] = self.filename
        dataset['bus'] = self.bus

        return dataset

    def make_analog_features(self, rec, analog_signals_names, analog_signals) -> dict:
        analog_features = dict()
        analog_features['current_phase_a'] = self.get_analog_signal_by_name(rec, f"IA {self.bus}ВВ", True, 5,
                                                                            name_two=f"IA{self.bus}")      # FIXME: исправить для любых названий
        analog_features['current_phase_c'] = self.get_analog_signal_by_name(rec, f"IC {self.bus}ВВ", True, 5,
                                                                            name_two=f"IC{self.bus}")
        for i in range(len(analog_signals_names)):
            analog_features[analog_signals_names[i]] = self.get_analog_signal_by_name(rec, analog_signals[i], True, 100)
        return analog_features

    def make_digital_features(self, rec, digital_signals_names, digital_signals, digital_signals_add) -> dict:
        digital_features = dict()
        for i in range(len(digital_signals_names)):
            digital_features[digital_signals_names[i]] = self.get_digital_signal_by_name(rec, digital_signals[i],
                                                                                         digital_signals_add[i])
        return digital_features

    def normalize_data(self, data, normalize_level=100) -> np.ndarray:
        # FIXME: normalize_level заменить на нормальный список типов (ток=5/напряжение=100)
        return np.array(data/normalize_level)

    def get_digital_signal_by_name(self, rec, name, name_two='') -> np.ndarray:
        index = self.get_channel_index_by_name(rec.status_channel_ids, name)
        if index == -1:  # FIXME: временный костыль для добавления сигналов межсекционных (12)

            index = self.get_channel_index_by_name(rec.status_channel_ids, name_two)
        if index != -1:
            return np.array(rec.status[index])
        else:
            return np.zeros(rec.total_samples)

    def get_analog_signal_by_name(self, rec, name, is_normalize=True, normalize_level=100, name_two=None) -> np.ndarray:
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

    def get_channel_index_by_name(self, string_list, name) -> int:
        if name in string_list:
            return string_list.index(name)
        return -1
