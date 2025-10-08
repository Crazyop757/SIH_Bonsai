import numpy as np
import pandas as pd
from enhanced_fra_generator import EnhancedFRAGenerator

class SCADARealisticGenerator:
    def __init__(self, real_baseline=None):
        # real_baseline: DataFrame of public SCADA patterns (optional)
        self.real_baseline = real_baseline
        self.fra_gen = EnhancedFRAGenerator()

    def generate(self, n_samples=10000):
        samples = []
        for _ in range(n_samples):
            freq, mag, ph, labels, meta, _ = self.fra_gen.generate_sample(
                transformer_type=np.random.choice(list(self.fra_gen.transformers)),
                vendor=np.random.choice(list(self.fra_gen.vendors)),
                fault_type=np.random.choice([None] + list(self.fra_gen.faults)),
                severity=np.random.uniform(0,0.9) if np.random.rand()>0.3 else 0.0
            )
            scada = self._map_to_scada(freq, mag, ph, labels, meta)
            samples.append(scada)
        return pd.DataFrame(samples)

    def _map_to_scada(self, freq, mag, ph, labels, meta):
        base_v = meta['voltage_kv'] * 1000
        base_i = meta['power_mva'] * 1000 / (base_v * np.sqrt(3))
        thd_v = np.std(mag[:10])
        thd_i = np.std(ph[:10])
        oil_top = 25 + labels['severity']*40 + np.random.randn()*3
        return {
            'timestamp': pd.Timestamp.now(),
            'transformer_id': f"T{meta['voltage_kv']}_{np.random.randint(1000)}",
            'voltage_l1': base_v + thd_v,
            'voltage_l2': base_v - thd_v,
            'current_l1': base_i + thd_i,
            'current_l2': base_i - thd_i,
            'frequency': 50 + np.random.randn()*0.1,
            'thd_voltage': thd_v,
            'thd_current': thd_i,
            'oil_temp_top': oil_top,
            'oil_temp_bottom': oil_top - 3 + np.random.randn(),
            'h2_gas_ppm': labels['severity']*20 + np.random.randn()*2,
            'severity': labels['severity'],
            'fault_type': labels['fault_type'],
            'anomaly': labels['anomaly'],
            'criticality': labels['criticality'],
            'voltage_kv': meta['voltage_kv'],
            'power_mva': meta['power_mva'],
            'age_years': meta['age_years']
        }

def load_public_datasets():
    # ETT temperature dataset
    ett = pd.read_csv("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    # Optionally download Kaggle datasets separately
    return ett
