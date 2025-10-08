class SCADAFeatureExtractor:
    def extract(self, row):
        features = {}
        features['VLF_mean'] = row.get('thd_voltage', 0)
        features['VLF_std'] = row.get('thd_current', 0)
        features['temp_grad'] = row.get('oil_temp_top', 0) - row.get('oil_temp_bottom', 0)
        h2 = row.get('h2_gas_ppm', 0)
        ch4 = row.get('ch4_gas_ppm', 1) or 1  # prevent div by zero
        features['h2_ratio'] = h2 / ch4
        features['voltage_kv'] = row.get('voltage_kv', 0)
        features['power_mva'] = row.get('power_mva', 0)
        features['age_years'] = row.get('age_years', 0)
        return features
