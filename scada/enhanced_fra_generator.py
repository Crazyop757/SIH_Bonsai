import numpy as np

FREQ_BANDS = [(20, 1e3), (1e3, 1e5), (1e5, 2e6), (2e6, 25e6)]
BAND_NAMES = ["VLF", "LF", "MF", "HF"]  # Very Low, Low, Med, High Freq bands

def extract_features(freq, mag_db, phase_deg):
    features = {}
    for (f_start, f_end), band in zip(FREQ_BANDS, BAND_NAMES):
        mask = (freq >= f_start) & (freq < f_end)
        submag = mag_db[mask]
        subph = phase_deg[mask]
        if submag.size > 0:
            features[f"mag_mean_{band}"] = np.mean(submag)
            features[f"mag_std_{band}"] = np.std(submag)
            features[f"mag_min_{band}"] = np.min(submag)
            features[f"mag_max_{band}"] = np.max(submag)
            features[f"ph_mean_{band}"] = np.mean(subph)
            features[f"ph_std_{band}"] = np.std(subph)
            features[f"ph_min_{band}"] = np.min(subph)
            features[f"ph_max_{band}"] = np.max(subph)
        else:
            for base in ["mag_mean_", "mag_std_", "mag_min_", "mag_max_", "ph_mean_", "ph_std_", "ph_min_", "ph_max_"]:
                features[base + band] = 0.0
    return features

class EnhancedFRAGenerator:
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.transformers = {
            'distribution_11kv': {'segments':8, 'R_base':0.02, 'L_base':0.001, 'C_base':1e-9, 'Rc':10000, 'Lm':5.0},
            'power_132kv': {'segments':12, 'R_base':0.5, 'L_base':0.1, 'C_base':5e-11, 'Rc':50000, 'Lm':50.0},
            'transmission_400kv': {'segments':16, 'R_base':2.0, 'L_base':0.5, 'C_base':1e-11, 'Rc':100000, 'Lm':200.0}
        }
        self.faults = {
            'axial_displacement': {'bands': (1e3, 1e6), 'dL_factor': 0.15, 'dCw_factor': -0.1},
            'radial_deformation': {'bands': (1e4, 2e6), 'dC_factor': -0.25, 'dCw_factor': 0.2},
            'core_grounding': {'bands': (20, 2e3), 'dRc_factor': -0.5, 'dLm_factor': -0.1},
            'shorted_turns': {'bands': (100, 1e5), 'dR1_factor': 0.2, 'dL1_factor': -0.2}
        }
        self.vendors = {
            'omicron': {'fmin':20, 'fmax':25e6, 'points':1000, 'mag_noise':(0.1,0.5), 'ph_noise':(1,5)},
            'megger': {'fmin':20, 'fmax':10e6, 'points':800, 'mag_noise':(0.15,0.4), 'ph_noise':(1.5,4)},
            'doble': {'fmin':20, 'fmax':20e6, 'points':1200, 'mag_noise':(0.12,0.45), 'ph_noise':(1.2,4.5)}
        }

    def generate_sample(self, transformer_type, vendor, fault_type=None, severity=0.0):
        spec = self.vendors[vendor]
        freq = np.logspace(np.log10(spec['fmin']), np.log10(spec['fmax']), spec['points'])
        omega = 2 * np.pi * freq

        tcfg = self.transformers[transformer_type]
        R_seg = tcfg['R_base'] * np.random.normal(1.0, 0.04, size=tcfg['segments'])
        L_seg = tcfg['L_base'] * np.random.normal(1.0, 0.04, size=tcfg['segments'])
        C_seg = tcfg['C_base'] * np.random.normal(1.0, 0.08, size=tcfg['segments'])
        Rc = tcfg['Rc'] * np.random.normal(1.0, 0.04)
        Lm = tcfg['Lm'] * np.random.normal(1.0, 0.04)
        Cw = tcfg['C_base'] / 10 * np.random.normal(1.0, 0.10)
        Cwg = tcfg['C_base'] / 5 * np.random.normal(1.0, 0.10)

        Z_total = np.zeros_like(freq, dtype=complex)
        for i in range(tcfg['segments']):
            Z_seg = R_seg[i] + 1j * omega * L_seg[i] + 1 / (1j * omega * C_seg[i])
            Z_total += Z_seg
        Zm = 1 / (1 / Rc + 1 / (1j * omega * Lm))
        Z_total += Zm + 1 / (1j * omega * Cw) + 1 / (1j * omega * Cwg)

        mag_db = 20 * np.log10(np.abs(Z_total))
        phase_deg = np.angle(Z_total, deg=True)

        anomaly = 0
        if fault_type and fault_type in self.faults:
            anomaly = 1
            info = self.faults[fault_type]
            mask = (freq >= info['bands'][0]) & (freq <= info['bands'][1])
            if 'dL_factor' in info:
                L_seg *= (1 + info['dL_factor'] * severity)
            if 'dC_factor' in info:
                C_seg *= (1 + info['dC_factor'] * severity)
            if 'dRc_factor' in info:
                Rc *= (1 + info['dRc_factor'] * severity)
            mag_db[mask] += severity * 18 * np.sin(2 * np.pi * np.log10(freq[mask] / info['bands'][0]))
            phase_deg[mask] += severity * 9 * np.cos(2 * np.pi * np.log10(freq[mask] / info['bands'][1]))

        mag_noise = np.random.normal(0, np.linspace(spec['mag_noise'][0], spec['mag_noise'][1], spec['points']))
        ph_noise = np.random.normal(0, np.linspace(spec['ph_noise'][0], spec['ph_noise'][1], spec['points']))
        mag_db += mag_noise
        phase_deg += ph_noise

        metadata = self._gen_metadata(transformer_type)
        criticality = self._calc_criticality(metadata, severity)
        labels = {
            'anomaly': anomaly,
            'fault_type': fault_type or 'healthy',
            'severity': float(round(severity, 3)),
            'criticality': round(criticality, 3)
        }

        features = extract_features(freq, mag_db, phase_deg)
        return freq, mag_db, phase_deg, labels, metadata, features

    def generate_dataset(self, n_samples=10000):
        samples = []
        types = list(self.transformers)
        vendors = list(self.vendors)
        faults = list(self.faults)
        weights = {'healthy': 0.28, 'core_grounding': 0.20}
        for f in faults:
            weights[f] = 0.52 / 3 if f != 'core_grounding' else 0.20
        dist = [weights['healthy']] + [weights.get(f, 0) for f in faults]
        for i in range(n_samples):
            ttype = np.random.choice(types)
            vendor = np.random.choice(vendors)
            ft_choice = np.random.choice([None] + faults, p=dist)
            sev = np.random.uniform(0.04, 0.84) if ft_choice else 0.0
            freq, mag, ph, labels, meta, features = self.generate_sample(ttype, vendor, ft_choice, sev)
            rec = {
                'id': i + 1,
                'transformer_type': ttype,
                'vendor': vendor,
                **labels,
                **meta,
                **features,
                "mag_first": mag[0],
                "mag_peak": np.max(mag),
                "ph_first": ph[0]
            }
            samples.append({**rec, "frequency": freq, "magnitude_db": mag, "phase_deg": ph})
        return samples

    def _gen_metadata(self, transformer_type):
        kv_map = {'distribution_11kv': 11, 'power_132kv': 132, 'transmission_400kv': 400}
        voltage_kv = kv_map[transformer_type]
        mva = round(np.random.lognormal(np.log(voltage_kv / 2), 0.55), 1)
        age = int(np.random.choice(np.arange(1, 41), p=np.ones(40) / 40))
        op_env = np.random.choice(["urban", "rural", "industrial"])
        return {
            'voltage_kv': voltage_kv,
            'power_mva': mva,
            'age_years': age,
            "operational_env": op_env
        }

    def _calc_criticality(self, meta, severity):
        vc = {11: 0.2, 132: 0.7, 400: 0.9}[meta['voltage_kv']]
        pc = min(meta['power_mva'] / 500, 1.0)
        network = 0.6 * vc + 0.4 * pc
        age_factor = min(meta['age_years'] / 30, 1.0)
        size_factor = min(meta['power_mva'] / 300, 1.0)
        rep = 0.7 * age_factor + 0.3 * size_factor
        op_factor = 1.0 if meta["operational_env"] == "industrial" else (0.8 if meta["operational_env"] == "urban" else 0.6)
        load_base = 0.7 + 0.25 * np.random.beta(2, 2)
        impact = min(load_base * op_factor, 1.0)
        crit = 0.38 * network + 0.33 * rep + 0.29 * impact
        return crit
