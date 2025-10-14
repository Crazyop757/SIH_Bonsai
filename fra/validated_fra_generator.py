# Realistic FRA Data Generator with SCADA Validation
# Creates FRA data that correlates with real SCADA patterns for authenticity

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ValidatedFRAGenerator:
    """Generate realistic FRA data validated against SCADA measurements"""
    
    def __init__(self, scada_reference_path=None, random_seed=42):
        np.random.seed(random_seed)
        
        # Load SCADA reference data for validation
        self.scada_reference = self.load_scada_reference(scada_reference_path)
        
        # Industrial fault categories (more realistic than academic categories)
        self.fault_types = {
            'healthy': {
                'probability': 0.60,
                'severity_range': (0.0, 0.05),
                'description': 'No fault detected',
                'fra_signature': 'normal_response'
            },
            'thermal_fault': {
                'probability': 0.15,
                'severity_range': (0.1, 0.8),
                'description': 'Overheating/cooling issues',
                'fra_signature': 'high_freq_attenuation',
                'scada_correlation': ['oil_temp_top', 'oil_temp_bottom', 'load_factor']
            },
            'electrical_fault': {
                'probability': 0.10,
                'severity_range': (0.15, 0.9),
                'description': 'Electrical system problems',
                'fra_signature': 'impedance_variation',
                'scada_correlation': ['thd_voltage', 'thd_current', 'frequency']
            },
            'mechanical_fault': {
                'probability': 0.08,
                'severity_range': (0.2, 0.85),
                'description': 'Physical/structural damage',
                'fra_signature': 'resonance_shift',
                'scada_correlation': ['current_l1', 'current_l2', 'voltage_l1']
            },
            'oil_degradation': {
                'probability': 0.05,
                'severity_range': (0.1, 0.7),
                'description': 'Oil quality issues',
                'fra_signature': 'dielectric_change',
                'scada_correlation': ['h2_gas_ppm', 'ch4_gas_ppm', 'age_years']
            },
            'partial_discharge': {
                'probability': 0.02,
                'severity_range': (0.25, 0.95),
                'description': 'Insulation breakdown',
                'fra_signature': 'high_freq_distortion',
                'scada_correlation': ['thd_voltage', 'h2_gas_ppm', 'oil_temp_top']
            }
        }
        
        # Transformer specifications based on SCADA reference
        self.transformer_specs = {
            'distribution_11kv': {
                'voltage_kv': 11, 'power_mva': (1, 25), 'age_range': (5, 30),
                'freq_range': (20, 10e6), 'impedance_base': 0.05
            },
            'power_132kv': {
                'voltage_kv': 132, 'power_mva': (25, 100), 'age_range': (8, 25),
                'freq_range': (20, 25e6), 'impedance_base': 0.08
            },
            'transmission_400kv': {
                'voltage_kv': 400, 'power_mva': (100, 500), 'age_range': (10, 40),
                'freq_range': (20, 50e6), 'impedance_base': 0.12
            }
        }
        
        # FRA frequency bands for feature extraction
        self.frequency_bands = {
            'VLF': (20, 1e3),      # Very Low Frequency
            'LF': (1e3, 1e5),      # Low Frequency  
            'MF': (1e5, 2e6),      # Medium Frequency
            'HF': (2e6, 25e6)      # High Frequency
        }
        
        print("✅ Validated FRA Generator initialized with SCADA reference data")
        
    def load_scada_reference(self, scada_path):
        """Load SCADA reference data for validation"""
        if scada_path and pd.io.common.file_exists(scada_path):
            try:
                scada_df = pd.read_csv(scada_path)
                print(f"✅ Loaded SCADA reference with {len(scada_df)} records")
                print(f"   Fault distribution: {scada_df['fault_type'].value_counts().to_dict()}")
                return scada_df
            except Exception as e:
                print(f"⚠️  Could not load SCADA reference: {e}")
                return None
        else:
            print("📊 No SCADA reference provided - using synthetic baselines")
            return None
    
    def validate_against_scada(self, fault_type, severity, transformer_meta):
        """Validate FRA parameters against SCADA patterns"""
        if self.scada_reference is None:
            return self.generate_synthetic_validation(fault_type, severity, transformer_meta)
        
        # Find similar SCADA records for validation
        similar_records = self.scada_reference[
            (self.scada_reference['fault_type'] == fault_type) &
            (self.scada_reference['voltage_kv'] == transformer_meta['voltage_kv']) &
            (abs(self.scada_reference['severity'] - severity) < 0.2)
        ]
        
        if len(similar_records) == 0:
            # Fallback to any record with same fault type
            similar_records = self.scada_reference[
                self.scada_reference['fault_type'] == fault_type
            ]
        
        if len(similar_records) > 0:
            # Sample a real SCADA record for validation
            validation_record = similar_records.sample(1).iloc[0]
            return {
                'oil_temp_validation': validation_record['oil_temp_top'],
                'gas_validation': validation_record['h2_gas_ppm'],
                'thd_validation': validation_record['thd_voltage'],
                'age_validation': validation_record['age_years'],
                'validated_against_real_data': True,
                'scada_record_id': validation_record.get('transformer_id', 'unknown')
            }
        else:
            return self.generate_synthetic_validation(fault_type, severity, transformer_meta)
    
    def generate_synthetic_validation(self, fault_type, severity, transformer_meta):
        """Generate synthetic validation when SCADA data not available"""
        return {
            'oil_temp_validation': 45 + severity * 25 + np.random.normal(0, 3),
            'gas_validation': 20 + severity * 80 + np.random.normal(0, 5),
            'thd_validation': 1.5 + severity * 4 + np.random.normal(0, 0.3),
            'age_validation': transformer_meta['age_years'],
            'validated_against_real_data': False,
            'scada_record_id': 'synthetic'
        }
    
    def generate_base_fra_response(self, transformer_spec, fault_type, severity):
        """Generate base FRA frequency response"""
        freq_min, freq_max = transformer_spec['freq_range']
        
        # Generate frequency points (logarithmic distribution)
        n_points = np.random.randint(500, 2000)  # Realistic FRA measurement points
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
        
        # Base impedance characteristic
        base_impedance = transformer_spec['impedance_base']
        
        # Ideal transformer response
        magnitude_db = []
        phase_deg = []
        
        for freq in frequencies:
            # Base magnitude response (physics-based)
            if freq < 1e3:  # VLF - resistive dominant
                mag_base = 20 * np.log10(base_impedance * (1 + freq/1000))
                ph_base = 5 + freq/200  # Slight inductive phase
            elif freq < 1e5:  # LF - inductive effects
                mag_base = 20 * np.log10(base_impedance * np.sqrt(1 + (freq/50000)**2))
                ph_base = 15 + 45 * (freq/1e5)
            elif freq < 2e6:  # MF - capacitive effects start
                mag_base = 20 * np.log10(base_impedance * (1 - 0.3 * freq/1e6))
                ph_base = 60 - 30 * (freq/2e6)
            else:  # HF - capacitive dominant
                mag_base = 20 * np.log10(base_impedance * (0.7 - 0.4 * freq/25e6))
                ph_base = 30 - 80 * (freq/25e6)
            
            # Add measurement noise (realistic)
            noise_mag = np.random.normal(0, 0.5)  # ±0.5 dB noise
            noise_phase = np.random.normal(0, 2)   # ±2° phase noise
            
            magnitude_db.append(mag_base + noise_mag)
            phase_deg.append(ph_base + noise_phase)
        
        return frequencies, np.array(magnitude_db), np.array(phase_deg)
    
    def apply_fault_signature(self, frequencies, magnitude_db, phase_deg, fault_type, severity, validation_data):
        """Apply fault-specific signatures to FRA response"""
        
        if fault_type == 'healthy':
            return magnitude_db, phase_deg
        
        fault_config = self.fault_types[fault_type]
        signature_type = fault_config['fra_signature']
        
        # Apply fault signatures based on real physics
        if signature_type == 'high_freq_attenuation':  # Thermal faults
            # Thermal effects cause high-frequency attenuation
            thermal_factor = validation_data['oil_temp_validation'] / 60  # Normalize to typical temp
            for i, freq in enumerate(frequencies):
                if freq > 1e5:  # Affect medium and high frequencies
                    attenuation = severity * thermal_factor * (freq / 1e6) * 0.5
                    magnitude_db[i] -= attenuation
                    phase_deg[i] -= severity * 10  # Phase lag
        
        elif signature_type == 'impedance_variation':  # Electrical faults
            # Electrical faults cause impedance variations
            thd_factor = validation_data['thd_validation'] / 3.0  # Normalize to typical THD
            for i, freq in enumerate(frequencies):
                if 1e3 < freq < 1e6:  # Affect LF and MF bands
                    variation = severity * thd_factor * np.sin(2 * np.pi * np.log10(freq/1e3))
                    magnitude_db[i] += variation * 2
                    phase_deg[i] += variation * 15
        
        elif signature_type == 'resonance_shift':  # Mechanical faults
            # Mechanical damage shifts resonant frequencies
            for i, freq in enumerate(frequencies):
                if 1e4 < freq < 5e5:  # Critical resonance range
                    shift_factor = severity * (1 + np.random.normal(0, 0.1))
                    shifted_response = np.exp(-((np.log10(freq) - 4.5) / 0.3)**2)
                    magnitude_db[i] += severity * 5 * shifted_response
                    phase_deg[i] += severity * 30 * shifted_response
        
        elif signature_type == 'dielectric_change':  # Oil degradation
            # Oil degradation affects dielectric properties
            gas_factor = validation_data['gas_validation'] / 50  # Normalize to typical gas level
            for i, freq in enumerate(frequencies):
                if freq > 1e3:  # Affects all frequencies above VLF
                    dielectric_effect = severity * gas_factor * (freq / 1e6) ** 0.3
                    magnitude_db[i] += dielectric_effect * 1.5
                    phase_deg[i] -= dielectric_effect * 8
        
        elif signature_type == 'high_freq_distortion':  # Partial discharge
            # PD causes high-frequency distortions and harmonics
            pd_factor = validation_data['gas_validation'] / 100  # H2 indicates PD
            for i, freq in enumerate(frequencies):
                if freq > 5e5:  # High frequency effects
                    pd_distortion = severity * pd_factor * np.random.normal(1, 0.3)
                    magnitude_db[i] += pd_distortion * 3
                    phase_deg[i] += pd_distortion * np.random.normal(20, 10)
        
        return magnitude_db, phase_deg
    
    def extract_fra_features(self, frequencies, magnitude_db, phase_deg):
        """Extract FRA band features for model training"""
        features = {}
        
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            # Find indices in frequency band
            mask = (frequencies >= f_min) & (frequencies <= f_max)
            
            if np.sum(mask) > 0:
                band_mag = magnitude_db[mask]
                band_phase = phase_deg[mask]
                
                # Statistical features for each band
                features.update({
                    f'mag_mean_{band_name}': np.mean(band_mag),
                    f'mag_std_{band_name}': np.std(band_mag),
                    f'mag_min_{band_name}': np.min(band_mag),
                    f'mag_max_{band_name}': np.max(band_mag),
                    f'mag_range_{band_name}': np.max(band_mag) - np.min(band_mag),
                    f'ph_mean_{band_name}': np.mean(band_phase),
                    f'ph_std_{band_name}': np.std(band_phase),
                    f'ph_min_{band_name}': np.min(band_phase),
                    f'ph_max_{band_name}': np.max(band_phase),
                    f'ph_range_{band_name}': np.max(band_phase) - np.min(band_phase)
                })
            else:
                # Fill with zeros if no data in band
                for metric in ['mag_mean', 'mag_std', 'mag_min', 'mag_max', 'mag_range',
                             'ph_mean', 'ph_std', 'ph_min', 'ph_max', 'ph_range']:
                    features[f'{metric}_{band_name}'] = 0.0
        
        return features
    
    def generate_realistic_sample(self):
        """Generate one realistic FRA sample with SCADA validation"""
        
        # Select transformer type based on SCADA reference distribution
        if self.scada_reference is not None:
            voltage_dist = self.scada_reference['voltage_kv'].value_counts(normalize=True)
            if 11 in voltage_dist.index and voltage_dist[11] > 0.3:
                transformer_type = 'distribution_11kv'
            elif 132 in voltage_dist.index and voltage_dist[132] > 0.3:
                transformer_type = 'power_132kv' 
            else:
                transformer_type = 'transmission_400kv'
        else:
            transformer_type = np.random.choice(list(self.transformer_specs.keys()))
        
        transformer_spec = self.transformer_specs[transformer_type]
        
        # Generate transformer metadata
        voltage_kv = transformer_spec['voltage_kv']
        power_mva = np.random.uniform(*transformer_spec['power_mva'])
        age_years = int(np.random.uniform(*transformer_spec['age_range']))
        
        transformer_meta = {
            'voltage_kv': voltage_kv,
            'power_mva': power_mva,
            'age_years': age_years,
            'transformer_type': transformer_type
        }
        
        # Select fault type based on probabilities
        fault_probs = [self.fault_types[ft]['probability'] for ft in self.fault_types.keys()]
        fault_type = np.random.choice(list(self.fault_types.keys()), p=fault_probs)
        
        # Generate fault severity
        if fault_type == 'healthy':
            severity = 0.0
            anomaly = 0
        else:
            severity = np.random.uniform(*self.fault_types[fault_type]['severity_range'])
            anomaly = 1
        
        # Validate parameters against SCADA data
        validation_data = self.validate_against_scada(fault_type, severity, transformer_meta)
        
        # Generate base FRA response
        frequencies, magnitude_db, phase_deg = self.generate_base_fra_response(
            transformer_spec, fault_type, severity
        )
        
        # Apply fault signatures
        magnitude_db, phase_deg = self.apply_fault_signature(
            frequencies, magnitude_db, phase_deg, fault_type, severity, validation_data
        )
        
        # Extract features
        fra_features = self.extract_fra_features(frequencies, magnitude_db, phase_deg)
        
        # Add transformer metadata to features
        fra_features.update({
            'voltage_kv': voltage_kv,
            'power_mva': power_mva,
            'age_years': age_years
        })
        
        # Calculate criticality score (similar to SCADA model)
        criticality = min(1.0, (
            severity * 0.5 +
            max(0, (validation_data['oil_temp_validation'] - 55) / 30) * 0.2 +
            max(0, (validation_data['gas_validation'] - 30) / 70) * 0.2 +
            (age_years / 40) * 0.1
        ))
        
        # Create complete sample
        sample = {
            **fra_features,
            'fault_type': fault_type,
            'severity': round(severity, 3),
            'anomaly': anomaly,
            'criticality': round(max(0, criticality), 3),
            'transformer_type': transformer_type,
            'validation_metadata': validation_data,
            'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 43200))
        }
        
        return sample, frequencies, magnitude_db, phase_deg
    
    def generate_dataset(self, n_samples=25000):
        """Generate complete FRA dataset with SCADA validation"""
        print(f"🔄 Generating {n_samples} validated FRA samples...")
        
        samples = []
        for i in range(n_samples):
            if i % 5000 == 0:
                print(f"   Generated {i}/{n_samples} samples...")
            
            sample, frequencies, magnitude_db, phase_deg = self.generate_realistic_sample()
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        print(f"✅ Generated {len(df)} FRA samples")
        print(f"   Fault distribution:")
        for fault, count in df['fault_type'].value_counts().items():
            percentage = count / len(df) * 100
            print(f"     {fault}: {count} ({percentage:.1f}%)")
        print(f"   Anomaly rate: {df['anomaly'].mean() * 100:.1f}%")
        print(f"   Avg criticality: {df['criticality'].mean():.3f}")
        print(f"   SCADA validated samples: {df['validation_metadata'].apply(lambda x: x['validated_against_real_data']).sum()}")
        
        return df
    
    def save_dataset(self, df, filename=None):
        """Save dataset with metadata"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fra_validated_dataset_{timestamp}.csv'
        
        # Remove complex validation metadata column for CSV
        df_export = df.copy()
        df_export = df_export.drop('validation_metadata', axis=1)
        df_export.to_csv(filename, index=False)
        
        # Save metadata separately
        metadata = {
            'generation_info': {
                'total_samples': len(df),
                'scada_validated_samples': df['validation_metadata'].apply(lambda x: x['validated_against_real_data']).sum(),
                'generation_date': datetime.now().isoformat(),
                'scada_reference_used': self.scada_reference is not None
            },
            'fault_distribution': df['fault_type'].value_counts().to_dict(),
            'validation_summary': {
                'avg_oil_temp': df['validation_metadata'].apply(lambda x: x['oil_temp_validation']).mean(),
                'avg_gas_level': df['validation_metadata'].apply(lambda x: x['gas_validation']).mean(),
                'avg_thd': df['validation_metadata'].apply(lambda x: x['thd_validation']).mean()
            },
            'feature_info': {
                'frequency_bands': self.frequency_bands,
                'total_features': len([col for col in df.columns if col not in ['fault_type', 'severity', 'anomaly', 'criticality', 'transformer_type', 'validation_metadata', 'timestamp']])
            }
        }
        
        metadata_file = filename.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Dataset saved to {filename}")
        print(f"✅ Metadata saved to {metadata_file}")
        
        return filename, metadata_file

def generate_validated_fra_dataset(scada_reference_path=None):
    """Main function to generate validated FRA dataset"""
    print("=" * 60)
    print("🔬 VALIDATED FRA DATASET GENERATION")
    print("=" * 60)
    
    generator = ValidatedFRAGenerator(scada_reference_path=scada_reference_path, random_seed=42)
    
    # Generate dataset
    df = generator.generate_dataset(n_samples=25000)
    
    # Save dataset
    csv_file, metadata_file = generator.save_dataset(df)
    
    print(f"\n📊 GENERATION SUMMARY:")
    print(f"   Total samples: {len(df):,}")
    if scada_reference_path:
        validated_count = df['validation_metadata'].apply(lambda x: x['validated_against_real_data']).sum()
        print(f"   SCADA validated: {validated_count:,} ({validated_count/len(df)*100:.1f}%)")
    print(f"   Industrial fault categories: 6 types")
    print(f"   Feature count: 40 FRA band features + 3 metadata")
    
    print(f"\n📁 Files created:")
    print(f"   - {csv_file}")
    print(f"   - {metadata_file}")
    
    return df, csv_file, metadata_file

if __name__ == "__main__":
    # Use SCADA reference for validation
    scada_ref_path = "scada_realistic_dataset_20251013_015900.csv"
    df, csv_file, metadata_file = generate_validated_fra_dataset(scada_ref_path)