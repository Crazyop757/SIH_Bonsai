# SCADA Data Generator with Real Data Reference
# Generates realistic SCADA data using real datasets as patterns

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class SCADADataGenerator:
    """Generate realistic SCADA data with real temperature patterns as reference"""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.real_baseline = self.load_real_temperature_data()
        
        # Transformer types and their characteristics
        self.transformer_types = {
            'distribution_11kv': {
                'voltage_kv': 11,
                'power_mva_range': (1, 25),
                'age_range': (5, 30),
                'base_temp_offset': 5
            },
            'power_132kv': {
                'voltage_kv': 132,
                'power_mva_range': (25, 100),
                'age_range': (8, 25),
                'base_temp_offset': 10
            },
            'transmission_400kv': {
                'voltage_kv': 400,
                'power_mva_range': (100, 500),
                'age_range': (10, 40),
                'base_temp_offset': 15
            }
        }
        
        # Fault types and their impacts
        self.fault_types = {
            'healthy': {'severity_range': (0.0, 0.05), 'probability': 0.65},
            'axial_displacement': {'severity_range': (0.1, 0.6), 'probability': 0.10},
            'radial_displacement': {'severity_range': (0.15, 0.7), 'probability': 0.08},
            'winding_deformation': {'severity_range': (0.2, 0.8), 'probability': 0.07},
            'core_ground': {'severity_range': (0.25, 0.9), 'probability': 0.05},
            'shorted_turns': {'severity_range': (0.3, 0.85), 'probability': 0.05}
        }
        
        print("✅ SCADA Data Generator initialized with real temperature patterns")
    
    def load_real_temperature_data(self):
        """Load real temperature dataset as baseline for realistic patterns"""
        try:
            print("📥 Loading ETT temperature dataset as baseline...")
            # ETT (Electricity Transforming Temperature) dataset
            url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
            ett_data = pd.read_csv(url)
            
            # Process temperature columns
            temp_cols = [col for col in ett_data.columns if 'T' in col and col != 'date']
            if temp_cols:
                ett_data['avg_temp'] = ett_data[temp_cols].mean(axis=1)
                print(f"✅ Loaded ETT dataset with {len(ett_data)} temperature records")
                print(f"   Temperature columns: {temp_cols}")
                return ett_data[['date', 'avg_temp'] + temp_cols[:3]]  # Keep first 3 temp cols
            else:
                print("⚠️  No temperature columns found in ETT dataset")
                return self.generate_synthetic_baseline()
                
        except Exception as e:
            print(f"⚠️  Could not load ETT dataset: {e}")
            print("📊 Generating synthetic temperature baseline...")
            return self.generate_synthetic_baseline()
    
    def generate_synthetic_baseline(self):
        """Generate synthetic temperature baseline if real data unavailable"""
        dates = pd.date_range(start='2023-01-01', periods=8760, freq='H')  # 1 year hourly
        
        # Seasonal temperature pattern
        day_of_year = dates.dayofyear
        seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
        
        # Daily temperature pattern
        hour_of_day = dates.hour
        daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak in afternoon
        
        # Add noise
        noise = np.random.normal(0, 2, len(dates))
        
        avg_temp = seasonal_temp + daily_temp + noise
        
        synthetic_baseline = pd.DataFrame({
            'date': dates,
            'avg_temp': avg_temp,
            'OT': avg_temp + np.random.normal(0, 1, len(dates)),  # Oil temp
            'HUFL': avg_temp - 5 + np.random.normal(0, 1.5, len(dates)),  # High util temp
            'HULL': avg_temp - 8 + np.random.normal(0, 1.2, len(dates))   # Low util temp
        })
        
        print(f"✅ Generated synthetic temperature baseline with {len(synthetic_baseline)} records")
        return synthetic_baseline
    
    def get_realistic_temperature(self, load_factor=0.7, age_years=15, fault_severity=0.0):
        """Get realistic temperature based on real data patterns"""
        if self.real_baseline is not None and len(self.real_baseline) > 0:
            # Sample from real temperature data
            sample_row = self.real_baseline.sample(1).iloc[0]
            base_temp = sample_row['avg_temp'] if 'avg_temp' in sample_row else 25
        else:
            # Fallback to synthetic
            base_temp = 25 + np.random.normal(0, 5)
        
        # Apply load factor (higher load = higher temperature)
        load_heating = load_factor * 20
        
        # Apply age factor (older transformers run hotter)
        age_heating = (age_years / 30) * 10
        
        # Apply fault heating (faults cause overheating)
        fault_heating = fault_severity * 25
        
        # Random variation
        random_variation = np.random.normal(0, 3)
        
        oil_temp_top = base_temp + load_heating + age_heating + fault_heating + random_variation
        oil_temp_bottom = oil_temp_top - np.random.uniform(2, 5)  # Bottom is cooler
        
        # Ensure realistic ranges
        oil_temp_top = np.clip(oil_temp_top, 20, 85)
        oil_temp_bottom = np.clip(oil_temp_bottom, 18, oil_temp_top - 1)
        
        return oil_temp_top, oil_temp_bottom
    
    def generate_realistic_dataset(self, n_samples=25000):
        """Generate realistic SCADA dataset"""
        print(f"🔄 Generating {n_samples} realistic SCADA samples...")
        
        samples = []
        
        for i in range(n_samples):
            if i % 5000 == 0:
                print(f"   Generated {i}/{n_samples} samples...")
            
            # Select transformer type
            transformer_type = np.random.choice(list(self.transformer_types.keys()))
            transformer_spec = self.transformer_types[transformer_type]
            
            # Generate transformer metadata
            voltage_kv = transformer_spec['voltage_kv']
            power_mva = np.random.uniform(*transformer_spec['power_mva_range'])
            age_years = int(np.random.uniform(*transformer_spec['age_range']))
            
            # Select fault type based on probability
            fault_probabilities = [self.fault_types[ft]['probability'] for ft in self.fault_types.keys()]
            fault_type = np.random.choice(list(self.fault_types.keys()), p=fault_probabilities)
            
            # Generate fault severity
            if fault_type == 'healthy':
                severity = 0.0
                anomaly = 0
            else:
                severity = np.random.uniform(*self.fault_types[fault_type]['severity_range'])
                anomaly = 1
            
            # Generate load factor (realistic daily/seasonal patterns)
            load_factor = np.random.beta(2, 2) * 0.8 + 0.2  # Beta distribution between 0.2-1.0
            
            # Calculate electrical parameters
            base_voltage = voltage_kv * 1000
            base_current = power_mva * 1000 / (base_voltage * np.sqrt(3))
            
            # Add variations due to load and faults
            voltage_variation = np.random.normal(0, 0.01) - severity * 0.02  # Faults reduce voltage
            current_variation = np.random.normal(0, 0.05) + severity * 0.1   # Faults increase current
            
            voltage_l1 = base_voltage * (1 + voltage_variation)
            voltage_l2 = base_voltage * (1 + np.random.normal(0, 0.01))
            current_l1 = base_current * load_factor * (1 + current_variation)
            current_l2 = base_current * load_factor * (1 + np.random.normal(0, 0.03))
            
            # THD increases with age and faults
            thd_voltage = 1.0 + age_years * 0.05 + severity * 3.0 + np.random.normal(0, 0.2)
            thd_current = 0.8 + age_years * 0.03 + severity * 2.0 + np.random.normal(0, 0.15)
            
            # Get realistic temperature using real data patterns
            oil_temp_top, oil_temp_bottom = self.get_realistic_temperature(load_factor, age_years, severity)
            
            # Gas analysis (key fault indicators)
            base_h2 = 15 + age_years * 1.5
            base_ch4 = 10 + age_years * 1.0
            
            # Faults dramatically increase gas concentrations
            fault_gas_multiplier = 1 + severity * 4
            h2_gas_ppm = base_h2 * fault_gas_multiplier + np.random.normal(0, 3)
            ch4_gas_ppm = base_ch4 * fault_gas_multiplier + np.random.normal(0, 2)
            
            # Ensure non-negative values
            h2_gas_ppm = max(0, h2_gas_ppm)
            ch4_gas_ppm = max(1, ch4_gas_ppm)  # Prevent division by zero
            
            # Calculate criticality score
            criticality = min(1.0, (
                severity * 0.4 +
                max(0, (oil_temp_top - 60) / 30) * 0.25 +  # Temperature factor
                max(0, (h2_gas_ppm - 30) / 70) * 0.2 +     # Gas factor
                (age_years / 40) * 0.1 +                    # Age factor
                max(0, (thd_voltage - 2) / 8) * 0.05        # THD factor
            ))
            
            # Generate transformer ID
            transformer_id = f"T{voltage_kv}_{np.random.randint(1, 999):03d}"
            
            # Create SCADA record
            sample = {
                'timestamp': pd.Timestamp.now() - timedelta(minutes=np.random.randint(0, 43200)),  # Random time in last 30 days
                'transformer_id': transformer_id,
                'voltage_l1': round(voltage_l1, 1),
                'voltage_l2': round(voltage_l2, 1),
                'current_l1': round(current_l1, 2),
                'current_l2': round(current_l2, 2),
                'frequency': round(50.0 + np.random.normal(0, 0.05), 3),
                'thd_voltage': round(max(0, thd_voltage), 2),
                'thd_current': round(max(0, thd_current), 2),
                'oil_temp_top': round(oil_temp_top, 1),
                'oil_temp_bottom': round(oil_temp_bottom, 1),
                'h2_gas_ppm': round(h2_gas_ppm, 1),
                'ch4_gas_ppm': round(ch4_gas_ppm, 1),
                'voltage_kv': voltage_kv,
                'power_mva': round(power_mva, 1),
                'age_years': age_years,
                # Labels for training
                'fault_type': fault_type,
                'severity': round(severity, 3),
                'anomaly': anomaly,
                'criticality': round(criticality, 3),
                # Additional metadata
                'transformer_type': transformer_type,
                'load_factor': round(load_factor, 3)
            }
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        print(f"✅ Generated {len(df)} SCADA samples")
        print(f"   Fault distribution:")
        for fault, count in df['fault_type'].value_counts().items():
            percentage = count / len(df) * 100
            print(f"     {fault}: {count} ({percentage:.1f}%)")
        print(f"   Anomaly rate: {df['anomaly'].mean() * 100:.1f}%")
        print(f"   Average criticality: {df['criticality'].mean():.3f}")
        
        return df
    
    def save_dataset(self, df, filename=None):
        """Save dataset to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'scada_realistic_dataset_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        print(f"✅ Dataset saved to {filename}")
        return filename
    
    def get_dataset_summary(self, df):
        """Get comprehensive dataset summary"""
        summary = {
            'total_samples': len(df),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'transformer_distribution': df['transformer_type'].value_counts().to_dict(),
            'fault_distribution': df['fault_type'].value_counts().to_dict(),
            'anomaly_rate': df['anomaly'].mean(),
            'avg_criticality': df['criticality'].mean(),
            'temperature_stats': {
                'oil_temp_top': {
                    'mean': df['oil_temp_top'].mean(),
                    'std': df['oil_temp_top'].std(),
                    'min': df['oil_temp_top'].min(),
                    'max': df['oil_temp_top'].max()
                }
            },
            'gas_stats': {
                'h2_gas_ppm': {
                    'mean': df['h2_gas_ppm'].mean(),
                    'std': df['h2_gas_ppm'].std(),
                    'max': df['h2_gas_ppm'].max()
                }
            }
        }
        
        return summary

def generate_scada_training_data():
    """Main function to generate SCADA training dataset"""
    print("=" * 60)
    print("🏭 SCADA TRAINING DATA GENERATION")
    print("=" * 60)
    
    generator = SCADADataGenerator(random_seed=42)
    
    # Generate dataset
    df = generator.generate_realistic_dataset(n_samples=25000)
    
    # Save dataset
    filename = generator.save_dataset(df)
    
    # Get summary
    summary = generator.get_dataset_summary(df)
    
    # Save summary
    summary_file = filename.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total samples: {summary['total_samples']:,}")
    print(f"   Anomaly rate: {summary['anomaly_rate']*100:.1f}%")
    print(f"   Avg criticality: {summary['avg_criticality']:.3f}")
    print(f"   Temperature range: {summary['temperature_stats']['oil_temp_top']['min']:.1f}°C - {summary['temperature_stats']['oil_temp_top']['max']:.1f}°C")
    print(f"   Max H2 gas: {summary['gas_stats']['h2_gas_ppm']['max']:.1f} ppm")
    
    print(f"\n📁 Files created:")
    print(f"   - {filename}")
    print(f"   - {summary_file}")
    
    return df, filename, summary

if __name__ == "__main__":
    df, filename, summary = generate_scada_training_data()