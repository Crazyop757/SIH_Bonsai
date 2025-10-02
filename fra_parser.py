import pandas as pd
import xml.etree.ElementTree as ET
import os

def parse_csv(file_path, vendor="Unknown"):
    df = pd.read_csv(file_path)
    return {
        "frequency": df.iloc[:,0].tolist(),
        "magnitude": df.iloc[:,1].tolist(),
        "phase": df.iloc[:,2].tolist() if df.shape[1] > 2 else None,
        "metadata": {"vendor": vendor, "source_file": os.path.basename(file_path)}
    }

def parse_xml(file_path, vendor="Unknown"):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    frequency, magnitude, phase = [], [], []
    for point in root.findall(".//Point"):
        frequency.append(float(point.find("Frequency").text))
        magnitude.append(float(point.find("Magnitude").text))
        phase_val = point.find("Phase")
        phase.append(float(phase_val.text) if phase_val is not None else None)
    
    return {
        "frequency": frequency,
        "magnitude": magnitude,
        "phase": phase,
        "metadata": {"vendor": vendor, "source_file": os.path.basename(file_path)}
    }

def parse_txt(file_path, vendor="Unknown"):
    # assume space-separated: freq mag phase
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return {
        "frequency": df.iloc[:,0].tolist(),
        "magnitude": df.iloc[:,1].tolist(),
        "phase": df.iloc[:,2].tolist() if df.shape[1] > 2 else None,
        "metadata": {"vendor": vendor, "source_file": os.path.basename(file_path)}
    }

def parse_file(file_path, vendor="Unknown"):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        return parse_csv(file_path, vendor)
    elif ext == ".xml":
        return parse_xml(file_path, vendor)
    elif ext in [".txt", ".dat"]:
        return parse_txt(file_path, vendor)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
