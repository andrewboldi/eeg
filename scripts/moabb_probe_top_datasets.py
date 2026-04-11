"""Probe top MOABB datasets: download 1 subject to get channel/srate info."""
import moabb.datasets as datasets
from moabb.datasets.utils import dataset_list
import warnings
warnings.filterwarnings('ignore')

# Top datasets by subject count (manually selected from listing)
TOP_DATASETS = [
    'PhysionetMI',       # 109 subj
    'Liu2022EldBETA',    # 100 subj
    'Dreyer2023',        # 87 subj
    'Liu2020BETA',       # 70 subj
    'BI2014a',           # 64 subj
    'Stieger2021',       # 62 subj
    'Dreyer2023A',       # 60 subj
    'Dong2023',          # 59 subj
    'Lee2019_MI',        # 54 subj
    'Lee2019_ERP',       # 54 subj
    'Lee2019_SSVEP',     # 54 subj
    'Cho2017',           # 52 subj
    'Yang2025',          # 51 subj
    'Liu2024',           # 50 subj
    'BNCI2020_001',      # 45 subj
    'BI2015b',           # 44 subj
    'BI2015a',           # 43 subj
    'ErpCoreERN',        # 40 subj (same subjects across ErpCore)
    'Kim2025BetaRange',  # 40 subj
    'BI2014b',           # 38 subj
    'HefmiIch2025',      # 37 subj
]

# Map short names to actual classes
name_to_cls = {}
for cls in dataset_list:
    name_to_cls[cls.__name__] = cls

# Probe each
results = []
for cls in dataset_list:
    ds = cls()
    n_subj = len(ds.subject_list)
    if n_subj < 20:
        continue
    name = cls.__name__
    paradigm = ds.paradigm if hasattr(ds, 'paradigm') else '?'
    code = ds.code if hasattr(ds, 'code') else name

    # Try to get metadata without downloading
    try:
        # Check if dataset has sampling rate info
        srate = getattr(ds, 'sfreq', None) or getattr(ds, 'sampling_rate', None)
        n_channels = getattr(ds, 'n_channels', None)

        results.append({
            'name': code,
            'cls_name': name,
            'n_subjects': n_subj,
            'paradigm': paradigm,
            'srate': srate,
            'n_channels': n_channels,
        })
    except Exception as e:
        results.append({
            'name': code,
            'cls_name': name,
            'n_subjects': n_subj,
            'paradigm': paradigm,
            'srate': None,
            'n_channels': None,
        })

# Sort and print
results.sort(key=lambda x: x['n_subjects'], reverse=True)
print(f"{'Dataset':<45} {'Subj':>5} {'Ch':>4} {'Hz':>6} {'Paradigm':<10}")
print("-" * 80)
for r in results:
    ch = str(r['n_channels']) if r['n_channels'] else '?'
    sr = str(r['srate']) if r['srate'] else '?'
    print(f"{r['name']:<45} {r['n_subjects']:>5} {ch:>4} {sr:>6} {r['paradigm']:<10}")

print(f"\nTotal datasets with 20+ subjects: {len(results)}")
print(f"Total subjects: {sum(r['n_subjects'] for r in results)}")
