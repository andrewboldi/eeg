"""List all MOABB datasets with subject count, channels, and sample rate."""
from moabb.datasets.utils import dataset_list

print(f"Found {len(dataset_list)} dataset classes\n")
print(f"{'Dataset':<45} {'Subjects':>8} {'Paradigm':<15}")
print("-" * 75)

results = []
for cls in dataset_list:
    try:
        ds = cls()
        n_subj = len(ds.subject_list)
        paradigm = ds.paradigm if hasattr(ds, 'paradigm') else 'unknown'
        name = ds.code if hasattr(ds, 'code') else cls.__name__
        results.append({
            'name': name,
            'cls_name': cls.__name__,
            'n_subjects': n_subj,
            'paradigm': paradigm,
        })
        print(f"{name:<45} {n_subj:>8} {paradigm:<15}")
    except Exception as e:
        print(f"{cls.__name__:<45} ERROR: {str(e)[:30]}")

print(f"\n{'='*75}")
print(f"Total datasets: {len(results)}")
print(f"Total unique subjects (upper bound): {sum(r['n_subjects'] for r in results)}")

# Sort by subject count
print(f"\n{'='*75}")
print(f"TOP 30 DATASETS BY SUBJECT COUNT:")
print(f"{'Dataset':<45} {'Subjects':>8} {'Paradigm':<15}")
print("-" * 75)
for r in sorted(results, key=lambda x: x['n_subjects'], reverse=True)[:30]:
    print(f"{r['name']:<45} {r['n_subjects']:>8} {r['paradigm']:<15}")
