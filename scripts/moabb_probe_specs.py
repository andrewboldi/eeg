"""Download 1 subject from top datasets to get channel count and sample rate."""
import moabb.datasets as ds
import warnings
import traceback
warnings.filterwarnings('ignore')

# Top datasets by subject count - instantiate and download 1 subject
DATASETS = [
    (ds.PhysionetMI, 109),
    (ds.Cho2017, 52),
    (ds.Lee2019_MI, 54),
    (ds.BNCI2014_001, 9),  # well-known 22ch dataset
    (ds.BNCI2020_001, 45),
    (ds.Stieger2021, 62),
]

for cls, expected_n in DATASETS:
    name = cls.__name__
    print(f"\n{'='*60}")
    print(f"Probing {name} (expected {expected_n} subjects)...")
    try:
        dataset = cls()
        subj = dataset.subject_list[0]
        print(f"  Downloading subject {subj}...")
        data = dataset.get_data(subjects=[subj])
        # data structure: {subject: {session: {run: raw}}}
        for s_id, sessions in data.items():
            for sess_id, runs in sessions.items():
                for run_id, raw in runs.items():
                    print(f"  Subject {s_id}, Session {sess_id}, Run {run_id}")
                    print(f"  Channels: {len(raw.ch_names)} ({raw.ch_names[:5]}...)")
                    print(f"  Sample rate: {raw.info['sfreq']} Hz")
                    print(f"  Duration: {raw.times[-1]:.1f}s")
                    print(f"  Channel types: {set(raw.get_channel_types())}")
                    break
                break
            break
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
    print()
