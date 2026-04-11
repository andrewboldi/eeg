"""
Download MOABB datasets for pretraining - maximize SUBJECT COUNT within 18GB.

Strategy:
- Download ALL subjects of each dataset at once (more efficient)
- Start with BNCI datasets (fast EU servers, small per-subject)
- Then P300/BrainInvaders (fast figshare hosting)
- Then medium datasets
- Skip datasets >200MB per subject
- Check disk after each complete dataset
"""
import moabb.datasets as ds
import json
import os
import shutil
import time
import warnings
import traceback
import subprocess
warnings.filterwarnings('ignore')

STATUS_FILE = '/home/andrew/eeg/docs/research/moabb_download_status.json'
SUMMARY_FILE = '/home/andrew/eeg/docs/research/moabb_download_status.md'
MNE_DATA_DIR = os.path.expanduser('~/mne_data')
MAX_DISK_GB = 18.0

# Ordered by download efficiency and hosting speed
# Format: (name, class, expected_subjects)
DOWNLOAD_PLAN = [
    # === BNCI datasets (fast EU server, ~10-50MB/subject) ===
    ('BNCI2014_001', ds.BNCI2014_001, 9),
    ('BNCI2014_002', ds.BNCI2014_002, 14),
    ('BNCI2014_004', ds.BNCI2014_004, 9),
    ('BNCI2014_008', ds.BNCI2014_008, 8),
    ('BNCI2014_009', ds.BNCI2014_009, 10),
    ('BNCI2015_001', ds.BNCI2015_001, 12),
    ('BNCI2015_003', ds.BNCI2015_003, 10),
    ('BNCI2015_004', ds.BNCI2015_004, 9),
    ('BNCI2015_006', ds.BNCI2015_006, 11),
    ('BNCI2015_007', ds.BNCI2015_007, 16),
    ('BNCI2015_008', ds.BNCI2015_008, 13),
    ('BNCI2015_009', ds.BNCI2015_009, 21),
    ('BNCI2015_010', ds.BNCI2015_010, 12),
    ('BNCI2015_012', ds.BNCI2015_012, 10),
    ('BNCI2015_013', ds.BNCI2015_013, 6),
    ('BNCI2016_002', ds.BNCI2016_002, 15),
    ('BNCI2019_001', ds.BNCI2019_001, 10),
    ('BNCI2020_001', ds.BNCI2020_001, 45),
    ('BNCI2020_002', ds.BNCI2020_002, 18),
    ('BNCI2022_001', ds.BNCI2022_001, 13),
    ('BNCI2024_001', ds.BNCI2024_001, 20),
    ('BNCI2025_001', ds.BNCI2025_001, 20),
    ('BNCI2025_002', ds.BNCI2025_002, 10),
    ('BNCI2003_004', ds.BNCI2003_004, 5),

    # === PhysionetMI (slow server but tiny files, 109 subjects) ===
    ('PhysionetMI', ds.PhysionetMI, 109),

    # === GigaDB / Korean datasets (fast) ===
    ('Cho2017', ds.Cho2017, 52),

    # === BrainInvaders (figshare, usually fast) ===
    ('BI2014a', ds.BI2014a, 64),
    ('BI2014b', ds.BI2014b, 38),
    ('BI2015a', ds.BI2015a, 43),
    ('BI2015b', ds.BI2015b, 44),
    ('BI2012',  ds.BI2012, 25),
    ('BI2013a', ds.BI2013a, 24),

    # === Other MI datasets ===
    ('AlexMI', ds.AlexMI, 8),
    ('GrosseWentrup2009', ds.GrosseWentrup2009, 10),
    ('Weibo2014', ds.Weibo2014, 10),
    ('Zhou2016', ds.Zhou2016, 4),
    ('Zhou2020', ds.Zhou2020, 20),
    ('Shin2017A', ds.Shin2017A, 29),
    ('Shin2017B', ds.Shin2017B, 29),
    ('Ofner2017', ds.Ofner2017, 15),
    ('Schirrmeister2017', ds.Schirrmeister2017, 14),
    ('Jeong2020', ds.Jeong2020, 25),
    ('Ma2020', ds.Ma2020, 25),

    # === SSVEP datasets ===
    ('Nakanishi2015', ds.Nakanishi2015, 9),
    ('Wang2016', ds.Wang2016, 34),
    ('MAMEM1', ds.MAMEM1, 11),
    ('MAMEM2', ds.MAMEM2, 11),
    ('MAMEM3', ds.MAMEM3, 11),
    ('Kalunga2016', ds.Kalunga2016, 12),

    # === ERP datasets ===
    ('EPFLP300', ds.EPFLP300, 8),
    ('Huebner2017', ds.Huebner2017, 13),
    ('Huebner2018', ds.Huebner2018, 12),
    ('Sosulski2019', ds.Sosulski2019, 13),

    # === Misc ===
    ('Brandl2020', ds.Brandl2020, 16),
    ('Rodrigues2017', ds.Rodrigues2017, 19),
    ('Wairagkar2018', ds.Wairagkar2018, 14),
    ('Tavakolan2017', ds.Tavakolan2017, 12),
    ('Wu2020', ds.Wu2020, 6),
    ('Zhang2017', ds.Zhang2017, 12),
    ('Rozado2015', ds.Rozado2015, 30),
    ('Kaya2018', ds.Kaya2018, 7),
    ('Speier2017', ds.Speier2017, 10),
]


def get_disk_usage_gb():
    result = subprocess.run(['du', '-sb', MNE_DATA_DIR], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.split()[0]) / (1024**3)
    return 0.0


def get_free_gb():
    return shutil.disk_usage('/home/andrew').free / (1024**3)


def download_dataset(name, cls, status):
    """Download all subjects. Returns updated meta dict."""
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")

    dataset = cls()
    subjects = dataset.subject_list
    n_total = len(subjects)

    # Check what's already done
    existing = status.get(name, {})
    already_done = existing.get('subjects_downloaded', [])
    remaining = [s for s in subjects if s not in already_done]

    if not remaining:
        print(f"  Already complete ({len(already_done)}/{n_total})")
        existing['complete'] = True
        return existing

    print(f"  {len(remaining)} remaining of {n_total} subjects")

    # Initialize meta
    meta = existing if existing else {
        'name': name,
        'n_subjects_total': n_total,
        'subjects_downloaded': list(already_done),
        'complete': False,
        'n_channels': None,
        'sfreq': None,
        'paradigm': getattr(dataset, 'paradigm', '?'),
        'duration_per_subject_s': None,
        'mb_per_subject': None,
    }

    disk_before = get_disk_usage_gb()

    for i, subj in enumerate(remaining):
        # Disk check every 10 subjects
        if i > 0 and i % 10 == 0:
            used = get_disk_usage_gb()
            free = get_free_gb()
            print(f"    [disk: {used:.1f}GB used, {free:.1f}GB free]")
            if used > MAX_DISK_GB or free < 8.0:
                print(f"  STOPPING: disk limit reached")
                break

        try:
            t0 = time.time()
            print(f"  [{i+1}/{len(remaining)}] s{subj}...", end='', flush=True)
            data = dataset.get_data(subjects=[subj])

            # Get metadata from first downloaded subject
            if meta['n_channels'] is None:
                for s_id, sessions in data.items():
                    for sess_id, runs in sessions.items():
                        for run_id, raw in runs.items():
                            eeg_ch = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg']
                            meta['n_channels'] = len(eeg_ch)
                            meta['sfreq'] = raw.info['sfreq']
                            meta['duration_per_subject_s'] = raw.times[-1]
                            break
                        break
                    break

            dt = time.time() - t0
            meta['subjects_downloaded'].append(subj)
            print(f" OK ({dt:.0f}s)", flush=True)

            # After first subject, estimate per-subject size
            if i == 0:
                disk_after_first = get_disk_usage_gb()
                mb_first = (disk_after_first - disk_before) * 1024
                meta['mb_per_subject'] = mb_first
                est_total_gb = mb_first * len(remaining) / 1024
                print(f"    ~{mb_first:.0f}MB/subj, est total: {est_total_gb:.1f}GB for {len(remaining)} subjects")
                if mb_first > 300:
                    # Cap downloads for large datasets
                    budget_gb = MAX_DISK_GB - disk_after_first
                    max_subj = max(3, int(budget_gb * 1024 / mb_first / 2))
                    if max_subj < len(remaining):
                        print(f"    LARGE dataset: capping at {max_subj} subjects (budget: {budget_gb:.1f}GB)")
                        remaining = remaining[:max_subj]

        except Exception as e:
            print(f" ERROR: {str(e)[:60]}")
            continue

    disk_after = get_disk_usage_gb()
    n_done = len(meta['subjects_downloaded'])
    meta['complete'] = n_done >= n_total
    meta['disk_gb_total'] = disk_after - disk_before
    print(f"  DONE: {n_done}/{n_total} subjects, +{(disk_after-disk_before)*1024:.0f}MB")
    return meta


def write_summary(status):
    lines = [
        "# MOABB Dataset Download Status",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"MNE data dir: `{MNE_DATA_DIR}`",
        f"Disk used: {get_disk_usage_gb():.2f} GB",
        f"Free space: {get_free_gb():.1f} GB",
        "",
        "## Datasets",
        "",
        "| Dataset | Subj | EEG Ch | Hz | Dur(s) | MB/subj | Paradigm | Status |",
        "|---------|------|--------|----|--------|---------|----------|--------|",
    ]
    total = 0
    for name, m in sorted(status.items(), key=lambda x: len(x[1].get('subjects_downloaded', [])), reverse=True):
        n = len(m.get('subjects_downloaded', []))
        nt = m.get('n_subjects_total', '?')
        ch = m.get('n_channels', '?')
        hz = m.get('sfreq', '?')
        if isinstance(hz, float) and hz == int(hz):
            hz = int(hz)
        dur = m.get('duration_per_subject_s', '?')
        if isinstance(dur, (int, float)):
            dur = f"{dur:.0f}"
        mb = m.get('mb_per_subject', '?')
        if isinstance(mb, (int, float)):
            mb = f"{mb:.0f}"
        p = m.get('paradigm', '?')
        st = 'DONE' if m.get('complete') else f'{n}/{nt}'
        total += n
        lines.append(f"| {name} | {n}/{nt} | {ch} | {hz} | {dur} | {mb} | {p} | {st} |")

    lines.extend(["", f"**Total subjects: {total}**", f"**Disk: {get_disk_usage_gb():.2f} GB**"])
    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(lines))


def main():
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            status = json.load(f)

    used = get_disk_usage_gb()
    free = get_free_gb()
    print(f"Starting: {used:.2f}GB used, {free:.1f}GB free, limit {MAX_DISK_GB}GB")
    print(f"Plan: {len(DOWNLOAD_PLAN)} datasets\n")

    for name, cls, _ in DOWNLOAD_PLAN:
        used = get_disk_usage_gb()
        free = get_free_gb()
        if used > MAX_DISK_GB:
            print(f"\n*** DISK LIMIT ({used:.1f}GB > {MAX_DISK_GB}GB) ***")
            break
        if free < 8.0:
            print(f"\n*** LOW SPACE ({free:.1f}GB free) ***")
            break

        try:
            meta = download_dataset(name, cls, status)
            status[name] = meta
            # Save after each dataset
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            write_summary(status)
        except Exception as e:
            print(f"  FATAL: {e}")
            traceback.print_exc()

    total = sum(len(v.get('subjects_downloaded', [])) for v in status.values())
    print(f"\n{'='*70}")
    print(f"SESSION COMPLETE: {total} subjects, {get_disk_usage_gb():.2f}GB used")
    write_summary(status)


if __name__ == '__main__':
    main()
