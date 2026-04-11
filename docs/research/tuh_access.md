# TUH EEG Corpus -- Access Guide

Researched: 2026-04-11

## Overview

The TUH EEG Corpus (TUEG) is the largest publicly available EEG dataset, maintained by
the Neural Engineering Data Consortium (NEDC) at Temple University. It contains **26,846
clinical EEG recordings** from Temple University Hospital (TUH), collected 2002--2017.
The number often cited as "10,874 subjects" comes from earlier versions; v2.0.1 lists
26,846 recordings (many subjects have multiple sessions).

## 1. Registration URL and Process

**Registration form (PDF):**
<https://isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf>

**Downloads page:**
<https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml>

### Steps

1. Download the PDF form from the link above.
2. Fill it out using Adobe Acrobat (must be an editable PDF, not a photo/scan).
3. Required fields:
   - Legal name (correct capitalization)
   - Institutional affiliation (full name, no abbreviations)
   - Complete surface mail address recognized by postal service
   - Telephone number with country code
   - Institutional email address (not Gmail etc.)
   - Signature and date (electronic signatures accepted)
4. Email the completed form to **help@nedcdata.org** with subject line
   "Download The TUH EEG Corpus".
5. Once approved, you receive credentials and instructions for transmitting your
   SSH public key.
6. After SSH key setup, you can download via rsync.

### Usage Conditions

- Acknowledge the dataset in publications using the citation in the AAREADME file
- Do not redistribute the data (have third parties register separately)
- No re-identification of anonymized subjects
- No malicious use; research and technology development only
- Delete data from all systems when finished

## 2. Approval Timeline

**24--48 hours** after submitting a correctly filled form. Forms with errors
(wrong address format, abbreviations, missing fields) are returned and must be
resubmitted, which adds delay.

## 3. Programmatic Download (rsync + SSH keys)

As of December 2025, NEDC switched from password-based access to **SSH key
authentication**. After registration approval, you submit your SSH public key
and then use rsync.

### Test download

```bash
rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST .
```

### Download a corpus

```bash
# Example: download the TUH Abnormal EEG Corpus (TUAB)
rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1 .
```

Always use the `-L` flag (follow symlinks) -- all sub-corpora link back to TUEG.

For verbose debugging, change `-auvxL` to `-auvvvxL`.

There is **no REST API or Python download client**. rsync over SSH is the only
programmatic method. No `wget`/`curl` HTTP download is available.

### Alternative: Physical Disk

For locations with poor internet, NEDC will copy data to an **8TB USB drive** you
mail to them. You provide the drive and a FedEx/UPS account number for return
shipping. Mail to:

> Joseph Picone, 1610 Rhawn Street, Philadelphia, PA 19111

## 4. Total Size and Recommended Starting Subsets

### Estimated total size: ~4--8 TB

The full TUEG corpus requires an 8TB USB drive for physical distribution. The
EDF+ files are ~20 MB each; 26,846 recordings x 20 MB = ~537 GB for raw EDF
alone, but with annotations, metadata, and multiple versions the total is much
larger.

### Available Corpora (best to start small)

| Corpus | Code | rsync path | Description | Recommended? |
|--------|------|-----------|-------------|:---:|
| TUH EEG Corpus | TUEG | `data/tuh_eeg/tuh_eeg/v2.0.1` | Full 26,846 recordings | Too large to start |
| **TUH Abnormal EEG** | **TUAB** | `data/tuh_eeg/tuh_eeg_abnormal/v3.0.1` | Normal vs abnormal labels | **Yes -- best starter** |
| TUH EEG Artifact | TUAR | `data/tuh_eeg/tuh_eeg_artifact/v3.0.1` | 5 artifact types annotated | Good for preprocessing |
| TUH EEG Epilepsy | TUEP | `data/tuh_eeg/tuh_eeg_epilepsy/v3.0.0` | 100 epilepsy + 100 control subjects | Small, well-annotated |
| TUH EEG Events | TUEV | `data/tuh_eeg/tuh_eeg_events/v2.0.1` | 6 event classes (SPSW, GPED, etc.) | Good for event detection |
| TUH EEG Seizure | TUSZ | `data/tuh_eeg/tuh_eeg_seizure/v2.0.6` | Seizure annotations (start/stop/channel/type) | Most-cited subset |
| TUH EEG Slowing | TUSL | `data/tuh_eeg/tuh_eeg_slowing/v2.0.1` | Slowing event annotations | Niche |

### Recommendation for our project

For **pre-training a general EEG encoder** (scalp-to-in-ear transfer learning):

1. **Start with TUAB** (~2,993 recordings, ~30 GB estimated) -- fast to download,
   clean binary labels, widely benchmarked
2. **Then TUSZ** for temporal annotation richness
3. **Full TUEG** only if pre-training shows clear benefit on the smaller subsets

## 5. Mirrors and Alternative Access

- **No official mirrors exist.** The NEDC rsync server is the only source.
- **Not on OpenNeuro, HuggingFace, or PhysioNet.** The license prohibits
  redistribution, so no third-party mirrors are authorized.
- **Physical disk shipping** is the only alternative to rsync (see section 3).
- Some papers reference downloading subsets via the old `nedc_tuh_eeg` Python
  tools, but these are deprecated in favor of direct rsync.

## Key References

- Obeid & Picone, "The Temple University Hospital EEG Data Corpus,"
  Frontiers in Neuroscience, 2016.
  DOI: 10.3389/fnins.2016.00196
- Downloads page: <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>
- Contact: help@nedcdata.org
