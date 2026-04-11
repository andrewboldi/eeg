#!/bin/bash
# Download HBN-EEG RestingState.mat files from AWS S3
# Skips existing subjects, checks disk space, targets 180 new subjects

set -euo pipefail

DATA_DIR="/home/andrew/eeg/data/raw/hbn_eeg"
MIN_FREE_GB=5
TARGET_NEW=180
LOG_FILE="/home/andrew/eeg/data/raw/hbn_eeg/download_log.txt"

# Initialize counters
downloaded=0
skipped_existing=0
skipped_nodata=0
failed=0

echo "$(date): Starting HBN-EEG batch download" | tee "$LOG_FILE"
echo "Target: $TARGET_NEW new subjects" | tee -a "$LOG_FILE"

# Get list of existing subjects
existing=$(ls "$DATA_DIR" | grep '^NDAR' | sort)
echo "Existing subjects: $(echo "$existing" | wc -l)" | tee -a "$LOG_FILE"

# Get all S3 subjects
echo "Listing S3 subjects..."
all_subjects=$(aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/HBN/EEG/ 2>/dev/null | grep 'PRE NDAR' | awk '{print $2}' | tr -d '/')

echo "Total S3 subjects: $(echo "$all_subjects" | wc -l)" | tee -a "$LOG_FILE"

# Filter out existing
new_subjects=""
for subj in $all_subjects; do
    if ! echo "$existing" | grep -q "^${subj}$"; then
        new_subjects="$new_subjects $subj"
    fi
done
new_count=$(echo $new_subjects | wc -w)
echo "New subjects available: $new_count" | tee -a "$LOG_FILE"

check_disk() {
    local free_gb=$(df --output=avail /home/andrew/eeg/ | tail -1 | awk '{printf "%.1f", $1/1048576}')
    echo "$free_gb"
}

for subj in $new_subjects; do
    # Check if we hit target
    if [ "$downloaded" -ge "$TARGET_NEW" ]; then
        echo "Reached target of $TARGET_NEW downloads" | tee -a "$LOG_FILE"
        break
    fi

    # Check disk space every 10 downloads
    if [ $((downloaded % 10)) -eq 0 ]; then
        free_gb=$(check_disk)
        echo "  [Disk: ${free_gb}GB free, downloaded: $downloaded, skipped: $skipped_nodata, failed: $failed]" | tee -a "$LOG_FILE"
        # Compare as integers (multiply by 10 to handle decimals)
        free_int=$(echo "$free_gb" | awk '{printf "%d", $1*10}')
        min_int=$((MIN_FREE_GB * 10))
        if [ "$free_int" -lt "$min_int" ]; then
            echo "STOPPING: Only ${free_gb}GB free (minimum: ${MIN_FREE_GB}GB)" | tee -a "$LOG_FILE"
            break
        fi
    fi

    # Try to download
    s3_path="s3://fcp-indi/data/Projects/HBN/EEG/${subj}/EEG/preprocessed/mat_format/RestingState.mat"
    local_dir="${DATA_DIR}/${subj}"
    local_path="${local_dir}/RestingState.mat"

    # Check if file exists on S3 first (fast ls check)
    if ! aws s3 ls --no-sign-request "$s3_path" &>/dev/null; then
        skipped_nodata=$((skipped_nodata + 1))
        continue
    fi

    mkdir -p "$local_dir"
    if aws s3 cp --no-sign-request "$s3_path" "$local_path" --quiet 2>/dev/null; then
        downloaded=$((downloaded + 1))
        if [ $((downloaded % 5)) -eq 0 ]; then
            echo "  Downloaded $downloaded: $subj" | tee -a "$LOG_FILE"
        fi
    else
        failed=$((failed + 1))
        # Clean up empty directory
        rmdir "$local_dir" 2>/dev/null || true
    fi
done

# Final summary
free_gb=$(check_disk)
total_subjects=$(ls "$DATA_DIR" | grep '^NDAR' | wc -l)

echo "" | tee -a "$LOG_FILE"
echo "=== DOWNLOAD COMPLETE ===" | tee -a "$LOG_FILE"
echo "Downloaded: $downloaded" | tee -a "$LOG_FILE"
echo "Skipped (existing): $skipped_existing" | tee -a "$LOG_FILE"
echo "Skipped (no RestingState): $skipped_nodata" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "Total subjects now: $total_subjects" | tee -a "$LOG_FILE"
echo "Disk free: ${free_gb}GB" | tee -a "$LOG_FILE"
echo "$(date): Done" | tee -a "$LOG_FILE"
