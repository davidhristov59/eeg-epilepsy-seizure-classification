import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import argparse

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATAFOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_data")
OUTPUT_FILE = "subjects.csv"        # Default output
OUTPUT_PARQUET = "subjects.parquet" # Optional faster format
MAX_WORKERS = min(8, (os.cpu_count() or 1) + 4)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def get_memory_usage() -> float:
    """Return current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def process_single_file(file_info):
    """Read a single CSV file and ensure subject + recording_id columns."""
    files_path, csv_file = file_info
    file_path = os.path.join(files_path, csv_file)

    try:
        data = pd.read_csv(
            file_path,
            low_memory=False,
            engine="c",  # Fast C parser
        )

        # Derive subject from filename if not present
        if "subject" not in data.columns:
            parts = os.path.splitext(csv_file)[0].split("_")
            data["subject"] = parts[0] if parts else "unknown"

        if "recording_id" not in data.columns:
            data["recording_id"] = os.path.splitext(csv_file)[0]

        return data, None, file_path
    except Exception as e:
        return None, f"Error processing {file_path}: {str(e)}", file_path

# ---------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------
def merge_all_subjects(datafolder: str = DATAFOLDER,
                       output_file: str = OUTPUT_FILE,
                       write_parquet: bool = True):

    print(f"Using up to {MAX_WORKERS} threads (detected {os.cpu_count()} cores)")
    print(f"Scanning {datafolder} ...")

    csv_files = [f for f in os.listdir(datafolder) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files\n")
    all_dataframes = []
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_file, (datafolder, f)): f for f in csv_files}

        with tqdm(total=len(csv_files), desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                try:
                    data, error, _ = future.result()
                    if error:
                        errors.append(error)
                    else:
                        all_dataframes.append(data)

                    memory_gb = get_memory_usage()
                    pbar.set_postfix_str(f"RAM: {memory_gb:.1f} GB")
                    pbar.update(1)

                except Exception as e:
                    errors.append(f"Unexpected error: {str(e)}")
                    pbar.update(1)

    print("\n✅ Processing complete!")
    print(f"Files processed: {len(all_dataframes)}")
    print(f"Current memory usage: {get_memory_usage():.1f} GB")

    if errors:
        print(f"\n⚠ {len(errors)} errors occurred:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors\n")

    if not all_dataframes:
        print("❌ No data found to combine!")
        return

    print("\nCombining datasets...")
    result = pd.concat(all_dataframes, ignore_index=True, copy=False)
    print(f"Combined shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

    print("\nSaving merged dataset...")
    output_path_csv = os.path.join(datafolder, output_file)
    result.to_csv(output_path_csv, index=False, chunksize=10000)
    print(f"✅ Saved CSV: {output_path_csv}")

    if write_parquet:
        output_path_parquet = os.path.join(datafolder, OUTPUT_PARQUET)
        result.to_parquet(output_path_parquet, index=False)
        print(f"✅ Saved Parquet: {output_path_parquet}")

    print("\n📊 Dataset summary:")
    if "subject" in result.columns:
        print("  Unique subjects:", result["subject"].nunique())
    if "recording_id" in result.columns:
        print("  Unique recordings:", result["recording_id"].nunique())
        print(result.groupby("subject")["recording_id"].nunique().head())

    print(f"\nFinal memory usage: {get_memory_usage():.1f} GB")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all recordings into a single CSV.")
    parser.add_argument("--datafolder", default=DATAFOLDER, help="Folder containing CSV files")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output CSV filename")
    parser.add_argument("--no-parquet", action="store_true", help="Disable writing Parquet version")
    args = parser.parse_args()

    merge_all_subjects(args.datafolder, args.output, write_parquet=not args.no_parquet)
