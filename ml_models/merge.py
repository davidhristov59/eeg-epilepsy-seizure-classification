import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

DATAFOLDER = "processed_data"
OUTPUT_FILE = "subjects.csv"

# Auto-detect optimal thread count based on CPU cores and I/O capacity
MAX_WORKERS = min(8, (os.cpu_count() or 1) + 4)  # CPU cores + some I/O threads
print(f"Using {MAX_WORKERS} threads (detected {os.cpu_count()} CPU cores)")


def process_single_file(file_info):
    """Process a single CSV file with optimizations"""
    subfolder, files_path, subject_file = file_info
    file_path = os.path.join(files_path, subject_file)

    try:
        # Optimizations for pandas
        data = pd.read_csv(
            file_path,
            low_memory=False,  # Read entire file into memory for faster processing
            engine='c',  # Use C parser (faster than python parser)
        )
        data["subject"] = subfolder
        return data, None, file_path
    except Exception as e:
        return None, f"Error processing {file_path}: {str(e)}", file_path


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


print("Scanning directories...")
file_tasks = []
subject_folders = set()

for subfolder in os.listdir(DATAFOLDER):
    files_path = os.path.join(DATAFOLDER, subfolder)
    if not os.path.isdir(files_path):
        continue

    csv_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    if csv_files:
        subject_folders.add(subfolder)
        for subject_file in csv_files:
            file_tasks.append((subfolder, files_path, subject_file))

print(f"Found {len(subject_folders)} subjects with {len(file_tasks)} total files")

# Process files with multithreading
all_dataframes = []
errors = []
current_subject = ""

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_file = {executor.submit(process_single_file, task): task for task in file_tasks}

    # Process completed tasks with progress bar
    with tqdm(total=len(file_tasks), desc="Processing files", unit="file") as pbar:
        for future in as_completed(future_to_file):
            task = future_to_file[future]
            subfolder = task[0]

            try:
                data, error, file_path = future.result()
                if error:
                    errors.append(error)
                else:
                    all_dataframes.append(data)

                # Update progress bar with current subject and memory usage
                if subfolder != current_subject:
                    current_subject = subfolder

                memory_gb = get_memory_usage()
                pbar.set_postfix_str(f"{subfolder} | RAM: {memory_gb:.1f}GB")
                pbar.update(1)

            except Exception as e:
                errors.append(f"Unexpected error with {task}: {str(e)}")
                pbar.update(1)

print(f"\nProcessing complete!")
print(f"Total subjects: {len(subject_folders)}")
print(f"Total files processed: {len(all_dataframes)}")
print(f"Memory usage: {get_memory_usage():.1f}GB")

if errors:
    print(f"⚠  {len(errors)} errors occurred:")
    for error in errors[:3]:  # Show first 3 errors
        print(f"  - {error}")
    if len(errors) > 3:
        print(f"  ... and {len(errors) - 3} more errors")

if all_dataframes:
    print("\nCombining datasets...")
    # Use concat with specific parameters for better performance
    result = pd.concat(all_dataframes, ignore_index=True, copy=False)

    output_path = os.path.join(DATAFOLDER, OUTPUT_FILE)
    print("Saving to CSV...")

    # Optimize CSV writing
    result.to_csv(output_path, index=False, chunksize=10000)

    print(f"✅ Saved combined dataset: {output_path}")
    print(f"Final shape: {result.shape}")
    print(f"Final memory usage: {get_memory_usage():.1f}GB")
else:
    print("❌ No data found to combine!")