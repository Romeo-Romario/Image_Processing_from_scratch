import os


def check_folder_stats(base_directory):
    # Check if the directory actually exists first
    if not os.path.exists(base_directory):
        print(f"Error: Could not find {base_directory}")
        return

    print(f"{'Folder Name':<20} | {'File Count':<12} | {'Size (MB)':<10}")
    print("-" * 50)

    total_dataset_files = 0
    total_dataset_size = 0

    # Loop through everything inside the dataset folder
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)

        # We only care about directories (folders)
        if os.path.isdir(item_path):
            folder_size = 0
            file_count = 0

            # Walk through all files inside this specific letter's folder
            for dirpath, _, filenames in os.walk(item_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # Add up the file sizes
                    if not os.path.islink(fp):
                        folder_size += os.path.getsize(fp)
                        file_count += 1

            total_dataset_files += file_count
            total_dataset_size += folder_size

            # Convert bytes to Megabytes
            size_mb = folder_size / (1024 * 1024)

            # Print the row
            print(f"{item:<20} | {file_count:<12} | {size_mb:.2f} MB")

    print("-" * 50)
    total_mb = total_dataset_size / (1024 * 1024)
    print(f"TOTAL: {total_dataset_files} images across all folders ({total_mb:.2f} MB)")


if __name__ == "__main__":
    # Point this to your dataset folder
    dataset_path = r"D:\Source\Diplom\tryouts\tryout2_image_deskweing\implementation\machine_learning\dataset"
    check_folder_stats(dataset_path)
