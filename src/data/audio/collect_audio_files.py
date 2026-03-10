import os
import shutil

archive_dir = "raw_data/archive"
output_dir = "raw_data/ravdess_audio"

os.makedirs(output_dir, exist_ok=True)

total_files = 0

for actor_folder in os.listdir(archive_dir):
    actor_path = os.path.join(archive_dir, actor_folder)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                src = os.path.join(actor_path, file)
                dst = os.path.join(output_dir, file)

                shutil.copy(src, dst)
                total_files += 1

print(f"Copied {total_files} audio files into ravdess_audio/")