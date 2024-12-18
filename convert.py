import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def convert_to_wav(src_file):
    # dest_file = src_file.replace('/dev/aac', '/wav').replace('.m4a', '.wav')
    dest_file = src_file.replace('/aac', '/wav').replace('.m4a', '.wav')
    dest_dir = os.path.dirname(dest_file)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Construct the ffmpeg command for conversion
    command = f'ffmpeg -i "{src_file}" -loglevel panic -ar 16000 "{dest_file}"'
    
    # Execute the command
    subprocess.run(command, shell=True, check=True)

def find_m4a_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.m4a'):
                yield os.path.join(dirpath, filename)

def main():
    # m4a_files = list(find_m4a_files('/home/jovyan/datasets/VoxCeleb2/dev/aac'))
    # m4a_files = list(find_m4a_files('/home/jovyan/datasets/VoxCeleb2/aac'))
    m4a_files = list(find_m4a_files('/home/jovyan/datasets/voxceleb2_test/aac'))
    print(len(m4a_files))
    # Use all available processors
    with ProcessPoolExecutor() as executor:
        # Map the convert_to_wav function over all found M4A files
        list(tqdm(executor.map(convert_to_wav, m4a_files), total=len(m4a_files)))

if __name__ == '__main__':
    main()