from ClutterSym.compute_rag_sym import clutter_sym_worker
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate complexity metrics for a directory of images.')
    parser.add_argument('--directory', type=str, help="Directory containing images.")
    parser.add_argument('--output', type=str, help="The output directory.")
    parser.add_argument('--processes', type=int, default=2, help="The number of processes to use.")

    args = parser.parse_args()

    IMAGE_DIR = Path(args.directory)
    OUTPUT_DIR = Path(args.output)
    PROCESSES = args.processes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(IMAGE_DIR.glob('*'))
    pool = mp.Pool(processes=PROCESSES)
    successes = tqdm(pool.imap(clutter_sym_worker, files), total=len(files))
    successes = list(successes)

    with open(OUTPUT_DIR / 'metrics.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['file', 'symmetry', 'clutter'])

        for entry in successes:
            writer.writerow([entry['filename'], entry['sym'], entry['clutter']])

    print("Finished.")
