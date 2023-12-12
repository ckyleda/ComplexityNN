import os
import torch
import numpy as np
import csv
from tqdm import tqdm
from ComplexityNet.carchitecture import ComplexityNet
from pathlib import Path
from ComplexityNet.dataset import TestDataset
from torch.utils.data import DataLoader
from PIL import Image


def eval_directory(model_path, directory, output_path, batch_size=2):
    model = ComplexityNet()
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Could not read model file. Does it exist?")

    if os.path.isdir(output_path):
        print("Output directory already exists. Files may be overwritten.")

    model.load_state_dict(torch.load(model_path))
    model = model.float()
    score_dict = {}

    directory = Path(directory)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = TestDataset(directory)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    for idx, data in tqdm(enumerate(test_dl), total=len(test_dl)):
        batch_images = data[0]
        filenames = data[1]
        predictions = model(batch_images)

        scores = predictions[1].cpu().detach().numpy().flatten()
        maps = predictions[0].cpu().detach().numpy()
        maps = np.transpose(maps, (0, 2, 3, 1))

        for idx, score in enumerate(scores):
            score_dict[filenames[idx]] = np.round(score, 2)

        for idx, map in enumerate(maps):
            out_map = map * 255.0
            out_map = out_map.astype(np.uint8)
            out_map = Image.fromarray(out_map)
            out_map.save(str(output_dir / filenames[idx]) + '.png')

    with open(str(output_dir / 'predictions.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Images', 'Scores'])
        for key, value in score_dict.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    eval_directory("../model/complexity_net.pt", "../samples", output_path='../output')
