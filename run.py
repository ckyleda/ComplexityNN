import argparse
from ComplexityNet.evaluate import eval_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate complexity scores & maps for a directory of images.')
    parser.add_argument('--model', type=str, help="Path to the model", default='model.pt')
    parser.add_argument('--directory', type=str, help="Directory containing images.")
    parser.add_argument('--output', type=str, help="The output directory.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default = 2)")

    args = parser.parse_args()
    MODEL = args.model
    DIR = args.directory
    OUT = args.output
    BATCH_SIZE = args.batch_size

    eval_directory(model_path=MODEL, directory=DIR, output_path=OUT, batch_size=BATCH_SIZE)
