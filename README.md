# DNN For Complexity Estimation of Scene Images

## Setup

1. Install Python 3 (3.8 recommended)
2. Install provided requirements either via pip (`pip install -r requirements.txt`) or Anaconda environment with required packages (see below)
3. Acquire the model weights and place them somewhere easily accessible (for example, `model/weights.pt`)
4. Set up your testing data. This should be a directory which contains JPG or PNG images.

The model weights are available [here](https://github.com/ckyleda/ComplexityNN/releases/download/1.0.0/complexity_net.pt).

### Anaconda / Required Packages

This project depends upon core packages:

- Torch
- Torchvision
- Numpy
- Pillow (PIL)
- tqdm

## Run

1. Either run `run.py` directly, or edit and execute the provided shell script `sh example_run.sh`
2. On first run weights for the ResNet backbone will be downloaded. 

### Quickstart

The general run command is: `python3 run.py --model path/to/model/weights.pt --directory input/image/directory --output output/directory/`

### Additional Details

Example images are provided under `samples/` and example outputs under `output/`. 
By running the model against`samples/' with the provided weights content matching that 
under the outputs directory should be generated.

You may also simply import the `eval_directory` function from `ComplexityNet.evaluate` and use as required.
See `run.py` for an example of this.

### Required flags for `run.py` execution:

`--model [path/to/model.pt]`: The path to the model weights, provided as a .pt file.

`--directory [path/to/input/images]`: Path to directory for which you wish to predict complexity data.
Must contain **only** .png or .jpg / .jpeg files.

`--output [path/to/output/]`: Directory to place output (generated) complexity maps and predictions.
If it does not exist, it will be created.

### Optional flags for `run.py`

`--batch_size [integer]`: Set batch size (default 2). As the network is provided pre-trained,
it is only strictly necessary to tweak this to minimise IO overhead.