# ChessNet

As this is meant for internal usage, no long description but rather only the commands needed to run this.

### Requirements

```bash
python3 -m pip venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Edit the values in `chessnet.yml` to your needs.

```yaml
epochs: 3 # Number of epochs of training for each run of the program
split: 0.7 # Relative number of training data over the total (70% here), the rest is for testing
batch: 20 # Size of a mini batch for training in number of samples
eta: 0.3 # Initial learning rate
decay: 0.1 # Decay of the learning rate (uses hyperbolic 1/(1+sum(decay)) function)

input-size: # Sizes of the input images
  x: 64
  y: 64
inner-layer-sizes: # Sizes of the intermediate layers, from left to right
  - 80
output-size: 10 # Size of the output layer (number of categories of data)
```

### Training + inference

```bash
python3 chessnet.py
```

### Generate graphs from data

```bash
cd stats
python3 plot-results.py
```

### Clean all

```bash
rm -rf testNet stats/*.csv stats/*.png stats/*.zip
```
