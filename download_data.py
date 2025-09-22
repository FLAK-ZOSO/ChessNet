import kagglehub

# Download latest version
path = kagglehub.dataset_download("s4lman/chess-pieces-dataset-85x85")

print("Path to dataset files:", path)
