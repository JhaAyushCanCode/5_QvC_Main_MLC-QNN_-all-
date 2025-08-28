import kagglehub

# Download latest version
path = kagglehub.dataset_download("debarshichanda/goemotions")

print("Path to dataset files:", path)
