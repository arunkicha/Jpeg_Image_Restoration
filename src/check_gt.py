from PIL import Image

# Replace with one of your dataset image paths
path = "C:/Vit/Fall semester 25-26/Project/SWE1010/Datasets/Primary Datasets/DIV2K_train_HR/0001.png"

img = Image.open(path).convert('RGB')
img.show()
