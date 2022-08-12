## Description
- This work describes how to use basic image captioning concepts to generate concept captions for design patents. 
- The neural network architecture consists of the encoder and decoder block.
  - The encoder consists of a ResNet-152 pretrained network used to extract image features from the patent figures
  - The decoder consists of a Long Short Term Memory (LSTM) layers to capture long-range dependencies, and generate captions for the patent figures.
  
## Steps
1. Clone this directory
```
git clone git@github.com:lamps-lab/Patent-figure-concept-captioning.git
cd Patent-figure-concept-captioning/
```
2. Download the design patent dataset using the link below
```
https://drive.google.com/drive/u/1/folders/1bzUWpix1MrZtCYx42wxx3QZrQfyNT0rC
```
3. Generate train and test data from downloaded design patents data (e.g. segmentation_2007.json). The resulting data is a json file with the images as keys and captions as values. The captions are lowercased. The outputs are train_descriptions and test_descriptions.
```
python data.py --json segmentation_2007.json --output_dir <path/to/save/training/and/test/data>
```
4. Resize training images for the ResNet-152 pretrained model. The resized images will be used for training
```
python resize.py --image_dir <path/to/training/images> --output_dir <path/to/save/resized/images>
```
