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
2. Download the design patent dataset using the link below and extract the images. Each year folder contains the original compound images, segmented images, and json file containing metadata of the segmented patent figures. 
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
5. Train the concept captioning network
```
python train.py --decoder_path <path/to/save/decoder> --encoder_path <path/to/save/encoder> --image_dir <path/to/resized/images> --caption_path <path/to/saved/train/descriptions/in/step 3> --num_epochs <default/is/10> 
```
6. Generate captions for test images
```
python test.py --decoder_path <path/to/save/decoder> --encoder_path <path/to/save/encoder> --image_dir <path/to/downloaded/images> --caption_path <path/to/saved/train/descriptions/in/step 3> --test_caption_path <path/to/saved/test/descriptions/in/step 3> --test_index <index/of/image/to/generate/caption/for/from/test/images>
```
7. Evaluate Model on all test images. Outputs are METEOR, NIST, TER, ROUGE, and ACCURACY scores
```
python evaluate.py --decoder_path <path/to/save/decoder> --encoder_path <path/to/save/encoder> --image_dir <path/to/downloaded/images> --caption_path <path/to/saved/train/descriptions/in/step 3> --test_caption_path <path/to/saved/test/descriptions/in/step 3>
```
