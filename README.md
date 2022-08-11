# Patent-figure-concept-captioning
- This work describes how to use basic image captioning concepts to generate concept captions for design patents. 
- The neural network architecture consists of the encoder and decoder block.
  - The encoder consists of a ResNet-152 pretrained network used to extract image features from the patent figures
  - The decoder consists of a Long Short Term Memory (LSTM) layers to capture long-range dependencies, and generate captions for the patent figures.
