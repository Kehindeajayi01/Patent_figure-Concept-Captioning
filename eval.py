import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocal import Vocabulary, build_vocab
from model import EncoderCNN, DecoderRNN
from PIL import Image
import json
import ntpath


# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.cuda.set_device(1)

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, help='path for trained decoder')
    parser.add_argument('--test_caption_path', type=str, help='path for test annotation json file')
    parser.add_argument('--caption_path', type=str, help='path for train annotation json file')
    parser.add_argument('--image_dir', type=str, help='path to the collection of all images')
    parser.add_argument('--test_index', type=int, help='index of image to generate caption for from the test images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    return args

args = get_args()
index = args.test_index   # index of image to generate caption for from the test images

def test_path():
    # load the test captions
    args = get_args()
    image_dir = args.image_dir  # path to the collection of all images (training + test images)
    test_desc_path = args.test_caption_path

    test_descs = json.load(open(test_desc_path))
    test_images = list(test_descs.keys())
    
    if index > len(test_images) - 1 or test_images[index] not in test_descs:
        return f"Image NOT Found in Test Images"
    # get the test image name from the test images and join with image path
    test_image_name = test_images[index]
    test_image_path = os.path.join(image_dir, test_image_name)
    # get the true image caption from the test descriptions
    gt_caption = test_descs[test_image_name]
    return test_image_path, gt_caption



def load_image(transform=None):
    test_image_path, _ = test_path()
    image = Image.open(test_image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def generateCaption():    
    args = get_args()
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    vocab = build_vocab(args.caption_path)  # train captions path

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    pred_caption = ' '.join(sampled_caption)

    test_image_path, gt_caption = test_path()
    image_name = ntpath.basename(test_image_path)    
    # Print out the image and the generated caption
    # image = Image.open(args.image)
    # plt.imshow(np.asarray(image))

    print("=================Generating Caption for a single test figure================")
    
    print(f"Figure name: {image_name}")
    print("\n")
    print("=======================True  Caption===========================")
    print(gt_caption.lower())
    print("\n")
    print("========================Generated Caption=======================")
    print(pred_caption)


if __name__ == '__main__':
    generateCaption()
