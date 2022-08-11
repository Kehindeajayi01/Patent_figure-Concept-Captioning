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
import nltk
import pyter
import warnings

warnings.filterwarnings("ignore")
# Device configuration
device = torch.cuda.set_device(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, help='path for trained decoder')
    parser.add_argument('--test_caption_path', type=str, help='path for test annotation json file')
    parser.add_argument('--caption_path', type=str, help='path for train annotation json file')
    parser.add_argument('--image_dir', type=str, help='path to the collection of all images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    return args


def load_image(test_image_path, transform=None):
    image = Image.open(test_image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def generateCaption(test_image_path):    
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
    image = load_image(test_image_path, transform)
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
    unwanted = {"<unk>", "<start>", "<end>"}
    pred_caption = [ele for ele in sampled_caption if ele not in unwanted]

    return pred_caption

def calculate_metrics():
    #actual, predicted = list(), list()
    meteor_score = 0.0
    bleu_score = 0.0
    nist_score = 0.0
    ter_score = 0.0
    count = 0   # count of list of predicted captions to compute mean score for meteor score
    # load the test captions
    args = get_args()
    image_dir = args.image_dir  # path to the collection of all images (training + test images)
    test_desc_path = args.test_caption_path
    test_descs = json.load(open(test_desc_path))
    for img_name, gt_cap in test_descs.items():
        test_image_path = os.path.join(image_dir, img_name)
        pred_caption = generateCaption(test_image_path)
        reference = gt_cap.lower().split()
        # compute the meteor score for each caption
        meteor_score += round(nltk.translate.meteor_score.single_meteor_score(reference, pred_caption), 4)
        bleu_score += round(nltk.translate.bleu_score.sentence_bleu(reference, pred_caption), 4)
        nist_score += round(nltk.translate.nist_score.sentence_nist([reference], pred_caption), 4)
        ter_score += round(pyter.ter(pred_caption, reference), 4)
        
        print(f"Average Bleu score: {bleu_score}")
        print(f"Average Meteor score: {meteor_score}")
        print(f"Average Nist score: {nist_score}")
        print(f"Average Ter score: {ter_score}")

        print(f"Ground truth: {reference}")
        print(f"Predicted: {pred_caption}")
        break
        #     count += 1
    # ave_meteor_score = meteor_score / count
    # ave_bleu_score = bleu_score / count 
    # ave_nist_score = nist_score / count
    # ave_ter_score = ter_score / count
    
    # print(f"Average Bleu score: {ave_bleu_score}")
    # print(f"Average Meteor score: {ave_meteor_score}")
    # print(f"Average Nist score: {ave_nist_score}")
    # print(f"Average Ter score: {ave_ter_score}")
    
if __name__ == '__main__':
    calculate_metrics()
