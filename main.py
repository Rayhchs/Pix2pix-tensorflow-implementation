"""
Created on Thu Sep 9 2021

main

@author: Ray
"""
from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
import os, sys
from pix2pix import *
from utils import *

parser = ArgumentParser(usage=None, formatter_class=RawTextHelpFormatter, description="Image translation using pix2pix: \n \n"
    "This code provides a pix2pix model from training to testing. "
    "Users can apply it for image translation tasks. \n"
    "------------------------------Parameters & data format------------------------------ \n"
    "txt format (train): <Path of image for generator>, <Path of image for discriminator> \n"
    "txt format (test): <Path of image to be translated>\n"
    "Epoch: default=400 \n"
    "Batch size: default=1 \n"
    "Save path: Path to save testing result, default='.\result' \n")

parser.add_argument("filename", help="txt file which includes image directions")
parser.add_argument("mode", help="train or test")
parser.add_argument("-e", "--epoch", type=int, default=400, dest="epoch")
parser.add_argument("-b", "--batch_size", type=int, default=1, dest="batch_size")
parser.add_argument("-s", "--save_path", type=str, default=None, dest="save_path")
args = parser.parse_args()

def main():
    
    image_path = args.filename if os.path.isfile(args.filename) else sys.exit("Incorrect file")
    
    if args.mode.lower() == 'train':
        with open(image_path) as f: 
            image_lists = f.read().splitlines()

        for i in image_lists:
            g_train, d_train = i.split(', ')
            sys.exit("Found wrong path or wrong format") if os.path.isfile(d_train) == False else None
            sys.exit("Found wrong path or wrong format") if os.path.isfile(g_train) == False else None

        print("All images are loaded")
                
        with tf.Session() as sess:
            model = pix2pix(sess, args.mode.lower())
            model.train(image_lists, args.epoch, args.batch_size)
            
    elif args.mode.lower() == 'test':
        with open(image_path) as f: 
            image_lists = f.read().splitlines()
            
        for i in image_lists:
            sys.exit("Found wrong path or wrong format") if os.path.isfile(i) == False else None
        print("All images are loaded")
                
        with tf.Session() as sess:
            model = pix2pix(sess, args.mode.lower())
            g_imgs, o_imgs = model.test(image_lists)
        output_path = save_data(g_imgs, o_imgs, args.save_path)
        print('Saved in {}'.format(output_path))
        
    else:
        sys.exit("Incorrect mode")
        
if __name__ == '__main__':
    main()