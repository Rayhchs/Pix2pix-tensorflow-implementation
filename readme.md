# Implementation of pix2pix
A tensorflow implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://github.com/phillipi/pix2pix). 



## Requisite

python 3.8
tensorflow 2.5.0
Cuda 11.1, CuDNN 8.1.1

	  pip install -r requirements.txt

## Getting Started
* Clone this repo

      git clone https://github.com/Rayhchs/Pix2pix-tensorflow-implementation.git
      cd Pix2pix-tensorflow-implementation
      
* Train

	  python -m main <txt filename> train

* Test

	  python -m main <txt filename> test

* Arguments

 | Positional arguments | Description |
 | ------------- | ------------- |
 | filename | txt file which includes image directions |
 | mode | train or test |
 
 | Optional arguments | prefix_char | Description |
 | ------------- | ------------- |------------- |
 | --epoch | -e | epochs, default=400 |
 | --batch_size | -b | batch size, default=1 |
 | --save_path | -s | Path to save testing result, default= .\result |
      
## Results
Here is the results generated from this implementation:

* Aerial map:

## Acknowledgements
Code heavily borrows from [pix2pix](https://github.com/phillipi/pix2pix) and [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow). Thanks for the excellent work!

If you use this code for your research, please cite the original pix2pix paper:

		@article{pix2pix2017,
  		title={Image-to-Image Translation with Conditional Adversarial Networks},
  		author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  		journal={CVPR},
  		year={2017}
		}


## Reference
 
 https://github.com/awslabs/open-data-docs/tree/main/docs/landsat-pds
