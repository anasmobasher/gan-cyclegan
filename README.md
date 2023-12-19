# gan_cyclegan
Radar signal enhancement (Denoising, Super-resolution) using GANs. 
  End-to-end NN framework for radar signal enhancement. 
## Methodology
Will be added
## Datasets
There are two different datasets
### Micro-doppler signatures of walking people on a treadmill.
This dataset is collected for 12 different persons from two different experiments. In the first experiment, a clean experiment, a subject is walking on a treadmill facing the radar with a clear line of sight, and the radar is positioned at a distance of 3 meters, facing the treadmill’s back. In the second experiment, the noisy experiment, the radar is positioned 10 meters away from the treadmill, and reflective objects are placed in front of the radar to add some sort of noise. This dataset is loaded in two different ways, For semisupervised training, the clean and noisy data of each person are loaded separately, but for unsupervised training, we shuffle all noisy data and shuffle all clean data and load random samples of clean and noisy dataset.  
* Regarding the super-resolution task a low-resolution version of the clean dataset is generated offline  (paired dataset) 
### Range-Azimuth spectra of random objects
We use in the range-azimuth super-resolution a dataset of range-azimuth measurements of random objects. Our objective is to make enhancements on the low-resolution range-azimuth measurements. By doing that we can increase the real-time capability of the radar as obtaining low-resolution range-azimuth measurements is much faster than high-resolution range-azimuth measurements. (low-resolution dataset is generated offline (paired dataset))

## Installation
Will be added

## Results

