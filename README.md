# Variational Autoencoder - FMNIST
Variational-Autoencoder on FMNIST dataset
erez
## Method:
We implemented Variational Autoencoder. 

*Semi-Supervised Learning with Deep Generative Models, DP. Kingma*,
In Neural Information Processing Systems, 2014. [paper](https://arxiv.org/abs/1406.5298)

Architecture details: (Based on M1 scheme in the paper)
* Input dimension: 784
* Hidden layers dimension: 600 (each) x Two layers
* latent size: 50
* Weights initialized same as the paper with `std=0.001`

We use the loss function:

<img src="AE/images/4.png" width="200"><br>

To improve the results, we used *Disentangled Variational Encoder-Decoder* with `beta` parameter :

<img src="AE/images/5.png" width="200"><br>

## Training:
* We used Fashin MNIST dataset.
* We train the model for 25 epochs.
* We set `beta=0.005`
* Optimization using adam algorithm, with `lr=0.001` and `wd=0.1`

 <p align="center">
 <img src="images/AE/images/6.png" width="400"><br>
  <i>Training loss on test set and train set.
</i>
</p>

## Classification:
We used SVM for classification (in the latent space).
We took only 100, 600, 1000 and 3000 samples to fit the SVM.
The rest used to test the performance:
 <p align="center">
 <img src="AE/runs/Jun20beta0.005/SVM_results_100_-1.png" width="425" /> <img src="AE/runs/Jun20beta0.005/SVM_results_600_-1.png" width="425"/>
 <img src="AE/runs/Jun20beta0.005/SVM_results_1000_-1.png" width="425" /> <img src="AE/runs/Jun20beta0.005/SVM_results_3000_-1.png" width="425"/> 
 
  <i>AAA
</i>
</p>

<img src="AE/images/table1.png" width="425" />

 <p align="center">
 <img src="AE/runs/Jun20beta0.005/generated.png" width="425" /> <img src="AE/runs/Jun20beta0.005/generated.png" width="425"/> 
  <i>GT and GEN
</i>
</p>

