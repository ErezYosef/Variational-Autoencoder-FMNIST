# Variational Autoencoder - FMNIST
Variational-Autoencoder on FMNIST dataset
erez
## Method:
We implemented Variational-Autoencoder:
*Semi-Supervised Learning with Deep Generative Models, DP. Kingma*,
In Neural Information Processing Systems, 2014. [paper](https://arxiv.org/abs/1406.5298)

Architecture details: (Based on M1 scheme in the paper)
* Input dimension: 784
* Hidden layers dimension: 600 (each) x Two layers
* latent size: 50
* Weights initialized same as the paper with `std=0.001`

We use the loss function:

<img src="AE/images/4.png" width="650"><br>

To improve the results, we used *Disentangled Variational Encoder-Decoder* with `beta` parameter :

<img src="AE/images/5.png" width="400"><br>

## Training:
* We used Fashion MNIST dataset.
* We train the model for 25 epochs.
* We set `beta=0.005`
* Optimization using adam algorithm, with `lr=0.001` and `wd=0.1`

 <p align="center">
 <img src="AE/images/6.png" width="500"><br>
  <i>Training loss on test set and train set.
</i>
</p>

## Classification:
We used SVM for classification (in the latent space).
We took only 100, 600, 1000 and 3000 samples to fit the SVM.
The rest used to test the performance:
 <p align="center">
 <img src="AE/runs/Jun20beta0.005/SVM_results_100_-1.png" width="350" /> <img src="AE/runs/Jun20beta0.005/SVM_results_600_-1.png" width="350"/>
 <img src="AE/runs/Jun20beta0.005/SVM_results_1000_-1.png" width="350" /> <img src="AE/runs/Jun20beta0.005/SVM_results_3000_-1.png" width="350"/><br> 
  <i>Confusion Matrix, SVM classification with: 100,600,1000,3000 samples.
</i>
</p>

**Classification accuracy:**

<img src="AE/images/table1.png" width="425" />

## Images Reconstruction:
The reconstructed images at the output of the *Decoder* with respect to the images input to the *Encoder*:

 <p align="center">
 <img src="AE/images/gt.png" width="400" /> <img src="AE/images/rec.png" width="400"/><br> 
  <i>Variational-Autoencoder Reconstruction: Encoder input (left) and Decoder output (right)
</i>
</p>

## Data Generation:
*Using Variational-Autoencoder we can generate new samples from the latent space:*
 
 <img src="AE/images/vae.png" width="500"><br>
  <i>New data semples by Variational-Autoencoder.
</i>

*Using classic Autoencoder, we cant generate samples from the latent space:*

 <img src="AE/images/ae.png" width="500"><br>
  <i>New data semples by Autoencoder.
</i>


