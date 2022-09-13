# NeuGuard

We use jupyter notebook to demo the codes and show the attack results on our pre-trained models.

The folder contains three sepereate jupyter files for different dataset.

- **NeuGuard_ALEXNET_cifar10_attack.ipynb** for CIFAR10 dataset
- **NeuGuard_ALEXNET_cifar100.ipynb** for CIFAR100 dataset
- **NeuGuard_Texas100.ipynb** for Texas100 dataset.


# Datasets
To run the attack evaluation, you will need to download the three datasets first.

For Texas100 dataset, the default way may take some time to load the data every time, we saved a npz file for faster loading. 
Name: texas100_data.npz download [here]().
We load Texas100 data with a randomize order following the file 'random_r_texas100_prune'. Please download it before runing the code.



# Run the code

To check the model used in the paper, please:
1. Run all the cells above the '# start train' cell;
2. Load model in the '# load saved model' cell and the following two cells to check the model accuracy
3. Load unsort NSH attack model in the '# load membership inference attack ' cell and the following five cells to check the model accuracy
4. Load sort NN attack model in the '# load NN attack model' cell and the following two cells to check the model accuracy
5. Run cell in '# load for metric base attack' and following cells to perform the metric based attack.


# Pretrained models 

We will release the pretrianed models for the work after the paper is published.

The pretrained models can be download [here](https://drive.google.com/drive/folders/1qjPOpicHpCoKcdmL2Iko5f7P6ho5MrIq?usp=sharing).

