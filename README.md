# NeuGuard

We use jupyter notebook to demo the codes and show the attack results on our pre-trained models.

Here are three separate jupyter files for different datasets.

- **[NeuGuard_ALEXNET_cifar10.ipynb](./NeuGuard_ALEXNET_cifar10.ipynb)** for CIFAR10 dataset
- **[NeuGuard_ALEXNET_cifar100.ipynb](./NeuGuard_ALEXNET_cifar100.ipynb)** for CIFAR100 dataset
- **[NeuGuard_Texas100.ipynb](./NeuGuard_Texas100.ipynb)** for Texas100 dataset.


# Datasets
To run the attack evaluation, you will need to download the three datasets first.

For the Texas100 dataset, the default way to load the data takes long time, we save a npz file to speed up the loading. We provide both way to load the data.
- Name: texas100_data.npz download [here](https://drive.google.com/file/d/1G9-oWyLqiSTDuB2ku6xYY7MVWOur6OOA/view?usp=sharing).
- We load Texas100 data with a randomized order following the file 'random_r_texas100_prune'. Please download it before running the code.



# Run the code

For each code in the jupyter notebook, we have similar running steps.
To check the model used in the paper, please follow the instructions:
1. Run all the cells above the '**# start train**' cell.
2. Load model in the '**# load saved model**' cell and the following two cells to check the model accuracy.
3. Load unsort NSH attack model in the '**# load membership inference attack**' cell and the following five cells to check the model accuracy.
4. Load sort NN attack model in the '**# load NN attack model**' cell and the following two cells to check the model accuracy.
5. Run the cell start in '**# load for metric base attack**' and the following cells to perform the metric based attack.
6. Run the cell start in '**# load for c&w label-only attack**' and the following cells to perform the c&w label-only attack.

# Pretrained models 

We provide pretrianed models used for the work. Specifically, we upload the models trained using the proposed NeuGuard method for all three datasets, and we include both sorted and unsorted NN based attack models, respectively.

The **pretrained models** can be download **[here](https://drive.google.com/drive/folders/1qjPOpicHpCoKcdmL2Iko5f7P6ho5MrIq?usp=sharing)**.

