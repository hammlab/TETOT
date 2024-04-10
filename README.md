# Test-time Assessment of a Model’s Performance on Unseen Domains via Optimal Transport

Gauging the performance of ML models on data from unseen domains at test-time is essential yet a challenging problem due to the lack of labels in this setting. Moreover, the performance of these models on in-distribution data is a poor indicator of their performance on data from unseen domains. Thus, it is essential to develop metrics that can provide insights into the model's performance at test time and can be computed only with the information available at test time (such as their model parameters, the training data or its statistics, and the unlabeled test data). To this end, we propose a metric based on Optimal Transport that is highly correlated with the model's performance on unseen domains and is efficiently computable only using information available at test time. Concretely, our metric characterizes the model's performance on unseen domains using only a small amount of unlabeled data from these domains and data or statistics from the training (source) domain(s). Through extensive empirical evaluation using standard benchmark datasets, and their corruptions, we demonstrate the utility of our metric in estimating the model's performance in various practical applications. These include the problems of selecting the source data and architecture that leads to the best performance on data from an unseen domain and the problem of predicting a deployed model's performance at test time on unseen domains. Our empirical results show that our metric, which uses information from both the source and the unseen domain, is highly correlated with the model's performance, achieving a significantly better correlation than that obtained via the popular prediction entropy-based metric, which is computed solely using the data from the unseen domain.

<hr>
This repository contains the codes used to run the experiments presented in our paper "Test-time Assessment of a Model’s Performance on Unseen Domains via Optimal Transport". 
In this repository, we describe how to obtain the data used for our experiments and the commands used to run experiments with different settings.

### Obtaining the data:
    1. For PACS we download the data using the code from https://github.com/facebookresearch/DomainBed.
    2. For VLCS we obtained the data from https://github.com/belaalb/G2DM#download-vlcs

### Description of the codes in different folders
<hr>

#### Single domain generalization: 
This folder contains the codes to train a model in a single source domain setting using ERM.
Within the folder, we provide the code for different datasets.
    
a. To run the model training with vanilla ERM algorithm, navigate into the specific dataset folder and run the following command 
    
    ./train_erm.sh 0 

b. To evaluate the correlation between accuracy and TETOT on different unseen domains 
    
    ./run_eval_corruptions.sh 0 
    
        
<hr>

#### Multi domain generalization:
This folder contains the codes to train and evaluate domain generalization models in a multi-source domain setting.
Within the folder, we provide the code for different datasets.

a. To run the model training with vanilla ERM algorithm, navigate into the specific dataset folder and run the following command 
    
    ./train_erm_M.sh 0 

b. To evaluate the correlation between accuracy and TETOT on different unseen domains 
    
    ./run_eval_corruptions_M.sh 0 
    
#### Citing

If you find this useful for your work, please consider citing
<pre>
<code>

</code>
</pre>
