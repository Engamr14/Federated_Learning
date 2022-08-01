# Federated_Learning
Federated Learning project for the course "Advanced Machine Learning" at Politecnico di Torino

## Dataset

The used dataset is CIFAR 10

- 60k images

- 50k for training, 10k for testing

- 10 classes

## Class Distribution Among Clients

I have chosen the dirichlet distribution that proposed by a state-of-art scientific paper, where the distribution is controlled by the concentration parameter (alpha)

If alpha approaches infinity, this is an Independent Identical Distribution where the distribution of each class is identical.

If alpha approaches zero, this is non IID where a client dataset can be belong to only one class.

Our scenario is that the number of samples for each client is randomly selected between 200 to 800 samples, which reflects an unbalanced distribution in terms of samples per clients.

## Models

The models I used are:

- LeNet5

- MobileNet V3 Small

- CNN Net

## Federated Algorithms
There are few Federated Algorithms available in Literature like: FedAvg, FedAvgM, FedIR, FedVC, FedProx and more.
However, I used the standard one: FedAvg

## Centeralized Model
I have implemented the centralized model to compare the results of the federated ones.
The following hyperparameters are used:
- server momentum 0.9
- batch size 64
- learning rate 0.01
- weight decay 4e-4
I got the following results:
![image](https://user-images.githubusercontent.com/101885589/182191460-5b455127-0629-42ed-b02a-e3089d8b4dff.png)

## Federated Model
I have used the following hyperparameters:
- local epochs 1
- local batch size 10
- momentum 0
And we got the following results:
![image](https://user-images.githubusercontent.com/101885589/182191911-120cfe7c-8038-453d-ab59-530df621634b.png)
![image](https://user-images.githubusercontent.com/101885589/182192003-d3f81521-a63e-4475-b5c9-020507298f21.png)
![image](https://user-images.githubusercontent.com/101885589/182192100-f467fbda-80a3-42a2-864a-0b3a202375b1.png)

## Normalization Effect
I have tested the Batch Normalization and the Group Normalization to test their results, and these are the results:
![image](https://user-images.githubusercontent.com/101885589/182192429-9b1a5923-4db8-43e6-9dad-e95538ee6b7e.png)


## Conclusion
Increasing local epochs better results (attention must be paid to avoid hyperspecialization on the local dataset owned by each client)

Number of samples per client has less impact than number of classes per client
