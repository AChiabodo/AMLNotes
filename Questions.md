# AML questions

1. **batch normalization**
   - Batch normalization is a technique used to normalize the input layer by adjusting and scaling the activations. It helps to reduce internal covariate shift, which can improve the stability and speed of training.
2. **Transfer Learning and Fine Tuning**
   - Transfer learning is a technique where a pre-trained model is used as a starting point for a new task, rather than training a model from scratch. This can save time and resources, and also improve performance. With Transfer Learning, the weights of the pre-trained model are frozen, and only the weights of the last layer are trained on the new task. Fine-tuning is a similar technique where the weights of the pre-trained model (more layers) are updated for the new task, typically by using a smaller learning rate.
3. **Vae vs GAN**
   
   VAE (Variational Autoencoder) and GAN (Generative Adversarial Network) are both generative models, but they work differently. VAE is a probabilistic model that aims to learn a latent representation of the data, while GANs are trained to generate new data by pitting a generator network against a discriminator network. Specifically :
      - VAEs use deep learning techniques to learn a latent representation ($z$) of input data ($x$) and are based on **AutoEncoders**, which are neural networks designed to learn a compressed representation of the input data, but with VAEs the representation is **probabilistic** rather than deterministic.
      - GANs are based on two neural networks playing a "MiniMax" game. The **Generator** creates new data samples with the intention of fooling the discriminator, while the discriminator tries to distinguish between the generated samples and real samples. There are no explicit likelihood distributions in GANs. Instead of minimizing the probability that the discriminator is right (gradient descent), we can maximize the probability that the discriminator is wrong (gradient ascent). The gradient works much better this way.
      Training the two networks together is challenging and can be unstable even with this optimization.
4. **Gradient policy vs q learning**
   - Gradient policy and Q-learning are both reinforcement learning algorithms, but they work differently. Gradient policy is a type of policy-based algorithm that uses gradient descent to optimize the policy, while Q-learning is a value-based algorithm that estimates the value of a state or action.
5. **Self supervised learning**
   - Self-supervised learning is a type of machine learning where the model learns from input data without the need for explicit labels. It can be used to learn useful representations of the data that can be used for other tasks.
6. **Perceptron vs knn**
   - Perceptron and KNN (k-nearest neighbors) are both supervised learning algorithms, but they work differently. Perceptron is a simple algorithm that can be used for binary classification, while KNN is a non-parametric algorithm that is used for classification and regression.
7. **Generative vs discriminator**
   - Generative and discriminator models are both used in GANs. The generator model generates new data, while the discriminator model is trained to distinguish between real and generated data.
8. **Reinforcement learning**
   - Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. The goal of RL is to learn a policy, which is a strategy that specifies the action the agent should take under each possible state. The agent's objective is to maximize the cumulative reward over time.
9. **Domain adaptation**
   - Domain adaptation is a technique used to adapt a model trained on one dataset to work well on a different but related dataset.
10. **Back propagation**
      - Backpropagation is an algorithm used to train neural networks by updating the weights in the network so as to minimize the error of the output.
11. **How optimization is done in cnn**
      - In CNNs (Convolutional Neural Networks), optimization is typically done using methods such as stochastic gradient descent (SGD) or Adam.
12. **Semantic segmentation**
      - Semantic segmentation is a task where each pixel in an image is classified according to the object it belongs to.
13. **Instance segmentation**
      - Instance segmentation is similar to semantic segmentation, but it also involves separating individual instances of the same object class.
14. **Object detection**
      - Object detection is the task of identifying and localizing objects within an image or video.
15. **Multitask vs transfer learning**
      - Multitask learning is a technique where a model is trained to perform multiple tasks simultaneously, while transfer learning is a technique where a pre-trained model is fine-tuned for a new task.
16. **Regression vs classification**
      - Regression is a type of supervised learning where the goal is to predict a continuous output variable, while classification is a type of supervised learning where the goal is to predict a discrete output variable.
17. **Linear regression vs logistic regression**
      - Linear regression is a simple algorithm used for regression problems, while logistic regression is an algorithm used for classification problems.
18. **Batch normalization and its use for domain adaptation**
      - Batch normalization can be used for domain adaptation by normalizing the activations of a pre-trained model to match the distribution of the new dataset.
19. **how to choose hyperparameters**
      - Hyperparameters can be chosen through methods such as grid search, random search, or Bayesian optimization.
20. **What is overfitting**
      - Overfitting is a phenomenon that occurs when a model is trained too well on the training data, and as a result, it performs poorly on unseen data. This happens when the model is too complex and has too many parameters, and it ends up memorizing the training data instead of generalizing to new data.
21. **Why we initialize the weights in cnn**
      - In CNN, we initialize the weights to small random values to break symmetry and avoid zero gradients. This allows the network to learn different features in different layers.
22. **Layer freezing**
      - Layer freezing is a technique used to keep the weights of certain layers fixed during training. This can be useful when fine-tuning a pre-trained model, where we want to train only the last few layers.
23. **Deep learning, how to find the best parameters**
      - Finding the best parameters for deep learning can be done through techniques such as grid search, random search, and Bayesian optimization. These methods allow us to test different combinations of hyperparameters to find the best set that results in the best performance on a validation set.
24. **Shallow learning vs deep learning**
      - Shallow learning refers to traditional machine learning algorithms such as linear regression and decision trees, which have a small number of parameters. Deep learning, on the other hand, is a subfield of machine learning that uses neural networks with multiple layers, and a large number of parameters.
25. **Why non linearity is important in cnn?**
      - Non-linearity is important in CNN because it allows the network to learn more complex representations of the data. This is because linear functions can only represent linear relationships, but non-linear functions can represent more complex relationships.
26. **Describe learning process**
      - The learning process in neural networks is the process of adjusting the weights of the network to minimize the error between the predicted output and the desired output. This is done through backpropagation and gradient descent.
27. **What is ridge regression?**
      - Ridge regression is a variation of linear regression that adds a regularization term to the cost function, to prevent overfitting.
28. **Pca vs autoencoder**
      - PCA (Principal Component Analysis) is a technique used for dimensionality reduction, it transforms the data into a new coordinate system where the dimensions are ordered by the amount of variance they explain. Autoencoder, on the other hand, is a neural network that aims to reconstruct its input, it is used for unsupervised feature learning and dimensionality reduction.
29. **Describe GAN**
      - GANs (Generative Adversarial Networks) are a type of deep learning model that consists of two parts: a generator network and a discriminator network. The generator network generates new data samples, while the discriminator network tries to distinguish between the generated samples and real samples.
30. **RevGrad**
      - RevGrad is a technique used to adapt a model trained on one domain to another domain. It works by reversing the gradient of the domain classifier and using it to update the model's parameters.
31. **ADDA**
      - ADDA (Adversarial Discriminative Domain Adaptation) is a method for unsupervised domain adaptation, it is an extension of GANs, where the generator network is trained to generate samples from the target domain, and the discriminator network is trained to distinguish between samples from the source and target domains.
32. **PAC learning**
      - PAC learning is a theoretical framework for understanding the sample complexity of machine learning algorithms. It provides bounds on the number of samples needed to achieve a certain level of accuracy.
33. **cycle GAN**
      - Cycle GAN is a type of GAN that is able to translate images from one domain to another, it is trained by using cycle consistency loss which trains the model to maintain the same content and style of the input image after translation.
34. **SBADA-GAN**
      - SBADA-GAN is a semi-supervised generative adversarial network (GAN) that is used for cross-domain adaptation. It is designed to adapt a model trained on one dataset to work well on a different but related dataset.
35. **pixel level DA**
      - Pixel level domain adaptation is a technique used to adapt a model at the pixel level. This is used when the input images have different characteristics such as different lighting conditions, resolutions or image noise.
36. **batch normalization for DA**
      - Batch normalization is a technique used to normalize the input data in order to improve the performance of a model. In the context of domain adaptation, batch normalization is used to adjust the model's internal representations to better align with the target domain.
37. **multi source BATCH NORMALIZATION for DA**
      - Multi-source batch normalization is an extension of batch normalization for domain adaptation, where the model is trained on multiple source domains and then adapted to a target domain. This allows the model to learn more robust features that are invariant to multiple source domains.
38. **Domain generalization**
      - Domain generalization is a technique used to train models that can generalize well to unseen domains. The goal is to learn features that are robust to variations in the input data, allowing the model to work well on unseen data.
39. **RNN-LSTM**
      - Recurrent Neural Networks (RNNs) are a type of neural network that can process sequential data such as time series or natural language. Long Short-Term Memory (LSTM) is a specific type of RNN that is designed to handle long-term dependencies in the input data.
40. **attention**
      - Attention is a mechanism used in neural networks to selectively focus on certain parts of the input when making a prediction. This allows the model to attend to the most relevant information in the input, and is particularly useful in tasks such as image and text understanding.
41. **self attention layer**
      - A self-attention layer is a type of attention mechanism that is applied to the input data within a single layer of a neural network. This allows the model to learn to attend to different parts of the input data within the same layer, rather than having to pass the data through multiple layers.
42. **transformer**
      - Transformer is a type of neural network architecture that is particularly well suited for tasks that involve sequential data such as natural language processing. It uses self-attention mechanisms to allow the model to selectively focus on different parts of the input data, which allows it to learn to handle longer sequences of data and make more accurate predictions.