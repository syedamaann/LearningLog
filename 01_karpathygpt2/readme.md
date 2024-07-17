# I finally implemented GPT-2, thanks to Karpathy's lecture!

I couldn't run the larger version of GPT-2 due to GPU limitations, but I learned a lot about advanced optimization techniques such as mixed precision, TF32, kernel fusion, flash attention, DDP (Distributed Data Parallel), and more. These techniques significantly improved my model's performance, reducing batch processing time from around 1100 milliseconds per batch to about 100 milliseconds per batch. 

I'm documenting this walkthrough to retain what I've learned and possibly help others who are on the same journey.

<br >

### Tweaked basic GPT model to load GPT-2 weights

I began by using a basic GPT model from my old repository and modified it to load GPT-2 weights from Huggingface using PyTorch. This was a crucial step because it allowed me to compare our modified model with the official GPT-2 model to ensure we were on the right track. Loading the pre-trained weights helped validate our model's architecture and functionality by providing a benchmark.

We initialized a random model without any pretrained weights and started training it using the Tiny Shakespeare dataset. This dataset is small and manageable, making it ideal for quick experiments. I processed the dataset by loading it, encoding the text into numerical tokens, and reshaping it to fit the model's input dimensions (B, T) for batch size and sequence length. During the training process, the loss was computed using cross-entropy, which measures the difference between the predicted and actual values. Initially, I aimed to overfit the model on a single batch of data to ensure the model could learn. Then, I set up a data loader to provide new batches in each training loop, which helped prevent overfitting and improved the model's generalization.

<br >

### Shared weights for embedding and unembedding matrices

To optimize the model further, I focused on weight sharing between the embedding and unembedding matrices. Both matrices have the same shape: the embedding matrix converts tokens into high-dimensional vectors, while the unembedding matrix transforms these vectors back into tokens to predict the next word in the sequence. Since their roles are complementary, maintaining separate matrices seemed redundant. By sharing the same matrix for both functions, we reduced the model's size and improved training efficiency. For those interested in the technical details of weight sharing, I recommend reading this [paper](https://arxiv.org/pdf/1608.05859).

<br >

### Initialized weights with recommended standard deviation, stabilized gradient flow

Following the guidelines from the [GPT-2](https://openai.com/index/better-language-models/) and [GPT-3](https://arxiv.org/pdf/2005.14165) research papers, I initialized the model's weights with a standard deviation of 0.02. This initialization strategy ensures that the weights start close to zero but with enough variance to prevent symmetry and dominance by any single neuron. Additionally, scaling the initialization inversely with the square root of the number of layers helps stabilize the gradient flow throughout the network. This approach reduces the impact of each layer's output variance, which is critical for maintaining stable and efficient training.

<br >

### Improved speed with TF32 and Bfloat16

To further enhance the model's performance, I utilized TF32, a new precision format supported by NVIDIA's A100 GPUs. TF32 offers a faster way to perform float32 computations with slightly reduced numerical accuracy. This optimization can lead to up to 10x speedups compared to standard single-precision floating-point (FP32) operations. After switching to TF32, I observed a significant reduction in batch processing time from around 1000 milliseconds to approximately 380 milliseconds per batch.

<br >

![image](https://github.com/user-attachments/assets/ec751d85-d085-4617-bd85-6726d207582d)

<br >

Even more impressive is the use of Bfloat16 precision. Bfloat16 is more aggressive in reducing precision compared to TF32, but it retains the same range as FP32 by keeping the exponent and sign bits unchanged. PyTorch manages the conversion process, typically maintaining weights and gradients in FP32 while converting activations to Bfloat16. This change further reduced batch processing time to about 334 milliseconds. With Bfloat16, we've maximized the precision reduction my A100 GPU can handle. Although there might be a slight loss in accuracy, the substantial speed boost makes this trade-off worthwhile.
