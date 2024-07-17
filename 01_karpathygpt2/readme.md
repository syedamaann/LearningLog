I finally implemented GPT-2, thanks to Karpathy's lecture! Couldn't run the bigger version due to GPU limits but learned a lot about mixed precision, TF32, kernel fusion, flash attention, DDP, and more. Went from ~1100 ms/batch to ~100 ms/batch. So much fun :)

I'm writing this walkthrough so I don't forget everything I learned. Plus, it might help someone else too!

To start, I used a basic GPT model from my old repo and tweaked it to load weights from GPT-2 on Huggingface in PyTorch. This was necessary to compare our model with theirs and make sure we were on the right track.

We initialized a random model, without any pretrained weights, and began training to evolve these random weights into the needed ones. Here's a quick overview of the steps involved:

First, I got the input data file, the Tiny Shakespeare dataset, and loaded, encoded, and reshaped it for dimensions (B, T). Then, I calculated the loss using cross-entropy from logits and targets inside the forward pass. I added a training loop with the initial goal of overfitting the model on a single data batch. After that, I created a data loader to load new batches each loop, preventing overfitting.

After setting up the basics, I focused on weight sharing. Both the embedding and unembedding matrices have the same shape. The embedding matrix transforms tokens into n-dimensional vectors, while the unembedding matrix converts these vectors back into tokens to calculate the probabilities of the next words in the sequence. Since their functions are complementary, maintaining two separate matrices seemed redundant. So, I used one matrix for both, which reduced our model size and sped up training. If you want more details on weight sharing, check out this [paper](https://arxiv.org/pdf/1608.05859).

For model initialization, I followed the recommendations from the GPT-2 and GPT-3 papers and used a standard deviation of 0.02 for weights. This kept the initial weights near zero but varied enough to prevent symmetry and dominance by any single neuron. Scaling inversely with the square root of the number of layers also stabilized gradient flow by reducing the impact of each layer's output variance.

Next, I made use of TF32. TF32 is a faster way to compute float32 but with slightly reduced numerical accuracy. On A100 GPUs, TF32 can provide up to 10x speedups compared to single-precision floating-point math (FP32). After switching to TF32, the speed improved from ~1000 ms/batch to ~380 ms/batch.

But what's even better is Bfloat16. Bfloat16 is more aggressive in reducing precision than TF32. It keeps the exponent bits and sign bit unchanged, maintaining the same range as FP32. PyTorch handles the conversion, usually keeping weights and gradients in FP32 while converting activations to Bfloat16. This change brought the speed down to ~334 ms/batch.

By now, we've pushed precision down as far as my A100 GPU can handle. While we might lose a bit of accuracy, the speed boost is worth it.
