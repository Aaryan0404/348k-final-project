# 348k-final-project

For our final project, we plan to extend an embedded CUDA DSL that we have been working on with a research lab at SAIL to optimize the hardware utilization of convolutional kernels used in Convolutional Visition Transformers (CvT). 

The Convolutional Vision Transformer architecture modifies a regular ViT architecture by (A) adding a convolutional token embedding and (B) a Transformer block that leverages a convolutional projection. Researchers have validated that these modifications to the architecture remove the need for positional encodings used in ViTs. Currently, the architecture has been shown to achieves state-of-the-art performance over other ViTs and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs. 

The primary goal of the CUDA DSL is to expose abstractions for memory management and computation on the GPU that enable ML engineers to write custom kernels for attention-based architectures that achieve optimal hardware utilization. 

In our project, our goal is to be able to write custom CUDA kernels for an existing CvT architecture that can be used in a plug-and-play fashion. In order to do this, our project will consist of two phases. First, we hope to add functionality to the DSL to enable writing efficient kernel implementations for convolutions. In particular, we predict this will involve adding ptx wrappers for specialized memory movement, matrix multiplies for convolutions, and additional asynchronous operations to enable pipelining. In parallel and as the second component of our project, we hope to leverage the DSL to write CUDA kernels that both achieve optimal hardware utilization (we hope to be able to outperform the baseline implementation we use from the GitHub repository linked below) and are intuitive to understand (i.e. maximize (TFLOPs)/(lines of code)). We forsee custom efficient kernels being useful for the attention, convolution, and fused MLP layers of the ViT architecture. 

We plan to implement the forwards and backwards pass of each operation of the CvT with a goal to optimize wall-clock time and hardware utilization through our custom CUDA kernels. Though we plan on writing our kernels for on-edge execution (on consumer GPUs like the RTX 4090), a stretch goal for us would be to extend this functionality to custom kernel implementations for industry-grade GPUs like the H100. 

Resources:
- https://huggingface.co/docs/transformers/en/model_doc/cvt
- https://github.com/leoxiaobin/CvT/tree/main
