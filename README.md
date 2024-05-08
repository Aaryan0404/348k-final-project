# 348k-final-project

We plan to extend a custom DSL that we have been working on to have efficient kernel implementations for convolutions. We have been working with a DSL in our research which has high-utilization implementations of operations useful for attention. The DSL that we have been working on does not yet have template support for convolutions. 

Our project will consist of two parts. First, we plan to extend the templating of the DSL to add support for convolutions, and second, we will leverage this DSL to implement custom CUDA kernels for the CvT. The convolutional vision transformer will require efficient implementations of attention, convolutions, and fused MLP layers. 

We plan to implement the forwards and backwards pass of each operation of the CvT and beat the compiled PyTorch implementation (with flash attention) in timing in the forwards and backwards pass. If there is time we would like to extend this functionality to implementations for the H100 and RTX4090 for both industry and consumer hardware.

Deliverables: Extend the functionality of a research DSL and beat a compiled Pytorch model in wall-clock time.

Resources:

https://github.com/leoxiaobin/CvT/tree/main
