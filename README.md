# Anomaly detection on high dimensional data
## Abstract
Video anomaly detection is the process of discovering events that differ with the most common events that are presented in a video. The problem of discovering anomalous events in videos has applications in surveillance and security cameras, health monitoring and other automated processes like autonomous driving. 

In this work, first, we design a deep recurrent neural network (RNN) architecture for sequential signal reconstruction. Specifically, we consider video frame acquisition under the principles of compressed sensing and we recover these frames with a sparse representation based method that takes advantage of the similarity between sequential frames. Our model is based on deep unfolding. In particular, we consider that the similarity between two consecutive frames can be captured in the representation domain using the l1 norm and we unfold the steps of a proximal gradient method that solves the l1-l1 minimization problem.

Second, we modify the previous network to employ it for video anomaly detection. The main concept is that we train our network for reconstruction only with videos that do not contain anomalous samples. The network learns to reconstruct well only normal frames and as a result when we give to it a video containing an unusual frame as input, the reconstructed image will not be similar to the original. So our model classifies each frame of the testing set as normal or anomalous based on the reconstruction performance.

Our model performs well on the experimental dataset we used to evaluate its performance, which is the moving MNIST dataset. Trying our network on some other dataset that are used in anomaly detection problems and comparing it with other methods is left for future work.
