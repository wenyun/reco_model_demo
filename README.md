# Overview

This code repo serves as a demo of our production recommendation model, highly simplified but structurally similar. Some known differences are as follows.
- The producion model is trained in a distributed way, thus the communication bandwidth and gradient synchronization also impact the training speed. But this simplified verison is trained in a single machine.
- The production model has much complicated input layer, such as different dimensions of dense features, dynamic embedding tables and so on. But this simplified verison only uses static embedding table and tf.gather to retrieve embeddings.
- The production model has slightly different network structure, such as dense feature normalization layer, some extra MLP layers and so on, which are hand tuned by our engineers for years. But here for simplicity, we just keep the main structure of the model, with transformers, MMOE and a few MLPs.

# Usage

The code is tested with python 3.6.8 and tensorflow 1.15. Just type the following command

>> python run.py

to run the code. After testing, one can uncomment the ``production hyperparameters'' to mimick the production model capacity

