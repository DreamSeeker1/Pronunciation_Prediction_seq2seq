"""All the parameters and hyper parameters for the model"""
# Learning rate
learning_rate = 0.001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 512
# Cell type, 0 for LSTM, 1 for GRU
Cell_type = 0
# Activation function used by RNN cell, 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 0
# Number of cells in each layer
rnn_size = 128
# Number of layers
num_layers = 2
# Embedding size for encoding part and decoding part
encoding_embedding_size = 128
decoding_embedding_size = encoding_embedding_size
# Decoder type, 0 for basic, 1 for beam search
Decoder_type = 1
# Beam width for beam search decoder
beam_width = 3
# Number of max epochs for training
epochs = 60
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 0
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5
