# RelaxRNN: Semilinear Recurrent Neural Network with Relaxed Stability Requirements

Add some description here...

Overview of models:

* Relax RNN (--model relaxRNN)
* Antisymmetric RNN (--model asymRNN)
* Cayley RNN (--model calRNN)
* Residual RNN (--model resRNN)
* Vanilla RNN: (--model RNN)


Get started
-----------

To get the latest stable and development versions of this library:

   $ git clone https://github.com/erichson/relaxrnn
   

Here is an example ro run RelaxRNN for pixel-by-pixel MNIST:

python3 driver_RNN_MNIST.py --T 784 --n_units 128 --model relaxRNN2 --epochs 90 --eps 0.01 --lr 0.1 --lr_decay 0.1 --lr_decay_epoch 30 60 80 --beta 0.75 --init_std 32 --gamma 0.01




References
----------
