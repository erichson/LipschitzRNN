# Lipschitz Recurrent Neural Networks



Overview of models:
-----------
* Lipschitz RNN (--model LipschitzRNN)
* Antisymmetric RNN (--model asymRNN)
* Cayley RNN (--model calRNN)
* Residual RNN (--model resRNN)
* Vanilla RNN: (--model RNN)


Get started
-----------

Here is an example ro run the Lipschitz RNN on the ordered pixel-by-pixel MNIST classification task:

```python3 driver.py --name mnist --model LipschitzRNN --T 784 --n_units 128 --epochs 100 --eps 0.03 --lr 0.003 --lr_decay 0.1 --lr_decay_epoch 90 --beta 0.75 --gamma 0.001 --seed 1 ```


Here is an example ro run the Lipschitz RNN on the permuted pixel-by-pixel MNIST classification task:

```python3 driver.py --name pmnist --model LipschitzRNN --T 784 --n_units 128 --epochs 100 --eps 0.03 --lr 0.0035 --lr_decay 0.1 --lr_decay_epoch 90 --beta 0.75 --gamma 0.001 --seed 1 ```



Reference
----------
[https://arxiv.org/pdf/2006.12070.pdf](https://arxiv.org/pdf/2006.12070.pdf)
