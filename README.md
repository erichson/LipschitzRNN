# Lipschitz Recurrent Neural Networks



Overview of models:
-----------
* Lipschitz RNN (--model lipschitzRNN)
* Antisymmetric RNN (--model asymRNN)
* Cayley RNN (--model calRNN)
* Residual RNN (--model resRNN)
* Vanilla RNN: (--model RNN)


Get started
-----------

Here is an example ro run the Lipschitz RNN on the ordered pixel-by-pixel MNIST classification task:

```python3 driver.py --name mnist --T 784  --model LipschitzRNN --n_units 128 --epochs 90 --eps 0.01 --lr 0.1 --lr_decay 0.2 --lr_decay_epoch 30 60 80 --beta 0.65 --init_std 32 --gamma 0.001```


Here is an example ro run the Lipschitz RNN on the permuted pixel-by-pixel MNIST classification task:

```python3 driver.py --name pmnist --T 784  --model LipschitzRNN --n_units 128 --epochs 90 --eps 0.01 --lr 0.1 --lr_decay 0.2 --lr_decay_epoch 30 60 80 --beta 0.8 --init_std 32 --gamma 0.001```



Reference
----------
[https://arxiv.org/pdf/2006.12070.pdf](https://arxiv.org/pdf/2006.12070.pdf)
