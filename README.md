# Learn machine learning the hard way

Try to implement machine learning models (especially deep learning models) using [numpy](http://www.numpy.org/) as the only dependency.

1. [Linear regression](1-linear-regression.ipynb)
2. [Logistic regression](2-logistic-regression.ipynb)
3. [Multiclass logistic regression](3-multiclass-logistic-regression.ipynb)
4. [Multilayer perceptron](4-simple-neural-network-framework.ipynb)
5. [Convolutional neural network](5-convolutional-neural-network.ipynb)

- [Appendix A. Broadcasting and dot product in numpy](appendix-a-broadcasting-and-dot-product.ipynb)


## Setup Jupyter Notebook theme

```shell
# install jupyterthemes
pip install jupyterthemes

# dark theme
jt -t onedork -fs 10 -altp -tfs 11 -nfs 115 -cellw 88% -T
```

## Run unit tests

```shell
python -m unittest discover . "*_test.py" -v
```
