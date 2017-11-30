# Introduction
Sparse fully connected neural network. Written in Python using numpy and scipy. See online lectures by [Nando De Freitas](http://www.cs.ubc.ca/~nando/340-2012/index.php) for theoretical background.

# Set up
git clone https://github.com/geoffwoollard/nn_sparse.git
cd nn_sparse
pip install -r requirements.txt
unzip data/data.pkl.zip

# Data
The data in `data/data.pkl.zip' are tweets with labelled sentiment (1-grams, dictionary of ~10 000 words). The labelling is not perfect, so complete accuracy is not expected.

# Benchmarking
## the number of neurons
python src/saveParams.py

|number of neurons|train acc|test acc|
|-|-|-|
|3 |0.758406666667 |0.74648|
|4 |0.762307777778 |0.74781|
|5 |0.761155555556 |0.74491|
|6| 0.760576666667 0|.74607|
|7| 0.76579 |0.75044|
|8| 0.768302222222 |0.7549|
|9| 0.766022222222 |0.75304|
|10| 0.770022222222 |0.75609|
|11| 0.76994 |0.75574|
|12| 0.76858 |0.75391|
|13| 0.771351111111 |0.75736|
|14| 0.771258888889 |0.751|
|15| 0.770935555556 |0.75779|
|16| 0.769733333333 |0.75696|
|17| 0.770744444444 |0.75783|
|18| 0.771171111111 |0.75914|
|19| 0.770772222222 |0.75831|
