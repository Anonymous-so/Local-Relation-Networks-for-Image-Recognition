## Local Relational Networks for Image Recognition
A Pytorch implementation of the local relational layer from Local Relation Networks for Image Recogntion [[paper](https://arxiv.org/pdf/1904.11491.pdf)]. 


![Local-Relational-Layer](loca_relation_layer.PNG)

## Background
This is a unofficial implementation of Local Relation Layer. 
There has been another implementation of [local-relational-nets][https://github.com/gan3sh500/local-relational-nets] before, but it cant't run when import it.
Therefore, we make modification and implement a runable version.


## To use the layer:
```
from local_relation_layer import LocalRelationalLayer

layer = LocalRelationalLayer(channels=64,k=7,stride=1,m=8)
...
output = layer(input)
```

## Note:
Since the implement of 2 x k x k geometric priors is not inferred in paper, we are unaware of how to constuct it and initialized it randomly.
It will be altered if I know how to construct it.