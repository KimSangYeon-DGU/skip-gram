# skip-gram

This code is from Assignments 2 of [CS224N course](https://web.stanford.edu/class/cs224n/).

## Naive softmax

### Probability
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?P(O=o|C=c)=\frac{\exp(u_{o}^{\textup{T}}v_{c})}{\sum_{w&space;\in&space;Vocab}\exp(u_{o}^{\textup{T}}v_{c})}" title="P(O=o|C=c)=\frac{\exp(u_{o}^{\textup{T}}v_{c})}{\sum_{w \in Vocab}\exp(u_{o}^{\textup{T}}v_{c})}" />
</p>

### Loss
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J_{naive-softmax}(v_c,o,U)=-\log{P(O=o|C=c)}" title="J_{naive-softmax}(v_c,o,U)=-\log{P(O=o|C=c)}" />
</P>

### Gradient w.r.t <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;v_{c}" title="v_{c}" />
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?-\log{P(O=o|C=c)}&space;\newline&space;=-\log{\frac{\exp(u_{o}^{\textup{T}}v_{c})}{\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c})}}&space;\newline&space;=-\log(\exp(u_{o}^{\textup{T}}v_c))&plus;\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))&space;\newline&space;=-u_{o}^{\textup{T}}v_c&plus;\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))&space;\newline&space;=\frac{\partial&space;-u_{o}^{\textup{T}}v_{c}}{\partial&space;v_{c}}&space;&plus;&space;\frac{\partial&space;\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}{\partial&space;v_{c}}&space;=\newline&space;=-u_{o}&space;&plus;&space;\frac{1}{\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot\frac{\partial&space;\sum_{x\in&space;Vocab&space;}\exp(u_{x}^{\textup{T}}v_c)}{\partial&space;v_{c}}&space;\newline&space;=-u_{o}&plus;\frac{1}{\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot&space;\sum_{x\in&space;Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot\frac{\partial&space;u_{x}^{\textup{T}}v_{c}}{\partial&space;v_{c}}" title="-\log{P(O=o|C=c)} \newline =-\log{\frac{\exp(u_{o}^{\textup{T}}v_{c})}{\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c})}} \newline =-\log(\exp(u_{o}^{\textup{T}}v_c))+\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c})) \newline =-u_{o}^{\textup{T}}v_c+\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c})) \newline =\frac{\partial -u_{o}^{\textup{T}}v_{c}}{\partial v_{c}} + \frac{\partial \log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}{\partial v_{c}} =\newline =-u_{o} + \frac{1}{\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot\frac{\partial \sum_{x\in Vocab }\exp(u_{x}^{\textup{T}}v_c)}{\partial v_{c}} \newline =-u_{o}+\frac{1}{\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot \sum_{x\in Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot\frac{\partial u_{x}^{\textup{T}}v_{c}}{\partial v_{c}}" />

<img src="https://latex.codecogs.com/gif.latex?=-u_{o}&plus;\frac{1}{\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot&space;\sum_{x\in&space;Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot\frac{\partial&space;u_{x}^{\textup{T}}v_{c}}{\partial&space;v_{c}}&space;\newline&space;=-u_{o}&plus;\frac{1}{\log(\sum_{w\in&space;Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot&space;\sum_{x\in&space;Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot&space;u_{x}&space;\newline&space;=-u_{o}&plus;\sum_{x\in&space;Vocab}P(x|c)\cdot&space;u_{x}&space;\newline&space;=-u_{o}^{\textup{T}}&plus;u_o^{\textup{T}}\hat{y}&space;\newline&space;=u_{o}^{\textup{T}}(\hat{y}-y)&space;\newline&space;where&space;\&space;y&space;=&space;\begin{cases}&space;0&space;&&space;\text{&space;if&space;}&space;w=&space;c\\&space;1&space;&&space;\text{&space;if&space;}&space;w=&space;o&space;\end{cases}" title="=-u_{o}+\frac{1}{\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot \sum_{x\in Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot\frac{\partial u_{x}^{\textup{T}}v_{c}}{\partial v_{c}} \newline =-u_{o}+\frac{1}{\log(\sum_{w\in Vocab}\exp(u_{w}^{\textup{T}}v_{c}))}\cdot \sum_{x\in Vocab}\exp(u_{x}^{\textup{T}}v_{c})\cdot u_{x} \newline =-u_{o}+\sum_{x\in Vocab}P(x|c)\cdot u_{x} \newline =-u_{o}^{\textup{T}}+u_o^{\textup{T}}\hat{y} \newline =u_{o}^{\textup{T}}(\hat{y}-y) \newline where \ y = \begin{cases} 0 & \text{ if } w= c\\ 1 & \text{ if } w= o \end{cases}" />



</p>

