# Model of working memory of Bouchacourt

## Idea

Several ring like attractor projecting to a random network. Feedback of this network help maintaing the activity on the first ones.
Question: the modle is very similar to our, since it produce random tuning curves on the random network , constraining the high dimensional acticity of such a network to lie on a low dimensional manifold.
Actually, it is the same of encoding each dimension of the stimulus separately (so no conjunctive coding, like lalazar) and then project it onto a convergent layer.

## Model

Very similar to our with multiple sensory layers randomly connected to one common layer.

### Sensory layer

First sensory layer is composed by K ring like attractors. For $k=1...K$  the synaptic activation function is given by the convolution with their own spike train.
$$
\frac{ds_j}{dt} = -s_j /\tau + \sum_\alpha \delta (t-t_j^\alpha)
$$
The spike are emitted with a poisson process of rate $r_j$ passing the input to the neuron j through a non linearity:
$$
r_j = \phi(\sum_l W_{jl} s_l +b_j)
$$
In the limits of lots of spikes, we can approximate the equations with additive gaussian noise
$$
\frac{ds_j}{dt} = -s_j/\tau  + r_j(t) + \xi_j(t)
$$
where 
$$
<\xi(t)\xi(t')> = r(t)\delta(t-t')
$$
