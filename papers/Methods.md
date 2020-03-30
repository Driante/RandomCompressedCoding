# Structure

## Introduction

- Tuning curves are often assumed to have a regular shape. Many studies are made analysing populations of neurons which are responsive in a localised region of the stimulus space (E.g. DS cells, Place Cells, etc...)

- Nevertheless data shows that in deeper regions of the brain, neurons does not always have a simple selectivity profile. For example, data from lalazar and Abbott shows that tuning curves of neurons in motor cortex may arise from random connectivity and possess an high degree of irregularity inside the same population. 

- A more structured  example is the coding schemes of grid cells. Grid cells encode the position of the animal by mean of a set of phases in a set of different periods. This coding scheme introduce high ambiguity at single cell level, but at level of population overcome the performance of classical coding schemes.

- Overall, heterogeneity and irregularities may improve the population code, but also introduce higher levels of ambiguity that can lead to disruptive errors when the population size is finite.

- Generally, a coding scheme must find a compromise between two instances. From one side, resources should be optimized and the information should be compressed and represented with few neurons. On the other hand, neurons are noisy channels and the neural code should be error-correcting. 

- We study the coding properties of a population of neurons whose tuning curves arise from a  general biologically plausible generative model. A two layer network where a sensory layer have to encode a low dimensional variable and trasmit it to a smaller one. 

- Here we ask if the high structure of grid cells is necessary to obtain a code with an high capacity or if we can achieve similar results with less structured codes. 

  ![Caption](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure1.svg)
  
  Different types of encoding schemes. a) Difference between Gaussian bump and grid cells coding the same circular 1-d variable. On the top panel 3 tuning curves example are plotted. On the bottom panel is plotted the activity of the three neurons as x is varied. The Gaussian scheme produce a very smooth manifold; even in presence of neural noise, one can read the stimulus without large errors with few neurons. In the case of grid cells, there are many auto-intersections and, in presence of neural noise, one has to collect the activity of a whole population to avoid large scale errors. At the same time, the activity space is more filled, and this is a sign of high capacity code. b) Feed-forward Neural network. An array of L cells with Gaussian tuning curves encode the same 1-d stimulus.  A specific stimulus evoke a gaussian bump of activity. Then, they random project onto a smaller layer of N cells, producing irregular tuning curves.
  c) Varying $\sigma$ , we change the neural manifold. For very low $\sigma$, each response is basically uncorrelated  with the others. Increasing $\sigma$ responses start to be correlated and we obtain a picture similar to the one of grid cells. For very large $\sigma$ we re-obtain the Gaussian bump scheme, with a very smooth manifold.

## Results

We studied the coding properties of a population of neurons whose tuning curves  heterogeneity arise from randomness in the connectivity structure. Our model consist in a two layer feed-forward network. The first layer is a sensory layer, that encode a low dimensional stimulus into an high dimensional representation, by mean of gaussian tuning curves centred on preferred positions that tiles uniformly the space. This population projects onto a smaller layer of N neurons with an all-to-all random connectivity. The random synapses causes second layer tuning curves to cover all the stimulus space, but with an irregular sensitivity profile. One parameter, $\sigma$ the width of the gausssian of first layer, govern the smoothness of this tuning curves, that is how far two similar stimuli evoke two similar responses.
Increasing $\sigma$ we go from a random uncorrelated code to a smooth one (Fig.1 e).

### Uncorrelated Responses

While the case of a population with Gaussian tuning curves is well studied in the literature, it is instructive to look at what happen when $\sigma ->0$. In this case only one neuron of the first layer respond to each stimulus and the manifold we obtain is very scattered. 





### Random Feed-Forward Network - 1D

![](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure2.svg)

Fig.2  Global-Local Errors a) Two types of error in an auto-intersecting neural manifold. $\bm{v}(x_0)$ is the true response. The trial to trial variability is represented by a grey cloud of possible responses. Noise can cause a local error if the response fall close to a point of the manifold that represent a similar stimulus ($r^{I}$). In presence of many intersections, it can happen that the noise fall onto a point of the manifold that represent a completely uncorrelated stimulus ($r^{II}$) causing a global error. b) Histogram of errors magnitude. The probability of this two type of error is ruled by $\sigma$ .For a fixed N and noise, a very small $\sigma$ will cause very  small local errors (rapid fall of the purple line), but it will produce a substantial number of global errors . Increasing $\sigma$ the local accuracy decrease, but also the number of global errors. c) Analytical form for the two types of errors. d) Numerical simulation and analytical prediction for the (Root Mean Square) error in function of $\sigma$ .  When N is relatively low , it exist a  non trivial  $\sigma$ that balance the two types of error.

- When $\sigma$ start to increase, the manifold become more smooth, and the responses start to be close.  We have two type of errors... Fig2 explained.



<img src="/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure3.svg" style="zoom: 67%;" />

Fig 3 Exponential scaling of the error. a)MSE in function of $\sigma$ for different N, for a fixed noise variance. Every curve reach the optimal performance at a given $\sigma^*$ that decrease increasing N. b) Same plot, but this time the error is showed in function of N, for a different fixed $\sigma$. c) -d) The optimal $\sigma$ decrease exponentially fast with the number of neurons.  This cause the optimal error, which is linear in $\sigma$, to decrease exponentially fast with the population size. e)-f) Optimal quantities for different pairs of N-SNR.

Exponential scaling of the optimal error with the population size. Figure 3 explained.

### Extension to Multiple Dimensions 

The extension to multiple dimensions can be made in many ways. We will consider here the relevant case where the tuning curves of the first layer are simply multidimensional Gaussians. This mean that the neurons encode conjunctively the different dimensions of the stimulus. This case has biological relevance since, for example, the position of a point in the space is encoded in this way in the parietal cortex. Nevertheless, also the case where the cells encode the single dimensions of the stimulus separately is of biological relevance. \cite{Bouchacourt paper}. In the case where the stimulus is $x\in[0,1]^K$ , the following extensions hold:
$$
\varepsilon_l^2 = K\frac{\sigma^2 \sigma_\eta^2}{N}   \\
\varepsilon_g^2 = \frac{1}{\sigma^K}\frac{1}{2}erfc(\sqrt{N'})
$$


### Data Analysis

- A slightly more complicated 3D version of the model has been used in LalazarAbbott to explain the diversity in the shape of the tuning curves in the M1 cortex. In their formulation, the weights were uniformly distributed and the random sum was passed by  a rectifying non linearity, with a varying threshold.
- Knowing that the 3D version has biologically relevance, we first checked that has the same qualitative behaviour of the 1D model. Fig.4





<img src="/home/simone/Documents/Neuroscience/Plots/figure3.svg" style="zoom:50%;" />

Finally, real data shows an high variance of the noise magnitude (insert data analysis?).

We fix the $\sigma_t$ to be the parameter that better fit the data, and we vary the noise level, showing that the size of the population required for doing better than a linear code is quite big and comparable with the number of input a single muscle receive.

![](/home/simone/Documents/Neuroscience/Plots/figure5.svg)



### Decoder

Until now we used an optimal decoder. It exist a network implementation of this decoder that achieve the same result. It requires to know the true responses. Can we actually LEARN this type of decoder, especially in  presence of global errors?

If we would have to learn the decoder, what is the optimal parameter for the encoding?

<img src="/home/simone/Documents/Neuroscience/Plots/figure4.svg" style="zoom:50%;" />





## Methods

Continuous stimulus $x \in [0,1]$   , array of traslational invariant tuning curves, with $c_j = \frac{1}{L} ,\frac{2}{L},...,1$    
$$
u_j(x) = A exp(-\frac{(x-c_j)^2}{2\sigma^2})
$$
If translational invariance, we have vonmises tuning curves that has to defined on an interval 0-2pi.  A general form is:
$$
u_j (x) = A exp(\frac{1}{(2\pi\sigma)^2}cos(2\pi(x-c_j)))
$$


The second layer mix the tuning curves with random synaptic weights 
$$
v_i(x) = \frac{1}{Z_i} \sum_j^L w_{ij}u_j(x)
$$
choosing $w_{ij} \sim \mathcal{N}(0,\frac{1}{L})$ , each tuning curve of the second layer is the realization of a gaussian process, where each pairs of response of single tuning curves

### Global and Local errors

Let $\bf r\bf (x) = \bf{v}(x) + \bf{\eta}$  the noisy response to the stimulus x, for a given realizatio n of W. We want to compute the Mean Squre Error

$$\varepsilon = \int dx dr p(\bf{r}|x)(\hat{x}(r) - x)^2$$ 


We can split the error function in two contributions :local and global. We will denote as  the probability that r will cause a local error: $p(r_l|x ) = p(r|x \qquad s.t. \quad  (\hat{x}-x) <\sigma)$, the same with the global error. We can write the error as the sum of two contributions, using the shortcut $\varepsilon(r_l,x) = (\hat{x}(r)-x)^2$
$$
\varepsilon = \int dr dx p(r_l|x) \varepsilon(r_l) + \int dr dx p(r_g|x) \varepsilon(r_g)
$$
Now, for the local error we know that, using an efficient estimator, the error will be gaussianly distributed with a variance equal to the CRAO bound. Using the mean field approximation that the fisher information is the same for every point of the curve, we obtain: $(\hat{x}(r_l) - x) \sim \mathcal{N}(0,1/J)$ . Therefore we can write the local error as 
$$
\varepsilon_l = P(\varepsilon_l) \frac{1}{J}
$$
where we defined the average probability of having a local error $\int drdx p(r_l|x)$ . For N very large, this will be the only contribution.

For the global error we can do the same reasoning,assuming that the error is uniformly distributed between $[\sigma,1]$ independently from x.

 If PBC, otherwise will be something slightly different since  we have to keep into account that point in the middle has a lower error. Anyway, what matters is that when we integrate, and $\sigma <<1$ $\int d \Delta x_g  p(\Delta x_g)\Delta x_g ^2 = \bar{\varepsilon}_g^2$ is of order 1 .

Then the contribution to the global error is given by
$$
\varepsilon_g = P(\varepsilon_g) \bar{\varepsilon}_g^2
$$
where  $P(\varepsilon_g) = \int dx dr p(r_g|x)$ is the average over the stimuli of the probability that r , given x, will cause a global error. Let's now fix $x'$.  If we imagine having discrete stimuli and uncorrelated, we have an error when $r$ if closer to $v(x')$ than to $v(x)$. The probability that this happen is (we denote as $||^2$ the squared norm)
$$
p(\varepsilon_{xx'}) = p(|r-v(x'|^2 < |r-v(x|^2) = p(\sum_i (v(x)-v(x')_i^2  -2(v(x)-v(x')_i\eta_i <0)
$$
The probability of an error, given x, is the union probability of the events $p(\bigcup_{x'} p(\varepsilon_{xx'}))$ .  This is the probability that exist at least one $x'$. For a given realization of the network, this is hard to compute. Instead, we can average over the distribution of W and obtain 
$$
p(\varepsilon_g) = <<1 - \Pi_{x'} (1-p(\varepsilon_{xx'}))>_x>_W
$$


We can assume that the probability of an error  of a fixed $x$ with any $x'$ is i.i.d. over the distribution of $W$and we can substitute the average of the product with the product of the average (1st approximation):
$$
p(\varepsilon_g) = 1 - (1-<p(\varepsilon_{xx'})>_{W}))^n
$$
We can compute the average of the probability of error between uncorrelated stimuli. Indeed, assuming the variance of the response is 1, the difference between uncorrelated stimuli is normally distributed $v(x)-v(x') = \tilde{v} \sim \mathcal{N}(0,2)$. We have to compute
$$
<p(\varepsilon_{xx'})>_{W} = p(\sum_i \tilde{v}_i^2 - 2\tilde{v}_i\eta_i<0)
$$


The distribution of $d = \sum_i{\tilde{v}_i^2 - 2\tilde{v}_i\eta_i}$ is a combination of chi squared distributions (for $\eta$ =0 is a standard chi square distribution). We can compute its mean $E[d] = 2N$ and its variance $Var[d] = 8N(1+\eta)$.  (This is correct).

2nd approximation:

 We can obtain a pessimistic estimation of the probability of error, assuming that d is distributed gaussianly with the same mean and variance.
Therefore the probability that d is less than 0 is $<p(\varepsilon_{xx'})> = 0.5 erfc(\sqrt{\frac{N}{4(1+\eta)}})$. 

This does not work.

 The distribution has a skewness higher than 0, and this procedure overestimate the probability of error. Indeed from numerical simulation, the behavior is well captured by the cdf of a gaussian with HALF variance $<p(\varepsilon_{xx'})> = 0.5 erfc(\sqrt{\frac{N}{2(1+\eta)}})$. (This is surprisingly good)

Now, let's estimate the number n. This is the number of possible points that, fixed an x, can cause a global error. In the discrete case with uncorrelated stimuli, this is of course the number of stimuli $L-1$. If we start to correlate the stimuli, we can imagine to group the stimuli within a correlation length and consider a discret set of $\frac{1}{\sigma}$ uncorrelated cluster of points. If N is sufficiently high, we can use the 3rd approximation and obtain for the global error
$$
p(\varepsilon_g) = \frac{1}{\sigma}\frac{1}{2} erfc(\sqrt{\frac{N}{2(1+\eta)}})
$$


This last formula is surprisingly robust and does not require any proportionality constant.

<img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/summary_fit.png" style="zoom:150%;" />

## Discussion

