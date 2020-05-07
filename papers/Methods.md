# Structure

## Todo

- [x] Write the uncorrelated case-probability of global error
- [ ] Multidimensional case: write methods
- [ ] Write input noise discussion
- [ ] Do Figures (Local error
- [x] Fig.4 rearrange



## Introduction

- Tuning curves are often assumed to have a regular shape. Many studies are made analysing populations of neurons which are responsive in a localised region of the stimulus space (E.g. DS cells, Place Cells, etc...)

- Nevertheless data shows that in deeper regions of the brain, neurons does not always have a simple selectivity profile. A well studied example are grid cells. Grid cells encode the position of the animal by mean of a set of phases in a set of different periods. This coding scheme introduce high ambiguity at single cell level, but at level of population overcome the performance of classical coding schemes.

- Another example, data from lalazar and Abbott shows that tuning curves of neurons in motor cortex may arise from random connectivity and possess an high degree of irregularity inside the same population. 

- Heterogeneity and irregularities may improve the population code, but also introduce higher levels of ambiguity that can lead to disruptive errors when the population size is finite.

- Generally, a coding scheme must find a compromise between two instances. From one side, resources should be optimized and the information should be compressed and represented with few neurons. On the other hand, neurons are noisy channels and the neural code should be error-correcting. 

- We study the coding properties of a population of neurons whose tuning curves arise from a  general biologically plausible generative model. A two layer network where a sensory layer have to encode a low dimensional variable and trasmit it to a smaller one. 

- Here we ask if the high structure of grid cells is necessary to obtain a code with an high capacity or if we can achieve similar results with less structured codes. 

  ![Caption](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure1.svg)
  
  Different types of encoding schemes. a) Difference between Gaussian bump and grid cells coding the same circular 1-d variable. On the top panel 3 tuning curves example are plotted. On the bottom panel is plotted the activity of the three neurons as x is varied. The Gaussian scheme produce a very smooth manifold; even in presence of neural noise, one can read the stimulus without large errors with few neurons. In the case of grid cells, there are many auto-intersections and, in presence of neural noise, one has to collect the activity of a whole population to avoid large scale errors. At the same time, the activity space is more filled, and this is a sign of high capacity code. b) Feed-forward Neural network. An array of L cells with Gaussian tuning curves encode the same 1-d stimulus.  A specific stimulus evoke a gaussian bump of activity. Then, they random project onto a smaller layer of N cells, producing irregular tuning curves.
  c) Varying $\sigma$ , we change the neural manifold. For very low $\sigma$, each response is basically uncorrelated  with the others. Increasing $\sigma$ responses start to be correlated and we obtain a picture similar to the one of grid cells. For very large $\sigma$ we re-obtain the Gaussian bump scheme, with a very smooth manifold.

## Results

We studied the coding properties of a population of neurons whose tuning curves  heterogeneity arise from randomness in the connectivity structure. Our model consist in a two layer feed-forward network. The first layer is a sensory layer, that encode a low dimensional stimulus into an high dimensional representation, by mean of gaussian tuning curves centred on preferred positions that tiles uniformly the space. This population projects onto a smaller layer of N neurons with an all-to-all random connectivity. The random synapses causes second layer tuning curves to cover all the stimulus space, but with an irregular sensitivity profile. One parameter, $\sigma$ the width of the gausssian of first layer, governs the smoothness of this tuning curves, that is how far two similar stimuli evoke two similar responses.
Increasing $\sigma$ we go from a random uncorrelated code to a smooth one (Fig.1 e).

### Uncorrelated Responses

While the case of a population with Gaussian tuning curves is well studied in the literature, it is instructive to look at what happen when $\sigma ->0$. In this case only one neuron of the first layer respond to each stimulus and the manifold we obtain is very scattered. <img src="/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure2_b.svg"  />

Fig. 1.5 Uncorrelated responses a)  2D version of the first panel of Fig 1 e. For a given stimulus $x_0$ , the trial to trial variability is represented as a cloud of possible responses surrounding the true one. If the noisy response happen to be close to a point representing another stimulus, we have an error in the estimate $\hat{x}$ . Since responses are uncorrelated, the error can be very large.  b) Error probability for a fixed noise variance and different number of stimuli L.  The error probability decreases exponentially fast with the population size. The number of stimuli translate vertically the curves, but does not affect the exponent. c) Error probability for a given number of stimuli L and different noise variance. The noise variance modify the exponent of the exponential scaling.

### Random Feed-Forward Network - 1D

![](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure2.svg)

Fig.2  Global-Local Errors a) Two types of error in an auto-intersecting neural manifold. $\bm{v}(x_0)$ is the true response. The trial to trial variability is represented by a grey cloud of possible responses. Noise can cause a local error if the response fall close to a point of the manifold that represent a similar stimulus ($r^{I}$). In presence of many intersections, it can happen that the noise fall onto a point of the manifold that represent a completely uncorrelated stimulus ($r^{II}$) causing a global error. b) Histogram of errors magnitude. The probability of this two type of error is ruled by $\sigma$ .For a fixed N and noise, a very small $\sigma$ will cause very  small local errors (rapid fall of the purple line), but it will produce a substantial number of global errors . Increasing $\sigma$ the local accuracy decrease, but also the number of global errors. c) Analytical form for the two types of errors. d) Numerical simulation and analytical prediction for the (Root Mean Square) error in function of $\sigma$ .  When N is relatively low , it exist a  non trivial  $\sigma$ that balance the two types of error.

- When $\sigma$ start to increase, the manifold become more smooth, and the responses start to be closer each other. 



<img src="/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure3.svg" style="zoom: 67%;" />

Fig 3 Exponential scaling of the error. a) Error in function of $\sigma$ for different N, for $\eta=0.5$.The optimal error is attained at the optimal $\sigma^*$ that decrease increasing N. b) Same plot, but this time the error is showed in function of N, for a different fixed $\sigma$. The error at first decrease exponentially fast until we eliminate the global errors, then the finite width makes the error decrease linearly. Decreasing $\sigma$, we increase the N at which the transition happen, but also the error. c) -d) The optimal $\sigma$ decrease exponentially fast with the number of neurons.  This cause the optimal error, which is linear in $\sigma$, to decrease exponentially fast with the population size. e)-f) Optimal quantities for different pairs of N-SNR. The optimal width, and the error, scale exponentially with the population size, with an exponent that depends from the variance of the noise.

### Extension to K-Dimensional Stimuli

A population of neurons can encode a multi dimensional stimulus in many ways. The way in which the first population respond to a K-dimensional stimulus will also alter the properties of the second layer. In two "extreme" cases, neurons of first layer can be selective to just one single dimension of the stimulus (pure cells) or to all stimulus dimension (conjunctive cells). 

The difference in this two encoding scheme has been exploited in \cite{Bat papers} in the context of head direction coding in bats. Here, we will add results on the trade off between local and global errors when the first layer population is randomly mixed in the second layer. Both cases has biological relevance. A model of random projection of a layer of conjunctive cell encoding a 3D stimulus (a position in the physical space) has been employed to explain the heterogeneity of tuning curves in motor cortex, therefore we will present simulations for this case.  On the other hand, a layer of neurons randomly connected to subnetworks encoding single features of one stimulus has been used to explain the flexibility of working memory.

In Fig.4 we show the results when a population of neurons of pure or conjunctive cells are randomly mixed in a second layer of N cells. Particularly, the number of neurons of first layer is kept constant and the variance of second layer tuning curves across all stimulus space is kept constant too.

In Method we derive the scaling for the global and local errors in the two cases., showing that 
$$
\varepsilon_{l,p}^2 = 3  \varepsilon_{l,c}^2 \\
\varepsilon_{g,p} = ...
$$


In Fig.4 a)-b) the  trade off between local and global errors in function of the width is illustrated.  We can see in both cases an optimal $\sigma$ , decreasing with N, that balance the two contributions. Since the number of neurons in the first layer is the same, the conjunctive case show a rapid increase of the error under a certain $\sigma$ , due to the incapacity to cover efficiently the space. This is also evident in the saturation of $\sigma^*$ and the consequent linear scaling of the error in the red curves of Fig.4 c)-d).  If we plot the ratio between the errors in the two cases as we did in Fig 4 e, this regime corresponds to the yellow region. 

As soon as $\sigma$ is sufficiently high to cover the space in the conjunctive case, this coding scheme outperform the pure case,  due to the slower decay of its global error. One can see this effect in the higher optimal width (blue curve ) in Fig.4 c that also produce a bigger optimal error in Fig. 4c. This is the blue region of Fig4e.  Increasing $\sigma$ increase this difference decrease since global errors become less important. (slightly dependent from $\sigma$ due to our choice of constraint).


![](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure4.svg)

Fig.4 3D generalization a) Error vs width for a first layer made up by conjunctive cells b)Error vs width for a first layer made up by pure cells  c)Optimal error scaling with Number of neurons, for pure(blue) and conjunctive cells(red). d) Optimal width scaling with number of neurons e) Error ratio between pure and conjunctive cells

### Data Analysis

- A slightly more complicated 3D version of the model has been used in LalazarAbbott to explain the diversity in the shape of the tuning curves in the M1 cortex. Briefly

  ​		 tuning curves of that region were supposed to be linearly tuned to the arm position

  - $r = a_0 + \sum a_ix_i$ 

  - Data shows that tuning curves exhibit a lot of heterogeneity, going from very linear shapes to highly irregular ones (Insert a figure?Maybe in the fig 1?)

  - The authors explained this heterogeneity with randomness in the connectivity structure. The model is the 3D version of the random feedforward network  with conjunctive cells, with two difference (uniform distribution and non linearity). 

  - The idea is that the first layer contain the visual information about the postion, and this information is transmitted to the motor neurons through random connectivity.

  - To check if the model was able to explained the data, they compared the histogram of two summary statistics of the neural population (see Methods).

    

  





Firstly therefore we asked how much of the data our model was able to explain. We generate the response of a 3D network to the same stimuli of the data, and we compare the histogram of the same summary statistic for different values of $\sigma$  (see methods for the data preprocessing) We found that this simple version of the model well capture the complexity distribution of the data at a given $\sigma_f$ , while it fails in capturing the distribution of the linear fitting. 

Finally, we want to analyse the coding properties at this best $\sigma_f$ , and compare them with the classical parametric shape of tuning curves. 

- The noise distribution is extremely irregular. The signal to noise variance  has a very broad distribution over neurons, not correlated with the number of trials. Moreover no simple parametric function of the mean is able to explain the noise distribution.
- Therefore, we first analysed the performance of the network over several values of noise variance, checking where is convenient having irregular tuning curves and where is not.
- Then we took a different noise distribution for each neuron (see Methods) and we did the same





![](/home/simone/Documents/Neuroscience/Random_coding/plots/paper_figures/figure5.svg)

Fig. 5 Random vs linear. a) Relative improvement $\Delta\varepsilon = (\varepsilon_l-\varepsilon_i)/\varepsilon_l$	 for a fixed noise variance, in the $N-\sigma$ plane. Negative values, sign that a smoother coding is more efficient, are in the low N-low $\sigma$ region. Keeping fixed $\sigma$, when we increase N, we have a very sharp transition and we go to high positive values, since we eliminated global errors and the local error is very low. Increasing $\sigma$ the improvement diminish, since the tuning curves start to be smoother. b) Error at high noise, for $\sigma = \sigma_f$ and same stimulus space of data. c) Relative improvement in the $N-\eta$ plane, for best $\sigma_f$ . We notice that at very high noise and low N, is much more favourable having a smoother code. d)  Relative improvement for $\sigma_f$ and noise extracted from data, average over 4 group of neurons of that size N. We can see that the improvement saturate at N =100.



### First layer constraints

In our model we deliberately ignored noise in the first layer neurons. Our aim was to show the balance between local accuracy and global noise robustness, in the attempt to construct an efficient code with random responses. Within this setting, the number of neurons L of the first layer need just to be "sufficiently high" to cover the stimulus space. For a given width, if L is too small such that lot of stimuli are encoded just in the tail of the gaussian tuning curves, this will produce no response (or too small) in second layer tuning curves, resulting in an enhancement of both local and global errors. In other words, L set a lower bound on the minimal width.

In a more realistic case, first layer neurons are affected by noise and the information we can extract  in the second layer is bounded. Noise in the first layer introduce correlations in the noise of second layer neurons. This correlations increase as soon as N becomes closer to L...

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
where  $P(\varepsilon_g) = \int dx dr p(r_g|x)$ is the average over the stimuli of the probability that r , given x, will cause a global error. 

Computing this probability in the case of uncorrelated images is hard. In order to obtain an approximation of this quantity, we will suppose to regroup the manifold of true responses in uncorrelated "clusters" . Since the responses to two stimuli at a distance of $\sim \sigma$ are uncorrelated, we can represent the manifold as $\sim \frac{1}{\sigma}$ clusters of points. We will have a global error when the estimate of the stimulus belong to a different cluster that the true response.  



We can compute the average of the probability of error between uncorrelated stimuli. Let's go back to the case  where $x= 1/L,...,1$  and responses to different stimuli are uncorrelated $v(x) \sim \mathcal{N}(0,1)$

 We have an error when the noisy response to the stimulus x happen to be closer to a point representing another stimulus x'  than to the true response:  $|\mathbf{r}-\mathbf{v}(x')|_2^2 < |\mathbf{r}-\mathbf{v}(x)|_2^2$  .  The probability of at least one event of this kind is therefore:
$$
<P(\varepsilon|x)>_W  = 1 - <P\Big(|\mathbf{r}-\mathbf{v}(x')|_2^2 >  |\mathbf{r}-\mathbf{v}(x)|_2^2)   \quad \forall x' \Big)>_W
$$
When averaging over different realizations of the synaptic matrix, the probability of not having an error on $x'$ are iid for different $x'$ . Moreover we have no dependence from x anymore and we can write:

$$<P(\varepsilon)>_W = 1 - (1 - <P\Big(|\mathbf{r}-\mathbf{v}(x')|_2^2 <  |\mathbf{r}-\mathbf{v}(x)|_2^2)>_W)^{L-1} \\
 \simeq L <P\Big(|\mathbf{r}-\mathbf{v}(x')|_2^2 <  |\mathbf{r}-\mathbf{v}(x)|_2^2)>_W$$

Inside the parenthesis can be rewritten as $\sum_i (v_i(x') -v_i(x))^2 + \eta_i^2 - 2(v_i(x)-v_i(x'))\eta_i < \sum_i \eta_i^2$ . Averaging over W, the difference between the response of the same neuron $\tilde{v} = v_i(x)-v_i(x')$ to two different stimuli is a Gaussian random variable. Averaging also over the noise distribution we obtain:
$$
 <P(\varepsilon)>_W =L  \int \Pi_i d \tilde{v}_i \Pi d\eta_i p(\tilde{v}_i) p(\eta_i) \Theta (-\sum_i \tilde{v}^2_i -2\tilde{v}_i\eta_i)
$$
In other words we have to compute the probability that the quantity $d = \sum_i \tilde{v_i}^2 - 2\tilde{v}_i\eta_i$ is less than 0, with $\tilde{v}$ and $\eta$ gaussianly distributed with 0 mean and variance 2 and $\sigma_\eta^2$ respectively.

#### List of approaches to the global error

##### Gaussian approximation

The distribution of $d = \sum_i{\tilde{v}_i^2 - 2\tilde{v}_i\eta_i}$ is a combination of chi squared distributions (for $\eta$ =0 is a standard chi square distribution). We can compute its mean $E[d] = 2N$ and its variance $Var[d] = 8N(1+\eta)$.  (This is correct).

2nd approximation:

 We can obtain a pessimistic estimation of the probability of error, assuming that d is distributed gaussianly with the same mean and variance.
Therefore the probability that d is less than 0 is $<p(\varepsilon_{xx'})> = 0.5 erfc(\sqrt{\frac{N}{4(1+\eta)}})$. 

This does not work.

 The distribution has a skewness higher than 0, and this procedure overestimate the probability of error. Indeed from numerical simulation, the behavior is well captured by the cdf of a gaussian with a scaled variance $<p(\varepsilon_{xx'})> = 0.5 erfc(\sqrt{\frac{N}{f(\sigma_\eta^2)(1+\eta)}})$. (This is surprisingly good)

This $f(\sigma_\eta^2)$ is a non linear function of the variance. For variance going to 0 this quantity goes to 0. For variance much higher than 2 (the variance of $\tilde{v} $) approach the value of 4.









Now, let's estimate the number n. This is the number of possible points that, fixed an x, can cause a global error. In the discrete case with uncorrelated stimuli, this is of course the number of stimuli $L-1$. If we start to correlate the stimuli, we can imagine to group the stimuli within a correlation length and consider a discret set of $\frac{1}{\sigma}$ uncorrelated cluster of points. If N is sufficiently high, we can use the 3rd approximation and obtain for the global error
$$
p(\varepsilon_g) = \frac{1}{\sigma}\frac{1}{2} erfc(\sqrt{\frac{N}{2(1+\eta)}})
$$


This last formula is surprisingly robust and does not require any proportionality constant.

<img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/summary_fit.png" style="zoom:150%;" />



##### 

##### d given lambda

W

### Multi-Dimension

First layer population can have pure and conjunctive encoding. In the pure case we have $L$ neurons monitoring each dimensions of the space for a  total number of K neurons. Second layer tuning curve are the linear combination of these , with weights different for each dimension
$$
v_i(\mathbf{x}) = \frac{1}{Z_p}\sum_k \sum_{j_k=1}^L w_{ij_k} exp(-(x_k-c_{j_k})^2/2\sigma^2)
$$
In the conjunctive case first layer neurons are selective to all the stimulus dimensionality. If we want a similar distance between preferred positions, we should tile the space with a grid of $L^3$ neurons
$$
v_i(\mathbf{x}) = \frac{1}{Z_c} \sum_j^{L^3} w_{ij} exp(-(\mathbf{x-c_j})^2/2\sigma^2)
$$

#### Constraint

Until now I adopted the same constraint of before, imposing that the variance of the firing rates across all the stimulus space is =1. Since we are considering here the transformation from layer to layer and we can rescale arbirtarly the firing rates of the second layer simply rescaling the synaptic weights, this comparison makes more sense. Within this context we obtain (using the unit volume of the stimulus space) $\frac{1}{Z_p^2} = \frac{1}{(\pi\sigma^2)^{1/2} - 2\pi\sigma^2}$ for pure cells and $\frac{1}{Z_c^2} = \frac{1}{(\pi\sigma^2)^{3/2} - (2\pi\sigma^2)^3}$ for conjunctive cells, assuming to rescale the variance of the synaptic weights according to the number of neurons of first layer : $w_{ij} \sim \mathcal{N}(0,\frac{1}{3L}) $ for pure cells and $w_{ij} \sim \mathcal{N}(0,\frac{1}{s*L^3}) $ for conjunctive cells (eventually including sparsity in the matrix).

Note: we could state a constraint on the first layer in the following way. Let's denote as $M_c$ the number of cells in the conjunctive case. The tuning curves of the first layer can be modulated by a maximum firing rate $A_c$, such that $u_j(x) = A*exp(...)$ . Let's suppose the stimulus space is just a cube of side 1,the mean (square  for better comparison with the previous case) activity will be therefore $M_c A_c^2 (\pi\sigma^2)^{3/2}$ . For pure cells it will be $M_p A_p^2 (\pi\sigma^2)^{1/2}$ . We could therefore put the constraint  $\frac{A_p^2}{A_c^2} = \frac{M_c (\pi \sigma^2)}{M_p}$ . 

We could satisfy this constraint fixing the number of neurons in the two layer and then adjusting the amplitudes.Note that if $M_c/M_p \sim L^2$ (fixing the coverage, or distance between preferred poistions) in principle we could have that the two maximum firing rate are comparable, since $\sigma^2$ is very small and $\sigma^2*L^2$ is of order 1. Let's fix for the moment the number of neurons in this way.

  If we impose the same constraint of before of the variance of firing rates in the second layer we obtain (assuming for the synaptic weights the same scaling as $1/M$)

$\frac{A_c^2 Z_p^2}{A_p^2 Z_c^2} = \frac{ (\pi\sigma^2)^{1/2} - 2\pi\sigma^2}{(\pi\sigma^2)^{3/2} - (2\pi\sigma^2)^3}$ 

Note that both in the local error and in the global error, what really matter is the ratio $A/Z $ , therefore changing the constraint on the first layer in this way does not change the results. What may change the result is  fixing the two populations of the first layer to have the same number of neurons. In this way, going to a very low $\sigma$ impact differently the two population since they have different coverage.

#### Local Error

We can compute the local error looking at the fisher information in the two cases. Define the scalar error as the sum of the squared error for each dimension $\varepsilon^2 = \sum_k \varepsilon^2_k$. 
The fisher matrix is the proportional to identity and therefore $\varepsilon^2 = \frac{3}{J_{xx}}$ where $J_{x_k x_k} $ is the diagonal element of the FI matrix.
We can compute it in the two cases (averaged over the weights distribution) obtaining:

for the pure case 
$$
J^p_{x_kx_k} = \frac{\sum_i |\partial v_i(\mathbf{x})/\partial x_k|^2}{\sigma^2_\eta} = \frac{N (\pi \sigma^2)^{1/2}}{3*2 \sigma_\eta^2 \sigma^2 Z_p^2}
$$
where the 3 comes from the fact that each dimension is encoded separately and therefore the derivative act only on 1/3 of the terms. In the conjunctive case, the derivative act on all terms and we have
$$
J_{x_k x_k}^c = \frac{\sum_i |\partial v_i(\mathbf{x})/\partial x_k|^2}{\sigma^2_\eta} = \frac{N (\pi \sigma^2)^{3/2}}{2 \sigma_\eta^2 \sigma^2 Z_c^2}
$$
Obtaining that $\varepsilon_{l,p}^2 = 3 \varepsilon_{l,c}^2$ 

#### Global error 

In the conjunctive case we can extend the same reasoning as before, but the number n of uncorrelated cluster goes like $\frac{1}{\sigma^K}$ . 
$$
\varepsilon_{g,c}^2 = \bar{\varepsilon}_g \frac{1}{2\sigma^K} erfc(\sqrt{N'})
$$
In the pure case, we have that each response is a linear combination of gaussian process of smaller variance (depending on the scaling of the synaptic weights, with our scaling it wil have a variance $1/K$). We can compute the probability of having a global error as the probability of having a global error in at least one coordinate. If these probability are very low, we can sum them to obtain the probability of the union of events. The result is a scaling
$$
\varepsilon_{g,c}^2 = \bar{\varepsilon}_g \frac{K}{2\sigma} erfc(\sqrt{N'/K})
$$
therefore the probabilty of global error has a lower prefactor but slower scaling with N.

### Model and data fitting

<img src="/home/simone/Documents/Neuroscience/Plots/figure4.svg" style="zoom:50%;" />

Fig.7 Data and fits. d) K-S distance between the distribution of complexity measure across different neurons. The minimum of the K-S distance is attained at $\sigma_f$ . The resulting Histogram is plotted in Fig. 5 a. b) Histogram of the R2 distribution. Even if the linear model underestimate the number of neurons with a good linear fit,  still gives a population with a broad distribution. More linearity can be achieved simply by introducing a non-linearity. c) Fraction of variance explained by the principal components.

## Discussion

\textcolor{red}{Where inserting -mixed selectivity neurons -discussion on possible model extents}
Biological neural networks are complex systems; neurons' tuning properties are very diverse and this is probably why the brain is able to perform and learn a series of different tasks efficiently. Focusing on the study of neurons exhibiting a clear and simple selectivity limits our understanding of how the brain encode the information. Non-intuitive, complex responses of single neurons have been showed to possess good coding properties if considered at population level \cite{Fiete2008WhatLocation,Sreenivasan2011GridComputation}. 
The so called "mixed selectivity" neurons, neurons that respond to different combinations of task parameters, have been showed to be good for cognitive purposes \cite{Fusi2016WhyCognition,Barak2013TheTrade-Off}. The common shared idea is an efficient use of the activity space of neurons, that depends from the number of resources (dimensionality), smoothness constraints,noise level and type of task.

The analysis of optimal shape of tuning curves has a long history in the literature \cite{Zhang1999NeuronalBroaden,Seung1993SimpleCodes}, but mainly focused on parametric description of their shape and constraints on the single cell. Some studies analyzed the benefits of random irregularitites on the shape parameters \cite{Shamir2006ImplicationsCoding}, while others derived the optimal heterogeneous population in the case where the stimulus distribution is not uniform \cite{Ganguli2014EfficientPopulations}. 
Here we took another approach describing the whole population using an implicit generative model, making the tuning curves as random as possible constraining only their average smoothness.  
\subsection{Random manifolds}
This has biological motivation. In early sensory areas tuning curves are localized and exhibit a clear selectivity, since they have to monitor the physical space. Once we go higher in the brain regions, neurons mix in a complex way and the selectivity can become much less intuitive. This can happen already at the first stages: for example irregularities in the retina ganglion cells receptive field, due to a non-perfect (maybe random?) sampling from bipolar cells, have been shown to improve the mutual information between the activity and the stimulus \cite{Soo2011FineCells}.
In our work we chose to mix the early inputs through random connectivity. There are several interesting  reasons to study such a setting. Random compression  have been showed to be a quasi-optimal way to preserve distances in sparse data \cite{Candes2006Near-optimalStrategies}, which has given rise to the well developed field of compressed sensing. There is an emergent idea \cite{SuryaGanguliandHaimSompolinsky22012CompressedAnalysis}that the brain may use this strategy to compress information in presence of convergent pathways, therefore we focused our study on the optimal use of limited number of channels.
Random expansion instead generates high dimensional representation of data, giving rise to the so called "mixed selectivity neurons" that allows a better pattern-separation  (\cite{Barak2013TheTrade-Off,Babadi2014SparsenessRepresentations}. 
Recent works \cite{Bouchacourt2018AMemory} shows how randoml
Random weights do not need to be learned; even in the presence of learning, many algorithms assume an initial random distribution of weights.
With the increase of the number of recorded neurons in experiments and the help of tools from other fields like statistical physics, we are starting to unveil how complex information in neural systems is treated at the level of population activity, which is much more than the sum of its components \cite{Saxena2019TowardsDoctrine}.  Studying the coding properties of neural manifolds is crucial since the high dimensional activity of neural population is often supposed to lie in a much lower subspace, and often this is exploited in data analysis adopting dimensionality reduction techniques \cite{Cunningham2014DimensionalityRecordings}. 
This network generates a coding manifold which is randomly oriented with respect to neurons' basis; this means that all the neurons carry more or less the same amount of information about a specific stimulus. As described in \cite{Gao2017AMeasurement,Gao2015OnNeuroscience}, this has some interesting properties. First, from the "scientist" point of view, it implies that despite sampling few neurons we are able to recover the underlying properties with a good accuracy.
On the other side, downstream neurons are able to recover the information contained in the upstream circuits just randomly subsampling a small population of neurons. 
Random manifolds with just smoothness constraints moreover saturate what they called "Neural Task Complexity", which quantify the number of dimensions a neural trajectory actually explores; therefore focusing on this kind of manifolds  reveals limits and bounds on the capacity of neural populations  to encode and represents complex informations.
The previous model gives a simple example of how this kind of manifolds may arise from random connectivity patterns and introduce a simple parameter that we can tune to explore different regimes in terms of intrinsic dimensionality (smoothness).
We studied how the activity space of neurons can be efficiently used keeping into account the trade off between local accuracy and robustness to noise



\subsection{Connection with grid cells}
\textcolor{red}{Grid cells are supposed to act as path integrator. We are totally not mentioning this fact?}
The idea that the brain contains codes with strong error-correction properties is inspired from the work on grid cells. Grid cells encode the position of the animal in the space through a set of phases within different modules and each module is characterized by a period of the tuning curves of the single cells. The periodicity, together with the existence of multiple periods which are assumed to be exponentially small with respect to the stimulus space (range of positions), allow the exponential capacity \cite{Sreenivasan2011GridComputation}.
Similarly, in the RFFN the tuning curves of the second layer contains all frequencies up to a cutoff one, which is given by the tuning width of the first layer. If this tuning width follows the optimal scaling with N, it decreases exponentially, allowing the error to do the same.
We suggested that the same result 

\subsection{Limitations and extensions}
\textcolor{red}{Non-linearity}
We kept our analysis intentionally simple, to analytically illustrate the different contributions to the error and show the principal quantities that affect the coding properties in the case of random tuning curves.
A first remark is that our model predicts negative firing rates (and negative synaptic weights);  this can be easily solved using a different distribution for synaptic weights (gaussian with non-0 mean, uniform, ecc...). This would affect the tuning curves shape that will be described by another type of noise process, not necessarily gaussian, but the underlying principle of balance between local an global errors is unchanged. Adding a non linearity will further modify the noise process; for example  Lalazar et al. \cite{Lalazar2016TuningConnectivity} enforced positivity passing the random sum through a threeshold-linear function; their aim was limiting the coding range  of the neurons, that on average are responsive only to a fraction of the stimulus space. This, keeping fixed the number of neurons, will typically increase the error of both types, since for some stimuli some neurons will not vary their response.
Interestingly, using an exponential function, will generate tuning curves that are an exponentially gaussian process, which is the prior utilized in \cite{article,Wu2017GaussianData}to make inference about the firing rate maps.
Another important approximation is made about the noise distribution. We supposed that noise is isotropic and independent across  neurons, trials, and stimuli. This is a conservative approach. Simulations with poisson neurons, drawing spikes instead of using a rate model, with an integration time long enough