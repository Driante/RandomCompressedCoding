# Week 26/01 -02/02

## Todo

- [ ] Explore L+N behavior

  - [ ] Define properly 1st layer, introducing amplitudes and widths
  - [ ] see other weights distribution

- [x] Finish (and write) behavior of covariance matri

  - [x]   First layer define parametrically and behavior of spectrum
  - [x] other distribution for W
  - [ ] read Babad Sompolinski paper      

  ​	

- [x] Retinal data: 
  
  - [ ] talk with roland
  - [ ] relook at Felix paper + Botond paper	
  
- [x] Salinas paper + noise?



## PCA and Covariance Matrix

Let's consider a discretization of our stimulus x in $n$ points. We consider the matrix $U$  s.t. at each colum we have the response vector at the stimulus x:  $U_x = \textbf{u}(x)$. 
These responses are multiplied by the random synaptic weights W to produce the responses of the second layer  to the same stimuli: V=WU. The rows of V can be read now as the discretization of the tuning curve of the N neurons. As said before, these tuning curves are the realization of N independend gaussian processes of mean 0 and covariance matrix K. The covariance matrix can be computed as the overlap between the tuning curves. Indeed, for a single tuning curve:
$$
K_{xx'} = <v_i(x)v_i(x')>_W = <\sum_{jj'}w_{ij}w_{ij'} u_j(x) u_j'(x')>\\
= \sum_{j}u_j(x)u_j(x')
$$
If tuning curves of the first layer are translational invariant $u_j(x) =  f(x-c_j)$. Assuming L large enough, we can replace the sum with an integral, obtaining the correlation function of the gaussian  process
$$
K(x,x') = \sum_j u(j-x)u(j-x') \simeq \frac{1}{L}\int dc_j u(x-c_j)u(x'-c_j)\\
= K(|x-x'|,\sigma^2)
$$

In our original formulation, where $u_j(x) = \frac{1}{(\pi \sigma^2)^{1/4}} exp(- (x-c_j)^2/2\sigma^2)$, $K(x,x') = exp(-(x-x')/4\sigma^2)$

Now let's consider the N x n matrix V. Let's suppose that is mean centered (for simplicity).We can read the dimensionality of the coding manifold described in the activity space by the N neurons, computing the eigenvalues of the matrix $C^N =  \frac{1}{n}VV^T$ It's element are the correlation between the tuning curves of the N neurons, that is
$C^N_{ik} = \int dx v_i(x) v_k(x) $ . For example, it is possible to relate the behavior of the eigenvalues to the actual dimensionality of the underlying manifold . A method proposed by Gao et al. , for example, propose the principal ratio
$$
PR = \frac{(Tr(C^N))^²}{Tr(C^N)^2}
$$
( measure of how many eigenvalues matters). Anyway, in our case computing the eigenvalue of such a matrix is hard. The entries are $C^N_{ik} = \sum_{i,k,j,j'}w_{ij}w_{kj'}(\int dx u_j(x)u_{j'}(x))$  . Nevertheless, the eigenvalues are the same of the sample correlation matrix $C^n = \frac{1}{N} V^TV$ . Its entries are precisely the similarity of the response vector of the second layer to different stimuli :$C^n_{x x'} = \textbf{v}(x)^T\textbf{v}(x')= \sum_i w_{ij} w_{ij'} u_j(x)u_{j'}(x') $ . This follow a Wishart distribution, of expected value the covariance matrix of the gaussian process K. Strictly, this is true only if $n<N$ . Otherwise the N bound the number of eigenvalues different form 0 to be n-N. Taking the limit of N large enough to capture the eigenspecturm of the true covariance matrix, the eigenspectrum of $C^n$ approach the on of K.

If the covariance function is translational invariant, its spectrum can be found simply taking the fourier transform. Example, if $K(x,x') =  exp(-(x-x')^2/4\sigma^2)$ , $\hat{K}(k) = \sigma exp(-\sigma^2 k^2) $  



,that is the case of gaussian tuning functions, then the eigenvalues scale as $\lambda_k \sim \sigma	exp(-\sigma^2 k^2/2)$  . We can obtain a power law eigenspectrum simply using a covariance function whose fourier transform is a power law. N=1000

![](/home/simone/Documents/Neuroscience/Random_coding/notebooks/PCAvsσhighN.png)

Finite size effect.
Studying the eigenvalue distribution of the matrix $C^n$ through random matrix theory is hard. If the covariance matrix is diagonal, then the eigenvalues follow the marchenko pastour distribution.



Distribution of widths
Let's try to extend the previous theory to an heterogeneous layer of different widths. To mantain the translational invariance of tuning curves, we assume having a layer of L neurons arranged on the line with centers distributed equally spaced in $l$ positions ($c_j = 1/l,...,1$). On each center, we associate $L/l$ neurons of different charactersitics. In our case, we will choose gaussian tuning function with different gain and different widths:
$$
u_{jm}(x) = A_m exp(- \frac{(x-c_j)^2}{2\sigma_m^2}) \qquad j=1,...,l \qquad m= 1,...,L/l
$$
The kernel function become now $K_{xx'} = \sum_{jm}u_{jm}(x)u_{jm}(x') = \sum_m A_m^2 \sigma_m exp(-(x-x')^2/4\sigma_m^2)$  where we integrated over j. The eigenspectrum can be found using fourier decomposition, and we obtain $\hat{K}(k) = \sum_m A_m^2 \sigma_m^2 exp(-2\sigma_m^2 k^2)$. Therefore, if we want to obtain a certain eigenspectrum , we have to solve the equation:
$$
\sum_m A_m^2 \sigma_m^2 exp(-2\sigma_m^2 k^2) = \hat{f}(k)
$$




Writing the first layer tuning curves as $v_i(x) = \sum_{jm}w_m(j) u_{jm}(x) = \sum_m\int dc_j w_m(c_j)u_m(x-c_j)$ , exploiting the convolution

## L+N

Let's fix a total cost C for the number of neurons: $\alpha L + \beta N = C$. 
Since $\int dx u_j(x) =1$ and $\int dx (v_i(x)-<v_i(x)>)^2$ (actually the system is insensible to the mean of the firing rate), we can simply redefine 
$$
\alpha' L + N = C'
$$
and use $\alpha'$ as the ratio between the cost of adding one neuron to the second in terms of neurons of the first one. Since L>>N, we will consider $\alpha' <1$. Es setting $\alpha=0.1$ will correspond to adding one N if we lower L of 10 units. Imaging all the neurons in the first layer to be the same (with the same noise), we can compute the tradeoff between L and N. Intuitively, increasing N should be advantegeous until we hit the limit set by the first layer tuning curves, that is  $\epsilon = \frac{L}{\sigma^2 \eta^u}$ , where $\eta^u$ is the variance of the noise of the first layer. This noise will correlate the noise in the second layer, giving a noise correlation matrix $\Sigma = \eta I + \eta^u W W^T./ (ZZ^T)$  where ./ is intended by elementwise division.
Having many parameters to explore, we make a selection. We fix $\eta$ to be low, and $\eta^u$ moderately high (high limit on info coding of first layer).

![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/LvsN.png)
LvsN for starting value of L=500. For high $\sigma$ we have an higher bound 

<img src="/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/LvsNhighN.png" alt="wi" style="zoom:100%;" />

![LvsNhigh η](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/LvsNhigh η.png)



How does the optimal sigma change? We can try to explore this situation keeping fixed $\alpha,\eta,\eta_u$

![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/NvsL.png)

Increasing the cost, we move the optimal sigma to higher values.

![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/NvsLhighC.png)

Increasing the noise on the first layer, we simply move the optimal sigma to higher values: for fixed value of $\eta$: ![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/LvsNη_uincreasing.png)





# Week  03/02 - 09/02

- [ ] - [x] 

- [ ] Covariance matrix: look at distribution of $\sigma$ performance

  - [ ] Energy function?
  - [ ] Longer gradient descent?
  - [ ] Read methods of bethge + variance of posterior (try analytics)

- [ ] Implement working memory model

## Retina and smooth pursuit

Look for phenomena related to smooth pursuit that can be investigated similarly to Franke's paper.
Idea: retinal motion for analog conditions and see which effects are due to retinal signals and which
to motor control (upstream process). E.g. inhanced sensitivity

## PCA and Covariance Matrix

Centers distributed equally spaced in $l$ positions ($c_j = 1/l,...,1$). On each center, we associate $L/l$ neurons with different tuning prop-erties. In our case, we will choose gaussian tuning function with different gain and different widths:
$$
u_{jm}(x) = A_m exp(- \frac{(x-c_j)^2}{2\sigma_m^2}) \qquad j=1,...,l \qquad m= 1,...,L/l
$$
The kernel function become now $K_{xx'} = \sum_{jm}u_{jm}(x)u_{jm}(x') = \sum_m A_m^2 \sigma_m exp(-(x-x')^2/4\sigma_m^2)$  where we integrated over j,assuming dense centers distribution. The eigenspectrum can be found using fourier decomposition: $\hat{K}(k) = \int dk K_{x,x'} exp(i\pi k)$ and we obtain $\hat{K}(k) = \sum_m A_m^2 \sigma_m^2 exp(-2\sigma_m^2 k^2)$. Therefore, if we want to obtain a certain eigenspectrum , we have to solve the equation:
$$
\sum_{m=1}^{L/l} A_{m}^2 \sigma_m^2 exp(-\sigma_m^2 k^2) = \hat{f}(k)
$$
If we have enough neuron and independent synapse, the spectrum is the same on the second layer.

### Finite m approximation

### Large m approximation

We can approximate the sum with an integral$
\int d\sigma A(\sigma) \sigma^2 exp(-\sigma^2 k^2) P(\sigma)$, rewriting and defining $M(\sigma)= P(\sigma)A(\sigma)\sigma^2$ we obtain that the condition is $\int d\sigma  M(\sigma)exp(-\sigma^2k^2) = \hat{f}(k)$   This is analog to the problem considered in Harris' paper. If we assume a power law dependence of $M(\sigma) = \sigma^n$, we obtain the well known integral
$$
\int_{\sigma_{min}}^{\sigma_{max}} d\sigma \sigma^n exp(-\sigma^2k^2)
$$
where $\sigma_{min/max}$ are determined by the probability distribution.

Otherwise, is  a mess. Even if synapse are idnependent from neuron to neuron:$v(x) = \int dc_j w(c_j)u(x-c_j)$ The covariance function is now$<v(x)v(x+x')> = <\int dc_j dc_{j'} w(c_j)w(c_{j'}) u(x-c_j) u(x'-c_{j'})>_W$ . If $<w_j w_{j'}>  \delta_{jj'}$

## L vs N

Why the N remain constant while $\sigma$ change? Actually we have a very flat minimum in that zone.

![](/home/simone/Documents/Neuroscience/Random_coding/notebooks/LvsNη_uincreasing_zoom.png)

 This is due to the fact that in that regime, increasing L increase the information contained in the first layer, while increasing N increase the ability of the second layer to read it out. We can not write this balance analytically, but numerically computing the fisher information it clearly comes out.

## Optimize over a set of $\sigma$

We should choose a set of  $\{\sigma_m\}$  such taht we minimiwe the total error function. Solving this optimization problem is dififcult. Recall that we have to minimize the RMSE of the second layer. This can be written as a loss function
$$
\mathcal{L} = RMSE = \frac{1}{T}\sum_t \int dx (\hat{x}(x)^t  - x)^2
$$
Where our extimate, that will be a function of the noise is the average over the posterior distribution:
$$
\hat{x}(x)^t = \int dx' x' p(\textbf{r}|x) = \frac{1}{Z}\int dx' x' exp(-\frac{|v(x)-v(x')+\eta^t|_2^2}{2\eta^2}
$$

since the noisy response is assumed to be gaussianly distributed $r_i = v_i(x) + \eta_i = \sum_j w_{ij}u_j(x) + \eta_i$ Note that we are interested only in the case where the $\sigma$ is not trivial, therefore, we cannot use the limit of N very large. Otherwise, in that case we could simply minimize the fisher information (this is valid unluckly for lots of computations).
We know from numerical simulations that our function has a non trivial minimum when N is low.

# Week 10/02-16/02

### Todo

- [x] Optimization for different $\sigma$
  - [x] minimize over different $\sigma$ and check the symmetry of the problem if the amplitude is the same
  - [ ] write multiamplitudes function in such a way that they are differentiable
  - [x] check with numerics that breaking the symmetry (allowing different weights) break triviality of energy landscape

- [ ] Implement working memory model

  - [x] Implement gaussian bump
  - [x] Write using SDE in Julia: how to implement time varying noise process?
  - [ ] Implement working meomry

- [ ] Summer schools

  - [ ] https://ws2.mbl.edu/studentapp/studentapp.asp?courseid=MCN Woodshole
  - [ ] https://www.fens.org/Training/CAJAL-programme/CAJAL-courses-2020/CN-2020/Application-form---CCN-2020/ Lisbon

- [ ] If cosine tuning minimize motor errors, conciliate with a loss function for random projections

  



### Error

Problem: we want to minimize the error function that is defined stochastically.
The error function indeed is the the Mean Square Error. For a given realization of the network, we have a noise model $r = v(x) +  \eta = \sum_j w_{ij}u_j(x) + \eta$  
$$
E[(\hat{x}(r)-x)^2|x,r] = \int dr p(r) \int dx (\hat{x}(r)-x)^2p(x|r)
$$
The best estimator is given by the MMSE estimator, that is the average over the posterior:
$$
\hat{x}(r) = \int dx p(x|r) x 
$$


One could do some analytics and show that the loss function can be described as
$$
\varepsilon = \int dx x^2 - \int dr p(r)  \hat{x}^2(r) = E[x^2] - \int dr p(r) \Big(\frac{\int dx xp(r|x)}{\int dx p(r|x)}\Big)^2
$$
Where 
$$
p(r) = \int dx p(r|x)p(x) = \int dx \frac{1}{\sqrt{2\pi\eta^2}} exp(-\frac{(r-v(x))^2}{2\eta^2})
$$
The problem of computing the error can be reduced to the computation 
$$
\int dr \frac{1}{2\pi \eta^2} \frac{(\int dx x exp(-\frac{(r-v(x))^2}{2\eta^2})^2}{\int dx exp(-\frac{(r-v(x))^2}{2\eta^2}}
$$


For low N we also have to average over the distribution of synaptic matrix W, since the minimum can strongly depends from this. The problem lies in the high dimensional distribution of r.

A possible solution could be the following:
define the loss to be the montecarlo extimate of the RMSE:
$$
\mathcal{L} = \epsilon(\sigma; W) = \frac{1}{N_t} \sum_t \int_0^1 dx (\hat{x}(r^t) - x)^2\\
	\hat{x}(r^t)  = \frac{\sum_m x_m h_m(r^t)}{\sum_m h_m(r^t)}\\
	h_m(r^t) = p(r^t|x_m) = exp(r^t v(x_m)/\eta) \\
	r^t = v(x) + \eta^t \qquad v_i(x) = \frac{1}{Z_i}\sum_j w_{ij}u_j(x)
$$
This function is fully differentiable numerically .

Then, the problem could be solved with SGD at each trial. At each step of the algorithm
$$
\sigma  \mathrel{-}= \frac{\partial }{\partial \sigma}\int dx (\hat{x}(r^t) - x)^2 \qquad \text{for t}=1...N_t
$$
In 1D it seems to give good results, at least comparable with grid search method.

![](/home/simone/Documents/Neuroscience/Random_coding/notebooks/sgd4Net1sigma.png)







# Weeks  17/02 - 29/02

## Todo

- [ ] Summer schools application

- [ ] Sharpee and Sompolinsky paper

- [x] Rewrite the code

  - [x] solve the problem of differentiation with uncostrained nulbe rof neurons/population: how to differentiate steprange

  - [ ] switch to train in flux?

    

- [ ] Write code for differentiation

  - [ ] random configuration
  - [ ] minimum in unconstrianed space
  
  



## 2 widths

With two widths, if we keep the same normalization coefficient, the resulting tuning curves
are simply the superposition of two independent gaussian processes. Indexing as 
$$
v(x) = \frac{1}{Z}(\sum_j w_{ij} u_{j,\sigma_1}(x) + \sum_j w_{ij} u_{j,\sigma_2}(x))
$$
Usually the solution is quite trivial due to the symmetry of the problem.

To break the symmetry we should require that
$$
v(x) = \frac{1}{Z} (A_{\sigma_1} \sum_j w_{ij} u_{j,\sigma_1}(x) + A_{\sigma_2}\sum_j w_{ij} u_{j,\sigma_2}(x))
$$
Question: which constraints we should imose? 

Define two ensembles of populations of neurons. Respectively made up by $L_n$ neurons.
The total number of neurons is $L_1 + L_2 = L$ . Each population is allowed to have a gaussian tuning function $u^n_j(x) = A_n exp(-\frac{(x-c^n_j)^2}{2\sigma_n^2})$   where the centers of each population are arranged to tile the space uniformly $c^n_j = \frac{1}{L_n}j$  with $j=1,...,L_n$ We impose a constraint on the total neural activity of the first layer in the following way
$$
\int dx \sum_j^{L_1} u^{(1)}_j(x) + \sum_j^{L_2} u^{(2)}_j(x) = R \\
L_1A_1\sigma_1 + L_2\sigma_2A_2 = R
$$
If we would have imposed the same constraint on the square of the tuning curves, we would have obtained
$$
L_1A_1^2 \sigma_1 + L_2A_2^2\sigma_2 = R
$$

### Alternative  constraint

Until now we imposed that for each neuron's tuning curves, the variance of the responses is =1. This constraint strongly depend from the realization of the random weights. An alternative way would be imposing that ON AVERAGE the standard deviation of the response is constant. Let's define $u_j(x) = A exp(-(x-c_j)^2/2\sigma^2)$ , $\alpha = \int dx u_j(x) = A\sqrt{2\pi\sigma^2}$,

$\beta = \int dx u^2_j(x) = A^2\sqrt{\pi\sigma^2}$ 

Until now we imposed
$$
\int dx [v_i(x) - \int dx v_i(x)]^2 = \int dx (\sum_j w_j u _j(x))^2 - \alpha^2 \sum_jw_jw_{j'} = 1 \\
\sum_j w_jw_{j'} \int dx u_j(x)u_{j'}(x) - \alpha^2 \sum_jw_jw_{j'} = 1
$$
This of cours eimply that A is dependent from the realization of the random weights for each tuning curve. An alternative is imposing  that the average over the tuning curves is constant, that is, if $w_j \sim \mathcal{N}(0,1/L)$
$$
<\cdot >_i = \beta - \alpha^2 = 1
$$
For a single tuning curve, this mean that 
$$
A^2(\sqrt{\pi\sigma^2} - 2\pi\sigma^2) = 1 \\
A^2 = \frac{1}{\sqrt{\pi}\sigma - 2\pi\sigma^2}
$$

Of course this can be done in the case where there is no non linearity between the 1st and second layer. This normalization works( pay attention to finite size effects)
Using vonMises, we obtain that $u_j(x) = A exp(\frac{1}{(2\pi \sigma)^2} cos(2\pi(x-c_j))$ . In this case $\beta = A^2 I_0(\frac{2}{(2\pi\sigma)^2})$  $\alpha = A I_0(\frac{1}{(2\pi\sigma)^2})$

We can extend this normalization including more sub population in the second layer. Since every random weight is independent, the equations become, defining $\alpha_i = \int dx u^i_j(x)$  and $\beta_i = \int dx (u^i_j)^2(x)$ 
$$
\frac{L_1}{L}(\beta_1 - \alpha_1^2)  + \frac{L_2}{L}(\beta_2 - \alpha_2^2) = 1 \\
L_1A_1^2(\sqrt{\pi\sigma_1^2} - 2\pi\sigma_1^2) + L_2 A_2^2(\sqrt{\pi\sigma_2^2} - 2\pi\sigma_2^2) =L
$$
We can recognize the term $\int dx \sum_ju_j^2 = C^2$ , and the equatio can be rewritten
$$
2\pi (L_1A_1^2\sigma_1^2 + L_2A_2\sigma_2^2) = C^2-L
$$
The problem is now in fixing $C^2$.

Ultimately we have to fix a parameter which is the ratio between the amplitudes. Every other constraint  can be rreduced to this.

# March 2020

## Todo

- [ ] Summer schools application
- [x] Book Basel
- [x] 2 widths problem
  - [ ] Make it fully differentiable and optmize over number of neurons
  - [x] rewrite gaussian process with the proposed normalization
  - [ ] learnability of the function
  - [x] Go on with the analytics
  - [x] Numerical simulation in a intelligent way to show optimality in some cinfigurations
  - [ ] check network decoder, write it and write code for error curves
- [ ] wrtie global error, check with one sigma and then extend
- [ ] Random moving objects

## 2 width problem

Summary:imposign the same constraint of before, we write the following equation for the first layer  tunign curves.
$$
L_1A_1^2(\sqrt{\pi\sigma_1^2} - 2\pi\sigma_1^2) + L_2 A_2^2(\sqrt{\pi\sigma_2^2} - 2\pi\sigma_2^2) =L
$$


an intuitive relationship could be relate the number of neurons in each population to their width $\frac{L_1}{L_2} = \frac{\sigma_2}{\sigma_1}$ , but still we have to imose a relationship between $A_1$ and $A_2$ . Introducing a constraint on the first layer does not add anything more, since still we will fix the sum of the neural cost to an arbirtary value (before it was sigma dependent),; which is equivalent to fix the ratio between the amplitudes.

Doing: simulation varying $\sigma_1 -\sigma_2$ and the ratio between $A_1/A_2$ . question: does exist a point that does better then the single width? At which ratio?

We did the following: we vary $\sigma_1-\sigma_2$, for a fixed unbalanced population of $L_1 =400,L_2 = 100 $ neurons. We also varied the ratio between their amplitude $A_1/A_2$ from 0.1 to 3. We aim to find 
1) if there is any configuration that achieve a better error (strongly)with respect to the single width.
2) at which ratio of the amplitudes and at which pair of width.
Surprisingly we find that the configuration that achieve the best error is at $c \simeq 0.5$ and when $\sigma_1 $ is relatively large and $\sigma_2$ is small.

![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/sigma2.png)

At the top is show the heatmap of the error for different width pairs, for the optimal c. At the bottom, the error is plotted when $\sigma_1$ is held fixed at the high value and we vary $\sigma_2$ 

Problem: is a very strong result?  Try to vary for fdfferent network realization and see if the minimum is always less, and not solely due to the high variance of the error in the regime of global errors.It seems to be very weak, as taken ON AVERAGE. In the sense that the same values of parameters are not always optimal for all the networks.
Viceversa, it seems that, with every network realization, it exist a configuration with two widths that improve the error.

### Analytical intuition?

We can write the tuning curves of the second layer neurons as a superposition of two gaussian processes
$$
v(x) = GP(0,K_1(x,x')) + GP(0,K_2(x,x'))
$$
where the covariance functions are given by $ K_i(x,x') = \frac{L_i}{L}A_i^2 \sqrt{\pi \sigma_i^2}exp(-\frac{\Delta x^2}{2\sigma_i^2})$ 

For simplicity we will assume $L_i=L/2$ . We denote as 

$F_i = \frac{L_i}{L}A_i^2 \sqrt{\pi \sigma_i^2}$

the total variance of the response is given by $V = F_1 + F_2$ , while the constraint impose that $V - \pi(A_1^2\sigma_1^2 + A_2^2\sigma_2^2) = 1$

The contribution coming from the large error, when is greater than $\sigma_2$,can be estimated in the following way:
$$
\varepsilon_{g2}^2 = \frac{1}{6\sigma_2^2} erfc(\sqrt{N"})
$$
where $N" = \frac{VN}{2(1+\sigma_\eta^2)}$   and $\bar{\varepsilon}_{g2} = \frac{1 + \sigma_2^2 + \sigma_2}{3} \simeq \frac{1}{3}$




 This resemble the approximate scaling of the global errors of a single width. We also have another type of error, comprised between the two correlation functions. The typical size of this kind of error squared will be of order $\frac{\sigma_1^2 + \sigma_1\sigma_2 + \sigma_2^2}{3}$.  To estimate its probability we can make the following assumptions. Let's say that if $\sigma_1 < \Delta x < \sigma_2$ , we can indicate $v(x) = c(\sigma_2) + \tilde{v}(x)$ . That is we approximate every point inside the larger correlation length as constant, plus a perturbation that will be Gaussian with variance $\tilde{v}(x) \sim \mathcal{N}(0,F_1)$ . The argument of the error function become therefore $N' = \frac{F_1N}{4(1+\sigma_\eta^2)}$ 
$$
\varepsilon_G^2 = \frac{\sigma_2}{\sigma_1^2}\frac{\sigma_2^2+\sigma_1^2 + \sigma_1\sigma_2}{6} erfc(\sqrt{N'})
$$
using the rough approximation for the error function $erfc(\sqrt{N}) \sim exp(-N)$
$$
\varepsilon = P(\varepsilon_l)\frac{L \eta \sigma_1 \sigma_2}{N[L_1A_1^2\sigma_2 + L_2A_2^2\sigma_1]} +A \frac{\sigma_2}{\sigma_1^2}\frac{\sigma_2^2+\sigma_1^2 + \sigma_2\sigma_1}{6}erfc(\sqrt{\frac{F_1N}{4(1+\sigma_\eta^2)}}) + B \frac{1}{\sigma_2^2}\frac{1+\sigma_2 + \sigma_2^2}{6} erfc(\sqrt{\frac{VN}{4(1+ \sigma_\eta^2)}})
$$

Now, even if we assume that the probability of local error is almost 1, we have still that the probability of global errors are approximated and with a proportionality constant that may change depending from lot of things. 

We made an extensive number of exmperiments, and overall they give very ambiguous results. We operate in the folowing regime: we compute the error for a $\sigma = [\sigma_1 \sigma_2]$ , increasing gradually c, the contributions of $\sigma_1 < \sigma_2$, trying to lower the local errors. What we obtain is incredibly variegate and depends from the realization of the network.

<img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_high1.svg"  /><img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_mhigh1.svg" alt="double_width_mhigh1"  /><img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_mlows1.svg" alt="double_width_mlows1" style="zoom:50%;" /><img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_mhigh1.svg" alt="double_width_mhigh1"  /><img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_mlows1.svg" alt="double_width_mlows1" style="zoom:50%;" /><img src="/home/simone/Documents/Neuroscience/Random_coding/notebooks/double_width_lows1.svg" alt="double_width_lows1" style="zoom:50%;" />

The problem was that the minimum is very noisy and therefore changing simply random weights can improve the situation or worsen it.

# April 2020

## TODO

- [ ] 2 widths problem
  - [ ] Make it fully differentiable and optmize over number of neurons
  - [x] rewrite gaussian process with the proposed normalization
  - [ ] learnability of the function
  - [x] Go on with the analytics
  - [x] Numerical simulation in a intelligent way to show optimality in some cinfigurations
  - [x] check network decoder, write it and write code for error curves
- [x] Random moving objects: start to think a project
- [x] Finish the paper

### Machine Learning view of our problem

A 1D latent variable $x$ is supposed to generate a set of N responses $\mathbf{v}$,  described by a random realziation of a gaussian process. The covariance function is assumed to be s.e., s.t. noise free

$K(\mathbf{v}(x),\mathbf{v}(x')) \propto exp(-(\Delta x^2)/(2\sigma^2))$.

We access only a set of noisy measurements of $v$, namely $\mathbf{r} = \mathbf{v}(x) + \mathbf{\eta}$. How we can perform regression on this noisy response to learn the decoding function $\hat{x} = f(\mathbf{r})$ , knowing a set of p training point $\mathbf{r} -> \hat{x}$ . How the rapidity of learning depends from $\sigma^2$ ?

Generally it is possible to relate the learning curve to the spectrum of the covariance function $K(x,x')$  for data without input noise. How input noise affect the learnability of the function?

Our aim is to learn the function $\hat{x} = f^*(\mathbf{v})$ , possibly generalizing to noisy variations of the input. 
Let's consdier a feedforward architecture with an infinitely wide layer, minimizing  the loss function through gradient descent is equivalent to kernel regression with the neural tangent kernel (NTK). Now, we would like to relate the content in frequency of $\mathbf{v}$, and possibly the input noise affecting the measurements, to the generalization error

# May 2020

## TODO

- [ ] Multiscale problem
- [ ] Decoder
- [x] Random moving objects: start to think a project
- [ ] Write a table of comparison between different approaches of coding w tuning curves



## Multiscale problem

We analyzed the problem of encoding the 1D variable with 2 widths $[\sigma_1,\sigma_2]$. We related the amplitude of one tuning curves to the other $A_2 = c A_1$. We then fixed the variance of each tuning curve to be EXACTLY 1. We split evenly the number of cells between the two population . We also replicated the connecvtivity matrix, such that $w_{ij} = w_{i(j+L/2)}$ , such that when sending $c\rightarrow \infty$ , $\varepsilon (\sigma_1,\sigma_2,c= \infty) = \varepsilon(\sigma_2,\sigma^*_1,c=0)$, for all pairs $\sigma_1,\sigma_1^*$. In order to see if introducing a lower width we reduce the error, we did the following simulations. For a given $W$, we compute the error curve in function of $\sigma_1$ when $c=0$. In order to restrict the hige spae of parameters to explore, we did the following analysis. We fixed $c=0$ and we computed the error curve, finding $\sigma_1^*$. We then fixed a $\sigma_2 < \sigma_2$ and we analyzed how the error curves evolve changing $c$. 
To see: how $\sigma_1^*(\sigma_2,c)$ evolve. A predictable result is that it increases with c to counter the effect of low $\sigma_2$.   We would like also to say something about the error at the optimal width. Generally, if the error at the optimal width decrease or not, depends from the specfic realization of the random matrix. Anyway, it is possbile average over it. If the average  show that there exist an improvement, than is a good sign. Since we are considering the MSE and generally it si very low, we worked on the following measure: $\Delta\varepsilon(c) = \frac{\varepsilon(\sigma_1^*,\sigma_2,c) - \varepsilon(\sigma_1^*,\sigma_2,c=0)}{\varepsilon(\sigma_1^*,\sigma_2,c=0)}$ , that is how the addition of $\sigma_2$ improve the optimal error. The following plot show how the optimal $\sigma_1^*$ and the error evolve introducing a $\sigma_2=0.01$, for 4 different networks.

![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/doublewidth_optimalvsc.svg)



We can see the different wrt the case of very low $\sigma_2$ ![](/home/simone/Documents/Neuroscience/Random_coding/plots/tmp/doublewidth_optimalvsc_σ2=0.008.svg)

Generally, we note that introducing a double width with different partecipation ration may allow the error to improve

## 	 Coding with tuning curves

Wrt to classical measures, Yarrow and Butts introduce Stimulus Specific Information which is defined by

$SSI (s) = <I(r)>_{r|s} = \sum_r p(r|s) \sum_{s'} p(s'|r)log(p(s'|r)) - p(s')log (p(s'))$. $I(r)$ is also called response specific information and depends on the specific response

| Paper                                                        | Stimulus                         | Encoding function                                            | Noise model                                                  | Loss function                              | Constraints  and pop size                                    |                     Parameters explored                      | Results                                                      | Methods Comments                                             |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Tuning curves:to sharpen or broaden. Zhang-Sejnowski-1999    | $x\in R^D$                       | Radially symmetric with constant gain $f(x) = F \phi(\frac{|x-c|^2}{\sigma^2})$with centers uniformly distributed | Generalized Poisson $P(N spikes) = S(N,f(x),\tau)$ + Noise correlation impact | J                                          | Fixed peak activity and neurons/preferred position such that allow continuous approximation |                            Width                             | Fisher Info scale as  $\sigma^{2-D}$                         | Sum over neurons -> sum over stimuli -> integral             |
| Brunel-Nadal 1999                                            |                                  |                                                              |                                                              |                                            |                                                              |                                                              | Relationship between FI and lower bound to MI $I_f \propto ln(J)$ |                                                              |
| Statistically efficient estimation,Pouget -1998 , (Partially NarrowvsWide-Pouget 1999, PopCodes-Pouget Rev 2000) | $\theta \in [0,\pi]$             | Von Mises                                                    | Poisson and iid gaussian                                     | Peak of activity of a RNN                  | Weights of RNN set equal to tuning curves of input(or at least same form) |   Difference in amplitude and width between RRN and Input    | Show that a plausible RNN act as a ML decoder                | Dependence of the estimator from the difference in width between decoder and ecndoe |
| F and Shannon Info in finite neural pop, Yarrow 2012,Tuning Curves, neuronal variability-Butts, Goldman 2006 | $\theta \in [0,\pi]$             | Von Mises (fixed amplitude)                                  | Multivariate gaussian with variance proportional to firing rate $\sigma(\theta)^2 = F\tau f(\theta)$ . Three correlations: diagonal, expdecay, uniform | MI (SSI ,$I_f$ , J  ) Discrimination tasks | Fixed width and peak                                         | Population size. Background  firing rate, noise level-time of integration | MI and $I_f$ both grow with logN even at small popsizes. Their difference is consistent at small population sizes. Difference between MI and J optimal coding at different noise regimes (from flank to peak coding). Pop size determine the transition from peak to flank coding | Numerical estimation of MI                                   |
| The infleunce of PopSize , Yarrow 2015                       | , not uniform priors             | VonMises, Montonic                                           | Poisson                                                      | SSI,FI                                     |                                                              |                                                              |                                                              |                                                              |
| Optimal neural rate coding leads to bimodal firing rate distributions-Betghe 2003 | $x\in R$                         | Fisher Optimized (parabolic and step function)               | Poisson                                                      | MSE                                        | Max and min firjg rate                                       |                       Integration time                       | Discrete to contniuous optimality reducing time window       | Numerical integration for MSE                                |
| When Fisher Information fail, Bethge 2002 (Reassessing Optimal codes... -Berens 2012) | $x \in R$                        | Monomodal (several considered)                               | Poisson                                                      | MSE                                        | Max firing rate and others                                   |        Width, integration time , also tuning function        | Optimal width depend on time of integration FIsher optimal tuning function can be non optimal wrt MSE Analysis of fisher optimal codes | Numerical integrtion of MSE and analytical computation of FI for different tuning curves |
| Tuning curve sharpening for orientation selectivity, Series  | $\theta \in [0,2\pi]$            | Feedforward model with recurrent conections of retina lng-V1 | Poisson                                                      | Information                                |                                                              |                 Sharpening vs non sharpening                 | Sharpening introduce correlation that are not beneficial     |                                                              |
| How Behavioral Constraints May Determine Optimal Sensory Representations,Salinas 2006 | $x \in R$                        | gaussians and monotonic, heterogeneity allowed               | Gaussian with variacnce dependent from the mean              | Approximation of downstream motor function |                                                              |     Width and shape for different function approximation     | Shape of optimal tc depends on behavioral constraints        |                                                              |
| Efficient coding heterogeneous population, Simoncelli 2014   | $x\in R$  with non uniform prior | Monomodal (no specification)                                 | Poisson                                                      | J (and related quantities, but not MSE)    | Mean total firing rate (gain), number of neurons (density)   |                                                              | Density proprotional to prior, constant gain                 | Define a convolutional population and then transform it according to some change of parameters coupling density and width. FIx fisher info for the convolutional population (that, implicitlty, depends on the width, here is the hack) |
| Implications of Neuronal Diversity on Population Coding- Sompolinski 2006 | $\theta\in [-\pi,\pi]$           | VonMises + fluctuation in the parameters                     | Multivariate gaussian                                        | J                                          |                                                              |                                                              |                                                              |                                                              |
| Error-based analysis of optimal tuning functions, Meir 2010  | $x\in R$ , gaussian prior        | Gaussian with homogenoeus width (or diagonal covariance matrix in the multivariate case) | Poisson                                                      | MSE                                        |                                                              |        Width, integration time, variance of the prio         | Existance of optimal width that decrease with increasing integration time | Write MMSE as a gaussian estimator allow to write MSE in a closed form. Still it is assumed that the number of neurons and width is "sufficiently high" to cover the stimulus space |

 

## Generalization of optimal coding with tuning curves

* Stimulus $x\in R$ dsitributed according $p(x)$ 
* N parametrized neurons $u_j(x,\{\theta_j\}) = A_j exp(- \frac{(x- c_j)^2}{2\sigma_j^2})$ 
* Model of noise $p(\mathbf{r}|x) = f(\mathbf{u},\mathbf{r})$
* Constraint(s) (reasonable ?) Constant response across stimuli $\sum_j u_j(x) = R$ / maximum firing rate of single neuron $A_j = A$ 
* Loss function $\varepsilon = \int dx p (x) \int dr (\hat{x}(r)- x)^2 p(r|x)$ 
* Optimal estimator $\hat{x}(r) = \int dx x p(x|r)$
* Problem: find $argmin_{\{\theta_j\}} \varepsilon$

Stated like this, the problem has 3N free parameters. The choice of the prior, constraint and loss function has each one some problem and they have been explored in different ways.

### Uniform Prior on a bounded region of the stimulus space

If $p(x) \sim \mathcal{U}[0,1]$  (without loss of generality), a natural  assumption is that all stimuli are "efficiently well encoded". Let's consider for example gaussian tuning curves with a maximal firing rate. The total number of spike for a given stimulus is $R(x) = \sum_{j=1}^N u_j(x) $ . To obtain an analytical estimate of this quantity we may consider constant maximal firing rates  and width and equally spaced centers. This introduce of course edge effects, but quantify them analytically is "hard". If we admit an array of equally spaced neurons, we can compute the limit of $N->\infty$ for stimuli in the middle of the stimulus space. This gives $R = N A \sqrt{2\pi }\sigma$ , independent from x. 

This reasoning can be made more rigorous considering Von Mises distribution where we have a translational invariant stimulus and we obtain $R = AN I_0(k)$ . All this mean that for every $\varepsilon $ small, we can find an $N_c (\sigma)$ such that $|R(x)/N_c-R/N_c| < \varepsilon$ for all x.

Let's consider the fisher information as a lower bound for the local error (writing the J in the case of gaussian noise):

$J(x) = \frac{1}{\sigma_\eta^2}\sum_j A^2 \frac{(x-c_j)^2}{\sigma^4} exp(-\frac{(x-c_j)^2}{\sigma^2})$ the limit of $N-> \infty $ exist and gives (in the central stimuli) 

$J = \sqrt{\frac{\pi}{2}}\frac{N A^2 }{\sigma_\eta^2\sigma}$ . Again, this mean that exist $N'_c(\sigma)$ such that $|J(x)/N'_C - J/N_c| <\varepsilon$ for all small $\varepsilon$. Now, we can formulate the problem in many ways. Numerically we can investigate the oscillations of J, but the analytical computation gives a good idea about the asymptotic behavior.  The following constraint have been explored:

* Fix A (Zhang 1999). For a finite N the problem is ill posed, but once we fix N, we can add the constraint (verificable numerically) to check that the oscilaltions of $J/N$ are below a given $\varepsilon$ .
* Fix R. This imply that $J \propto  \frac{R^2}{N\sigma^3}$ . This is "strange" because adding neurons, in order to keep the overall activity constrained, one lower their gain. Still , if one state that the fisher information have to be more or less cosntant (as before), one can lower the $\sigma$ more if we have more neurons, and the advantage of lowering the $\sigma$ is greater than the disadvantage of increasing the number of neurons. 

An alternative formulation is trying to compute the variance of J across stimulus space. We obtain (for Poisson neurons, which is easier):

$<J(x)>_x = \frac{NA\sqrt{2\pi}}{\sigma}$. The second moment instead is given by:

$<J(x)^2>_x = \int dx A^2\sum_{i,i'}\frac{(x-c_i)^2(x-c_{i'})^2}{\sigma^8}exp(-\frac{(x-c_i)^2 + (x-c_{i'}^2)^2}{2\sigma^2})  = A^2 \frac{\sqrt{\pi}}{16\sigma^7}\sum_{ii'} exp(-\frac{(c_i-c_{i'})^2}{4\sigma^2})[(c_i-c_{i'})^4 - 4(c_i-c_{i'})^2\sigma^2 +12\sigma^4] $   In the limit of N to infinity, the variance goes to 0.



Note that using the MSE, in the case of asymptotic good estimator, is just a different way to fix $\varepsilon$ , but again we cannot expect a different result that "lower sigma until you can". An interesting generalization we can consider is the one of two populations, parametrized by $A_1,A_2$ and $\sigma_1,\sigma_2$ 

In this case $<J(x)>_x = A\sqrt{2\pi}(\frac{N_1}{\sigma_1} + \frac{N_2}{\sigma_2}) $, and, calling $J^2_1 = <J_1(x)^2>_x$ , the second moment will be $<J(x)^2>_x = J_1^2 + J_2^2 + J_{1,2}$  with 

$J_{1,2} = \sum_{i,j}$

We want to check that  J in the case of two widths is greater. Calling $N_1 = \alpha N$ we obtain:



$\begin{cases} \frac{\alpha}{\sigma_1} + \frac{1-\alpha}{\sigma_2} = \frac{1}{\sigma}\\   \alpha \sigma_1 + (1-\alpha) \sigma_2 = \sigma \end{cases}$







### Prior (2D)

we want to analyze the impact of the prior on the optimal arrangement of the tuning curves. Let's take a 2D stimulus space with prior $p(\mathbf{x})$ where $\mathbf{x} \in R$ encoded by tuning curves which depends only on the distnace between the center and the stimulus $u(\mathbf{c}_j-\mathbf{x})$ .  We can define the expected sum of spikes at a given stimulus as $R(x) = \sum_j u_j(x)$

The constraint Simoncelli imposed, in 1D, is the one of constant mean number of spikes $R = \int p(x ) R(x)$ . The loss function considered is the lowe bound on the mutual information. Locally this is expressed by $I_f(x) = \int dx p(x)log(|J(\mathbf{x})|)$ where $|J(x)|$ is the determinant of the fisher info matrix. For a Poisson independent noise model the entries of the diagonal fisher matrix are igven by 

$J_{x_1,x_1} = \sum_j \frac{(\partial u_j(x)/\partial x_1)^2}{u_j(x)}$  



# June 2020

## TODO

- [ ] Correct the paper
- [ ] Add section about input noise: mantain fixed the ratio between the variance and the peak of tc
- [ ] Choice a journal
- [ ] Think what to do next (multiscale problem efficient coding): use information theory constraints (capacity of the channel) to see if we find multiscale solutions (analougsly to Georgjieva)
- [ ] prepare presentation: title and absteact

## Journals



| Journal        | Maximum length                       | Figures  | Nreferences | Pro/cons                  |
| -------------- | ------------------------------------ | -------- | ----------- | ------------------------- |
| eLife          |                                      |          |             | Too much biological       |
| Pnas           | 6 pages                              | 4        |             |                           |
| Plos Comp.Bio. | No                                   | No       | No          |                           |
| Neuron         | 45000 words (excluding star methods) | 8        |             | Modify figures, very hard |
| Manuscripit    | 11122 words -25 pages                | $\sim 7$ | $\sim 60$   |                           |
|                |                                      |          |             |                           |

## Input Noise- Results to present

If the first layer is also  perturbed by iid gaussian noise, the reesponse of the second layer becomes
$$
r_i = v_i(x) + \frac{1}{Z}W\eta_u + \eta
$$
where $\eta_{u_i} \sim \mathcal{N}(0,\sigma_{\eta_u}^2)$  . The likelihood function  is now more complex:
$$
p(\mathbf{r}|x) \sim \exp(- \frac{1}{2}(\mathbf{r} - \mathbf{v}(x)^T)\Sigma_\eta^{-1}(\mathbf{r} - \mathbf{v}(x)^T))
$$
where the noise covariance matrix is given by $\Sigma_\eta = \frac{\sigma^2_{\eta_u}}{Z}WW^T + \sigma_\eta^2I$ 
According to me, would make sense to present the comparison between the input noise and a system with an analog output noise but with just the diagonal term. 

The MMSE estimator has still an interpretation as neural network: 
$$
\hat{x} = \frac{\sum_m x_m \exp(\mathbf{v}^T(x_m)\Sigma_\eta^{-1}\cdot r - \frac{1}{2}\mathbf{v}(x)^T\Sigma^{-1}\mathbf{v}(x))}{Z} = \frac{\sum_m x_m h_m}{\sum_m h_m}
$$
where the parameters of the NN are given by 
$$
h_m = \exp(\sum_i \lambda_{mi}r_i - b_m)\\
\lambda_m = v(x_m)^T *\Sigma^{-1}
$$


Add section with the following comparisons: for almost 0 output noise, comparison between 2nd layer with input noise and 2nd layer with equivalent diagonal noise.

Is decoding directly from the fisr layer a good comparison?