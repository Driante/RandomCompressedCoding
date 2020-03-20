# Methods

## General Methods

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

## Global and Local errors

MSE:  $\varepsilon = \int dx dr p(r|x)(\hat{x}(r) - x)^2$  where r is the noisy response
We can split the error function in two terms:local and global. We will denote as $p(r_l|x ) = p(r|x \qquad s.t. \quad  (\hat{x}-x) <\sigma)$, the same with the global error. We can write the error as the sum of two contributions, using the shortcut $\varepsilon(r_l,x) = (\hat{x}(r)-x)^2$
$$
\varepsilon = \int dr dx p(r_l|x) \varepsilon(r_l) + \int dr dx p(r_g|x) \varepsilon(r_g)
$$
Now, for the local error we know that, using an efficient estimator, the error will be gaussianly distributed with a variance equal to the CRAO bound. Using the mean field approximation that the fisher information is the same for every point of the curve, we obtain: $(\hat{x}(r_l) - x) \sim \mathcal{N}(0,1/J)$ . Therefore we can write the local error as 
$$
\varepsilon_l = P(\varepsilon_l) \frac{1}{J}
$$
where we defined the total probability of having a local error $\int drdx p(r_l|x)$ . For N very large, this will be the only contribution. Instead for the global error, we can do the following approximation. The error will be typically uncorrelated with the exact stimulus,  and can be approximated with a uninform distribution, as long as it is greater than the correlation length. $\Delta x_g = (\hat{x}(r_l) - x) \sim \mathcal{U}[\sigma,1/2]$ (if PBC, otherwise will be something slightly different since  we have to keep into account that point in the middle has a lower error. Anyway, what matters is that when we integrate, and $\sigma <<1$ $\int d \Delta x_g  p(\Delta x_g)\Delta x_g ^2 = \bar{\varepsilon}_g^2$ is of order 1 .

Then the contribution to the global error is given by
$$
\varepsilon_g = P(\varepsilon_g) \bar{\varepsilon}_g^2
$$
Now, the problem lies in estimating the total probability of having  global erors, that is $\int dr dx p(r_g|x)$ . Let's suppose that we have no more an efficient estimator, but a maximum likelihood one. That is given a realization of the noise, the probability that the error is global is the probability that the noisy image is closer to a uncorrelated one. Now, strictly this quantity can be computed as 
$$
\int _{|x-x''|> \sigma}dx'' \int^{x-\sigma/2} _{x-\sigma/2}dx' p(|r-v(x'')|<|r-v(x')|)
$$
following the intuition of before (independnet responses), we can group the points in "correlation clusters" of size $\sigma$, and ask what is the probabiltiy that a noisy response from one cluster is closer to another correlated cluster than to the true response.We can compute the probability that for two uncorrelated stimuli $|v(x) + \eta - v(x')|^2 < |\eta|^2$ .  We can assume that these events are independent for pairs of uncorrelated stimuli and add them Having  $\frac{1}{\sigma^2}$ possible of pairs of uncorrelated stimuli, we can say that the global error probability is given by 
$$
\frac{1}{\sigma^2}\int d\eta p(\eta) p(|v(x)-v(x')+\eta|_2^2<|\eta|)_2^2
$$
Now, we can use the fact that x and x' are uncorrelated, and writing  $\tilde{v_i} = v_i(x)-v_i(x') \sim \mathcal{N}(0,2A^2)$, we want compute 
$$
\int \prod d\eta_i d\tilde{v}_i p(\eta_i)p(\tilde{v}_i) \Theta(\sum \eta_i\tilde{v}_i - \sum \tilde{v}_i^2)
$$
Approximating the behavior as a gaussian