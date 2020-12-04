# Symmetry breaking solutions in efficient coding problems

Neurons are supposed to be tuned to features of the stimulus space. The map between the stimulus and the mean response of a neuron is called tuning curves. A very common shape of the tuning curves is the so called gaussian-bump, where the selectvity of a neuron is described by a gaussian function peaked at a certain preferred stimulus value. The width of this tuning curves can be viewed as the correlation length of the system, describing how far similar stimul evoke similar responses. Efficient coding hypothesis state that the neurons' responses are such that they maximize the information about the stimulus (+ constraints). Nevertheless, a common hypothesis in this kind of problems is that neurons have homogeneous tuning properties, like width and amplitude. Few study analyzed heterogeneity; among this studies, Simoncelli's work (together with Wei and Stocker) is perhaps the most famous. Nevertheless, he also obtain the heterogeneity deforming a homogeneous population according to a prior on the stimulus. He derive the optimal transofmration, but the result will always depend on the features of the homogeneous pop.

Recent studies advanced the ideas that the brain encode  stimuli in a "scale invariant way" (vague). This would imply the existence of multiple population encoding the stimulus at different levels of detail. Nevertheless, there is no theory of efficient coding considering different widths. Here we try to reanalyze the general problem considering multiple population widths.

## Model

We consider N neurons responding to a stimulus, potentially with any prior $x\sim p(x)$, with gaussian tuning curves centered on a preferred position:
$$
u_j(x) = A_j \exp(-\frac{(x-c_j)^2}{2\sigma_j^2})
$$
The vector of a spike count in a fixed time window (note that we can incorporate the time in the prefactor $A_j$)   is given by a poisson process with rates given by the mean response to acertain stimulus:
$$
p(r|x) = \Pi_j \frac{u_j(x)^{r_j}\exp(-u_j(x))}{r_j!}
$$
This is a classical problem in theoretical neuroscience, nevertheless it has been addressed in very particular cases. Usually, due to the non linearity,  general optimization problems are unfeasibile. 

Problems with homogeneous populations lead to degenerate $\sigma$, is it possible to find a solution where  the optimal $\sigma$ is asymmetric (without of course introducing prior which are asymmetric and so on).   Usually, additional constraint are imposed: the more common one is a single neuron firing rate which cannot exceed a certain threeshold, other requires that the average numer of spikes is below a certain threshold
$$
A_j \le A_{max}\\
\int dx p(x) \sum_j u_j(x) \leq E_{max}
$$

## Error function

One could choos to minimize the expected error in the estimation,assuming that we have an ideal estimator: $\hat{x}(r)$ which, given a spike count, output a value of the stimulus. For example, we can try to minimize the MSE:
$$
\varepsilon^2 = \langle(\hat{x} - x)^2\rangle_{r,x} = \int dx p(x) \int dr p(r|x) (\hat{x}(r) - x)^2
$$
The estimator which minimizes the MSE is the average of posterior distribution
$$
\hat{x}(r)_{MMSE} = \int dx p(x|r) x
$$
and one can arrive to write that the MMSE is precisely the average over responses of the  variance of the posterior distribution given the responses:
$$
\varepsilon^2 = \int d r p(r) \langle (x-\langle x\rangle_{x|r})^2\rangle_{x|r}
$$
What is often done, is assuming that this estimator satisfy the CRAO bound and the variance is given by the inverse of the fisher information $J(x)$,  therefore from (4) we arrive to write
$$
\varepsilon^2 \geq \int dx p(x) \frac{1}{J(x)}
$$
It is worth to recall the Fisher information for such a population which reads:
$$
J(x) = \sum_j \frac{(u'_j(x))^2}{u_j(x)} = \sum_j \frac{(x-c_j)^2}{\sigma_j^4} u_j(x)
$$
This has not a close form. What is done, usually, is assuming an homogeneous population with equally distributed centers, approximate the sum with an integral  (or taking the average over x, even if we have $1/J$) obtaining the usual expression for the FI (uniform prior)
$$
\varepsilon^2 = \frac{\sigma}{NA\sqrt{2\pi}}
$$
which has the clear problem which lead to 0 errrors in the narrow width limit. 



### Via global error: Fiete's approach

The computation sketched before does not consider that we have actually a fintie number of neurons and we have to 'cover' sufficiently the stimulus space (for all stimuli, at least one neuron has to respond). Nevertheless, making computations with finite number of neurons is hard. Numerical computations by Bethge show that the optimal width goes actually as $\frac{1}{N}$ .

We will now discuss how to obtain analytically a scaling compatible with this numerical result. We consider for the moment a bounded  uniform distribution of the stimulus $p(x)\sim \mathcal{U}[0,1]$ (we ignore edge effects, as usual this can be made rigorous using PBC, but this complicate the anayltics). For comodity, we will consider a fixed maximum spike count $A = \frac{1}{\sqrt{2\pi}}$. A rescaling of this quantity is equivalent to a rescaling of the time window considered.

#### Approximated computation

If the stimulus is uniform, a common assumption is that the mean spike count is independet from $x$ : $\sum_j u_j(x) = \lambda(x) \simeq \lambda = N\sigma $ (substituing the sum with the integral).

#############################################################################################

_Note that the following analysis is hard to generalize to the case where also the total number of spikes is kept constant, namely the maximum firing rate is adjusted such that $\lambda$ is constant. In this case is harder to calculate the proabability of 0 spikes, since for $\sigma$ low the average effect does not take into account the different tiling of the stimulus space. This is expected, since under energy constraints narrow tuning is even more favorable_

#############################################################################################

A recent paper (Fiete et al.) solved this contraddiction considering the global ambiguities that we can have, in this case given by cases when no neurons spike. Its approach was equivalent to minimize the inverse of the mean fisher information subject to the 0 spikes condition:
$$
minimize \qquad \varepsilon_J = \frac{\sigma}{AN}\\
s.t. \exp{(-AN\sigma)} \leq p_0
$$
which leads ot the following lagrangian (KKT conditions) and optimal non-trivial tuning width 
$$
\varepsilon = \frac{\sigma}{AN} + \beta \exp{(-AN\sigma)}\\
\sigma^* = \frac{\log(A^2N^2\beta)}{AN}\\
\varepsilon(\sigma^*) = \frac{\log(A^2N^2\beta)}{A^2N^2} + \frac{1}{A^2N^2}
$$
We can extend this idea to 2 widths population. Denoting $\alpha = \frac{N_1}{N}$, the straightforward (incomplete) extension is to consider the constrained optimization problem (again, for simplicity we consider $A=1$)
$$
\varepsilon = \frac{\sigma_1\sigma_2}{N(\alpha\sigma_2 + (1-\alpha)\sigma_1)} + \beta\exp{(-N(\alpha\sigma_1+(1-\alpha)\sigma_2))}
$$
Let's consider for a moment the case of equal population $\alpha=0.5$. The symmetric solution $\sigma_1=\sigma_2$ is now no more a minimum but a saddle point . Indeed, considering the 'isoJ' hyperbole  $\sigma_2 = \frac{C(1-\alpha)\sigma_1}{\sigma_1 - CN\alpha} $ , the point $\sigma_2=\sigma_1$ are always on the lowest "isoglobalerror" line $\sigma_2 = \frac{log(\frac{1}{p_0}) - \alpha\sigma_1}{1-\alpha}$. Viceversa, all the other points on that line will be on a lower "isoJ" curve (to see it, just solve the equation with one width and trace the curves). This is for example the error landscape when $\beta=1, N=120$

![](/home/simone/Documents/Neuroscience/Random_coding/plots/illustrative/simple_loss.png)

The problem is still ill-defined, since the minimum this time tend to make one sigma as small as possible, since the other one will avoid the global errors. Th reason is that is an incomplete description. We could improve it with the following consideration: when $\sigma_1 \rightarrow 0$, we should recover the solution with just one population. Another possible reasoning, is that when one population does not spike, the error is fully determined by the FI of the remaining population. This lead us to consider the following loss function
$$
\varepsilon = \frac{\sigma_1\sigma_2}{N(\alpha\sigma_2 + (1-\alpha)\sigma_1)} + \frac{\sigma_1}{\alpha N} \exp{(-N(1-\alpha)\sigma_2)} + \frac{\sigma_2}{N(1-\alpha)}\exp{(-N\alpha)\sigma_1} + \beta\exp(-N(\alpha\sigma_1+(1-\alpha)\sigma_2))
$$
For the same values of before, this function has a symmetry-breaking solution where $\sigma_2\neq\sigma_1$ is optimal (analytically not easy to find, but numerically simple)

![](/home/simone/Documents/Neuroscience/Random_coding/plots/illustrative/loss_extented.png)

How to find analytical intuition of this splitting? 
Btw, note that this formulation is similar to the situation where we are considering one spike only.

#### Posterior variance

We can make the previous approximate reasoning more rigorous considering the actual distribution of the variance of the posterior.

Knowing the noise model, we could try to derive the posterior form through Bayes theorem:
$$
p(x|r) = \frac{p(r|x)p(x)}{p(r)}
$$
where the problem lie in computing the marginal. Passing to log, we could write:
$$
log(p(x|r)) = - \sum_j \frac{r_j (x-c_j)^2}{2\sigma_j^2} -\sum_j u_j(x) + log(p(x))
$$


Now if we assume constant $\lambda $ , an uniform prior on the stimulus space, and a constant tuning width, we obtain a gaussian approximation for the posterior
$$
log p(x|r) \approx  - \frac{(x-\hat{x}(r))^2}{2\sigma^2/R} +C
$$
##############################################################################################################_APPROXIMATION: we are assuming constant prior and constant mean number of spikes. What happens if this is not the case? The posterior is no more gaussian. Describing $\lambda(x) = \bar{\lambda} + \delta\lambda(x) $ the normalization constant would be, properly, become_
$$
Z = \int dx \exp{(-\frac{(x-\hat{x})^2}{2\sigma^2/R}) - \delta \lambda(x) + log(p(x))}
$$
_how to treat it? Let's suppose that we have the mode of the posterior $\hat{x}$, at the second order expansion we obtian:_

$ p(x|r)  = C - \frac{(x-\hat{x})^2}{2\sigma_p^2}$ where the variance is $\frac{1}{\sigma_p^2} = \frac{R}{\sigma^2} + \sum_j\frac{\sigma^2 - (x-c_j)^2}{\sigma^4}u_j(x)|_{x=\hat{x}}$ where the second term is a correction that can be rewritten as $\frac{\sum_j u_j(\hat{x})}{\sigma^2} - J(\hat{x})$ 

#############################################################################################################

where the estimator is simply the average of the neurons preferred positions $\hat{x}(r) = \sum_j c_j \frac{r_j}{R}$ and $R = \sum r_j$  is the total spike count, and C can be imposed through normalization. Therefore we can write the  MSE as the average of the variance of the posterior distribution:
$$
\varepsilon^2 = \int dr p(r) \frac{\sigma^2}{R} \approx \int dR p(R) \frac{\sigma^2}{R}
$$
where we integrated over all the possible spike counts, which we know having a Poisson distribution  $p(R)  = \frac{\lambda^R e^{-\lambda}}{R!}$ . Note that if we make a gaussian approximation for this quantity and we assume $\lambda$ is sufficientlgaussian distribution plus perturbationsy large, we reobtain the result from the  Fisher information. Nevertheless, we have a degeneracy in the variance in the case of 0 spikes. To avoid it, we could esplicitly use the fact that the error in the case of 0 spikes is given by a random guess on the stimulus space, and denote as $\beta$ this quantity. The expression for the error becomes now:
$$
\varepsilon^2 = \sum_{R>1} p(R) \frac{\sigma^2}{R} + e^{-\lambda}\beta
$$
The first term can be rewritten as the exponential integral function and we obtain the following optimization problem:
$$
\varepsilon^2 = \sigma^2 e^{-\lambda}[Ei(\lambda) - \gamma - \log(\lambda)] + e^{-\lambda}\beta
$$
Using the fact that $N\sigma= \lambda$ Deriving this expression we obtain the following equation for the optimal $\lambda$.
$$
\lambda[e^\lambda -1 + (2-\lambda)(Ei(\lambda) - \gamma - log(\lambda))] = \beta N^2
$$
The term inside the parenthesis is dominated by $e^\lambda$ and we obtain
$$
\lambda^* \simeq W(\beta N^2)  \rightarrow \sigma^* \sim\frac{W(\beta N^2)}{N}
$$
which is very similar to the solution found by Fiete considering the Fisher Information

#### Extension to 2 (K) subpopulations

Let's suppose now that we have 2 subpopulations, arranged perfectly to cover the stimulus space. The gaussian approximation for the psoterior can be written as function of the spike countsof the two subpopulations. Denoting again $\lambda_i= \sum_j^{N_i} u_{i,j}(x) = N_i \sigma_i$ aand $\lambda = \sum_k \lambda_k$ the posterior will be
$$
p(x|r)\sim \mathcal{N}(\hat{x},\sigma_p^2)\\
\frac{1}{\sigma_p^2} = \sum_i^K \frac{R_i}{\sigma_i^2}\\
\hat{x} = \sigma_p^2\sum_i\frac{\hat{x_i}}{\sigma_i^2}
$$
The error becomes now quite intractable, even for small values of K:
$$
\varepsilon^2 = \sum_{R_k s.t. R\neq 0}\Pi_k p(R_k) \sigma_p^2 + e^{-\lambda} \beta
$$
for two populations forexample we obtain:
$$
\varepsilon^2 = e^{-\lambda}\sum_{R_1>0,R2>0} \frac{\lambda_1^{R_1}\lambda_2^{R_2} }{R_1!R_2!}\frac{R_1 \sigma_2 + R_2 \sigma_1}{R_1R_2} + e^{-\lambda_1}\sum_{R_2>0}p(R_2) \frac{\sigma_2^2}{R_2} + e^{-\lambda_2}\sum_{R_1>0}p(R_1) \frac{\sigma_1^2}{R_1} + e^{-\lambda}\beta
$$
which recall our initial guess.


### Information theory 

An alternative objective function which is usually maximized is the mutual information between the stimulus and the response
$$
I(r,x) = \int dr dx p(r,x) \log{\frac{p(r,x)}{p(r)p(x)}}
$$
This quantity, again, is very difficult to compute analytically. One of the most famous approximations rely on the existence of an idela estimator which satisfy the CRAO bound, and set the bound
$$
I(r,x) = H(x) - H(r|x) \geq H(x) - \int dx p(x)\frac{1}{2}\log(\frac{2\pi e}{J(x)})
$$
############################################################################################################

_As a remark, both maximizing this quantity (with the full integral) or the average of $1/J$ will result in an extrmely small $\sigma$, since it does not include the case of 0 spikes and as soon as there is a limited smoothness of the tuning curves, lead to a very narrow $\sigma$ , less than the spacing between centers. _Therefore again a common approach is maximizing the FI. A similar computations from Wei and Stocker showed a correction to this approach. Assuming an ideal estimator corrupted by additive noise:_
$$
\hat{x} = f(x) + z
$$
_they showed that the mutual information can be written as_
$$
I(\hat{x},x) = H(x) -\int dx p(x)\frac{1}{2}\log(\frac{2\pi e}{J(x)}) + C_0 -D_0
$$
_where $C_0$ represent a correction term monotonic in the noise lagnitude and $D_0 =  H(z) - \frac{1}{2}log(\frac{2\pi e}{J[z]})$ stay for non gaussianity of the noise. Also Zhang et al proposed a correction which involve the prior._

##############################################################################################################

We can write an  exact expression for the MI as
$$
I(r,x) = H(x) + \int dr dx p(r)p(x|r) \log(p(x|r))
$$
The second par is what is often called Distortion in rate-distortion theory. Again, using the defintion ad hoc to prevent divergence in the variance of the guassian
$$
p(x|r) = 
\begin{cases}
\mathcal{N}(\hat{x},\sigma^2/R) \qquad &\textit{if } R>0 \\
p(x) &\textit{if } R=0
\end{cases}
$$
we obtain for the MI
$$
D =  \frac{1}{2}\sum_{R>1} p(R) \log(\frac{2\pi\sigma^2 e}{R}) + e^{-\lambda}H(x)
$$
which has the desiredable property that goes to the entropy of the stimulus when the firing rate is too low. This approach can be extendend, for example to consider mulitple population of neurons, giving rise to a formulua very similar to the one for the error (with the addition of the log).

Considering the more general case, the decoder in case of uniform firing rate, is given by a product of experts (Hinton 02)
$$
log(d(x|r)) = -\sum_j \frac{(x-c_j)^2}{2\sigma_j^2/r_j} + log(Z)\\
d(x|r,\{\theta_j\}) = \frac{\Pi_j f(x|r_j,\{\theta_j\})}{Z} \qquad \textit{with } Z = \int dx \Pi_j f(x|r_j,\{\theta_j\})
$$
Note that given a noisy response, the individual experts are gaussians (witha scaled variance depending on the number of spikes) or uniform distributions.  This is exactly the formof the decoder assumed by Simoncelli, where m neurons compute the posterior for a given preferred position as: 
$$
p(x_m|r) \propto \exp{(\sum_n r_n \log(h_n(x_m)))}
$$
and then normalize.

In the limit where each neuron can emit 0-1 spikes: $p(r|x) = \Pi_j(1-e^{-u_j(x)})^{r_j}(e^{-u_j(x)})^{1-r_j}$, the decoder distribution is a mixture of uniform and gaussians.
$$
f_j(x|\{\theta_j \}) \sim  
\begin{cases}
\mathcal{N}(x,\{ \theta_j \})      &\textit{ with prob }  1-e^{-u_j(x)}\\
\mathcal{U}(x_{min},x_{max})   &\textit{ with prob }  e^{-u_j(x)}
\end{cases}
$$

The parameters of the tuning curves determine both the gaussian distribution and the mixture coefficients. Therefore we can define a proper optimization problem on the parameters of the gaussian distribution:
$$
\min _{\theta_m} D = \int dx p(x) log(d(x|\{\theta_m \}))
$$
We can therefore try to do gradient descent to find the minimum of this quantity. The idea is that x is encoded stochastically by N neurons with 0/1 spikes, and then a decoder output the estimate of the stimulus according to the pattern of activity, as a product of experts. 

This formulation is quite close to the idea of Woodford, where instead the x was encoded stochastically but only one expert was selected at each time, since the model of the data was a Mixture of gaussians.

#### Algorithmical minimization of the distortion through SGD

The problems lie now in computing the optimal value for the parameter vector $\{\theta_j\}$ such that (34) is minimized, where the parameters determine both the mixture coefficients and the gaussian distribution.  Taking the gradient we obtain:
$$
\partial_{j} D = \int dx p(x) \Big(\partial_j\log(f_j{(x}) -  \langle \partial_j\log(f_j(x)) \rangle_{x|\{\theta_j\}} \Big)
$$

#### The problem lies now in computing the average of the partition function: this can be done using Gibbs sampling or, more efficiently, using COnstrastive Divergence.

Ok this is wrong. We cannot take inside the average.

# Incomplete ideas

## Perturbative approach to FI/Error

The error quantity has the problem that minimizing with respect to $\sigma$, we obtain a degenerate solution $\sigma -> 0$ and an error going to 0. This is due to the fact that we assumed a continuum limit in the number of neurons. Numerical computations made by Bethge and al. showed how actually the optimal $\sigma $ scale as $\sim \frac{1}{N}$ . Knowing this, we can write the error in the following for
$$
\varepsilon^2 = \langle \frac{1}{J_\infty - \delta J(x)}\rangle \simeq \frac{\sigma}{NA} + f_e(\sigma,N,A)\\
= \frac{\sigma}{NA} (1 + \frac{\sigma}{NA}f_j(\sigma,N,A))
$$
where $f(\sigma,N,A)$ is the difference between the real MSE and the approximation with $J_\infty$. Ideally we could write this quantity at 2nd order, but what we obtain is:
$$
\langle\frac{1}{J_m + \delta J(x)}\rangle \simeq \frac{1}{J_m}(1 + \langle\frac{\delta J^2(x)}{J_m^2}\rangle)\\
\textit{where}\\
\langle\delta J^2(x)\rangle = \int dx A^2\sum_{i,i'}\frac{(x-c_i)^2(x-c_{i'})^2}{\sigma^8}exp(-\frac{(x-c_i)^2 + (x-c_{i'}^2)^2}{2\sigma^2})  - J_m^2 =\\
= A^2 \frac{\sqrt{\pi}}{16\sigma^7}\sum_{ii'} exp(-\frac{(c_i-c_{i'})^2}{4\sigma^2})[(c_i-c_{i'})^4 - 4(c_i-c_{i'})^2\sigma^2 +12\sigma^4]  -J_m^2\\
\textit{simplifying we obtain}\\
\varepsilon \simeq \frac{S(\sigma,N)}{J_m^3}
$$
where $S(\sigma,N)$ is the finite sum. 
In finding the minimum, we have to solve the following equation (fixing for simplicity $A=1$)
$$
\frac{\sigma^2}{N^3}S(\sigma,N) + \frac{\sigma^3}{N^3}\frac{\partial S}{\partial \sigma} = 0\\
\frac{\partial S}{\partial \sigma}/S = -\frac{1}{\sigma}
$$


More generally, we can think to $f(\sigma,N,A)$ as a 'cost' on making too narrow tuning curves. 

Without knowing something about the specific form of the function, is difficult to make comparisons.  



Q: how to extend them to consider two subpopulations with different tuning curves?
In the case where we have two subpopulations of neurons and we consider the sum of the fisher information
$$
\varepsilon^2 = \frac{\sigma_1\sigma_2}{N_1\sigma_2 + N_2\sigma_1} + f(\sigma_1,\sigma_2,N_1,N_2)
$$
USing the quadratic expansion, indicating with $S(\sigma,N)$ the finite sum, and considering $J_m = J_1 + J_2$, we obtain 
$$
\varepsilon \simeq \frac{S_1+S_2 + S(\sigma_1,\sigma_2,N_1,N_2)}{J_m^3}
$$




Q: it exist a biologically motivated shape of the function $f$ such that we have a solution $\sigma_1 \neq \sigma_2$?

An alternative formulation is considering the correction to the FI, not to the error:
$$
<\frac{1}{J(x)}> = <\frac{1}{J^1_\infty + J^2_\infty + f(x,\sigma_1,N_1) + f(x,\sigma_2,N_2)}>=\\
\frac{\sigma_1\sigma_2}{N_1\sigma_2+N_2\sigma_1} (1 + \frac{\sigma_1\sigma_2}{N_1\sigma_2+N_2\sigma_1}(f(\sigma_1,N_1) + f(\sigma_2,N_2)))
$$

## Rate distortion theory and variational prior

Intuition (general): think to the R-D plane and that targeting a certain R, we can always reduce the D allowing for  more $\sigma$ 

The non trivial $\sigma$ should appear also considering the maximization of the MI Unluckily the computation become hard at a certain point. Let's consider first the defintion of MI
$$
I(r,x) = \int dx p(x,r) \log(\frac{p(r,x)}{p(r)p(x)})
$$
This quantity is difficult to treat: analytically the marginal is not computable, while numerically it require an integration over an high dimensional space which is difficult to do as soon as we don't have few neurons.Usually, this quantity is treated using the approximation of Nadal. But let's consider first this way of writing:
$$
I (r,x) = \int dx p(x)\sum_{\{r\}} p(r|x) log(p(r|x)) = \langle KL(p(r|x)||p(r))\rangle_x
$$
This is the average dkl between the conditional distribution imposed by the population and the marginal over stimuli. The marginal is the difficult part to compute, since has not a close form in general:
$$
p(r) = \int dx p(x)p(r|x)
$$


An alternative is considering a factorized uniformative  variational prior, where we assumed that each neuron emits a spike with a firing rate equal for all neurons:
$$
q(r) = \Pi_j \frac{(\lambda'/N)^{r_j} exp(-\lambda'/N)}{r_j!}
$$
Now, the KL divergence is an upper bound to the true mutual information can be written as
$$
KL = \sum_{\{r\}} \Pi_j \frac{e^{-u_j(x)}u_j(x)^{r_j}}{r_j!} \log(e^{-(\lambda-\lambda')}\Pi_j((\frac{N u_j(x)}{\lambda'})^{r_j}) \\
= -\lambda + \lambda' + \sum_j \sum_{\{r_j\}} \frac{e^{-u_j(x)}u_j(x)^{r_j}}{r_j!}r_j\log(N\frac{u_j(x)}{\lambda'})\\
= \lambda' - (1+\log(\lambda'/N))\lambda + \sum_j u_j(x)\log(u_j(x))
$$
where we defined the total number of spikes $\sum_ju_j(x)= \lambda$. This quantity in principle is dependent from x, but if the coding scheme is 'well designed' it shouldn't.  The MI become
$$
I(r,x) = \int dx p(x) KL = \lambda' - (1+\log(\lambda'/N))\lambda  - \int dx p(x)\exp(-\frac{(x-c_j)^2}{2\sigma^2})\frac{(x-c_j)^2}{2\sigma^2}\\
= \lambda' -(\frac{3}{2} + log(\frac{\lambda'}{N}))\sqrt{2\pi}N\sigma
$$
where we substitued $\int dx u_j(x) = \sqrt{2\pi\sigma^2}$  . (Make sense?) . Even with this variational prior, the MI (first term) is related to.

A common loss function in unsupervised learning that try to find a balance between the necessity of an accurate representation and generalization requirements is the Information Bottleneck, which in turns is connected to ELBO, a loss function used in autoencoder. This can be mapped to the problem  of neural coding asking what is the optimal neural representation $\mathbf{r}$ such that we minimize the reconstruction error $\hat{x}(r)$ and but still we don't encode a trivial function. There are multiple, conected ways of formulating the problem. One possibility is considering the mutual information between the stimulus and the response
$$
I(r,x) = \int dx p(x) p(x,r) \log(\frac{p(x,r)}{p(x)p(r)})
$$
where $p(r|x)$ is the encoding probability given by the encoder function with noisy neurons, in the case of poisson neurons with gaussian tuning curves: $p(r|x) = \Pi_j \frac{u_j(x)^{r_j}e^{-u_j(x)}}{r_j!}$

We know the following variational lower and upper bounds:
$$
H -D \leq I \leq R\\
H = -\int dx p(x) \log(p(x))\\
D = - \int dx p(x)\int dr p(r|x) \log(q(x|r))\\
R = \int dx p(x)\int dr p(r|x) \log{\frac{p(r|x)}{\tilde{p}(r)}}
$$
where $q(x|r)$ is the variational approximation of the decoder and $\tilde{p}(r)$ is the variational approximation of the marginal of the response. 
A common approach is consider the estimator as gaussianly distributed around the true value $q(x|r) = \frac{1}{\sqrt{2\pi d(x)}} \exp(-\frac{(\hat{x}(r)-x)^2}{2d(x)^2})$ and obtaining for the distortion term:
$$
D = \int dx p(x)\frac{\varepsilon^2(x)}{2 d(x)} - \int dx p(x) \log(d(x))
$$
note that if the decoder is optmized, it should achieve the Cramer-Rao bound with $d(x) =\frac{1}{J(x)}$ and we re obtain the lower bound to the mutual information of Nadal.
The problem of this approach is that it does not capture the case of 0 spikes, so is totally independend from the number of spikes. This, in case of very low firing rate, lead to a very high distortion. Nevertheless, we could try to use the $ D+\beta R$ loss function (?) using 
$$
D = -\int dr dx p(r,x)\log(d(x|r)) = -\int dx p(x) log(\frac{J(x)}{2\pi e})
$$




## Single neuron MI

One idea to constrain the width of tuning curves could be penalize too high mutual information of a single neuron :

$\sum_i I(r_i,x)$. Nevertheless, using the fisher approxiamtion result in a divergent integral:
$$
I(r_i,x) = H(x) - \int dx p(x)\log(\frac{2\pi e}{J(x)}) = c + \int dx p(x)\log[\frac{(x-c_i)^2}{\sigma^4}exp(-(\frac{(x-c_i)^2}{2\sigma^2})]
$$
where the divergence come from the exponential.Redefining $x=x-c_i$ (assuming the stimulus space and t.c are translational invariant, we obtain for the second term
$$
2\int dx p(x) (log(x) - \frac{(x-c_i)^2}{2\sigma^2}) - 4log(\sigma)
$$
if the stimulus space is finite, the first part is inversely proportional to $\sigma$, therefore when $\sigma -> 0$, the I goes to -infinity, and it is not counterbalanced by the log-term. Anotehr approach would be consider the posterior implied by an ideal decoder
$$
d_j(x|r_j) = \cases{
\mathcal{N}(c_j,\sigma_j^2/r_j)  \qquad \textit{prob } p_j\\
U[0,1]                     \qquad \textit{prob } e^{-u_j(x)}
}
$$
Thus the MI become
$$
I(r_j,x) \approx H(x) - \langle e^{-u_j(x)}\rangle_x H(x) -\sigma_j^2 \langle e^{-u_j(x)}[Ei(u_j(x)) - \gamma - \log(u_j(x))]\rangle_x
$$
Considering all these computations, we realize that the MI of a single neuron is 0 when the tuning width is extremely small and grows, reaching a maximum at a broad width value. This results are confirmed by numerical simulations

##  Interesting Hints from PPC (works of Ma, Haefner, etc...)

" The so called Probabilistic Population Coding (or PPC) framework[8, 9, 10] takes this link seriously
by proposing that the function encoded by a pattern of neural activity r is, in fact, the likelihood
function p(r|s). When this is the case, the precise form of the neural variability informs the nature
of the neural code. For example, the exponential family of statistical models with linear sufficient
statistics has been shown to be flexible enough to model the first and second order statistics of in vivo
recordings in awake behaving monkeys[9, 11, 12] and anesthetized cats[13]. When the likelihood
function is modeled in this way, the log posterior probability over the stimulus is linearly encoded by
neural activity, i.e.
$$
p(s|r) \propto g(s) \exp{(h(s)^Tr)}
$$
Note that in the case of iid neurons with gaussian tuning curves , poisson noise, constant sum of firing rates,

the kernel is directly related to the the log of tunign curves $h_i(s) = log(f_i(s))$ . In a more general setting
$$
h'(s) = \Sigma^{-1} f'(s)
$$
This imply 