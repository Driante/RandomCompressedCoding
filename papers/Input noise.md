## Input noise

Given the responses of the second layer $v_i(x) = \frac{1}{Z}\sum_{j}w_{ij}u_j(x)$  with $w_{ij} \sim \mathcal{N}(0,\frac{1}{L})$ and $u_j(x) = \exp({-(x-c_j)^2/2\sigma^2})$ . 
$$
r = \frac{1}{Z}\sum_j w_{ij }(u_{j}(x) + \xi_j)  + \eta_i = \frac{1}{Z}\sum_jw_{ij}u_j(x) + \sum_j w_{ij}\frac{\xi_j}{Z} +\eta_i
$$

To eliminate the factor Z in the output noise, is sufficient redefine $\tilde u (x) = \frac{u}{Z}$ , then the noise is no more multiplied by Z and we simply have $\xi^2$

If the input layer is affected by iid gaussian noise of variance $\xi^2$  and the output layer by noise of $\sigma_\eta^2$, the noise covariance matrix is 
$$
\Sigma = \sigma_\eta^2 I + \frac{\xi^2}{Z^2} WW^T
$$
we kept fixed the ratio $\frac{\xi^2}{Z^2}$ to avoid effect of magnifying the noise for different $\sigma$ . We would like to show that correlations induced by this model have a small effect on the coding properties. The local error for the ouput layer is given by the inverse of the fisher information
$$
J(x) = v'(x)^T\Sigma^{-1}v'(x)
$$
Computing the inverse of the covariance matrix is hard. We can do a perturbative expansion using the fact that the Wishart matrix $WW^T$ has mean $I$ and variance of the terms  proportional to $\frac{1}{L}$ . Assuming that L is large enough we can write 
$$
\Sigma = \tilde{\sigma}^2_{\eta}I + \delta \Sigma
$$
where $\tilde{\sigma}_\eta^2 = \sigma_\eta^2 + \frac{\xi^2}{Z^2}$ and $\delta \Sigma = \frac{\xi^2}{Z^2}(WW^T -I)$ . The inverse of this matrix can be approximated as 
$$
\Sigma^{-1} \approx \frac{1}{\tilde{\sigma}_\eta^2} I - \frac{1}{\tilde\sigma_\eta^4}\delta\Sigma
$$
which result in an approximation for the FI
$$
J(x) = J^{ind}(x) -\frac{\xi^2}{Z^4\tilde{\sigma}_\eta^4} u'^T(x)(W^TWW^TW - W^TW)u'(x)
$$
Also $W^TW$ follow a wishart distribution, but has rank N, mean $N/L *I$ and variance of the terms proportional to $N/L^2$

Denoting by $A= W^TW$ the matrix, we can vcompute the mean and variance of the entries (Wick theorem) $A_{ij} = \sum W_{j'j}W_{j'i}$ . 
$$
E[A_{ij}] = \delta_{ij}\frac{N}{L}
$$

$$
E[A_{ij}A_{lk}] = E[\sum_{j'j"}W_{j'i}W_{j'j}W_{j"l}W_{j"k}] = \frac{N^2}{L^2}\delta_{ij}\delta_{lk} + \frac{N}{L^2}(\delta_{il}\delta_{jk} + \delta_{ik}\delta_{jl})
$$
a particular case of Eq.(7 ) is useful to compute the mean entries of $A^2$, since $(A^2)_{mn} = \sum_{j=1}^L A_{mj}A_{jn}$ 
The expect value of this entries is given by 
$$
E[A_{mj}A_{jn}] = (\frac{N^2}{L^2} + \frac{N}{L^2})\delta_{mj}\delta_{jn} + \frac{N}{L^2}\delta_{mn}
$$
summing over j we obtain the mean value of 
$$
E[(A^2)_{mn}] = (\frac{N}{L} + \frac{N^2}{L^2} + \frac{N}{L^2})\delta_{mn}
$$
Therefore the mean of the perturbation is given by 
$$
E[A^2-A] = (\frac{N^2}{L^2} + \frac{N}{L^2})I
$$
With the same technique it should be possible to compute also the variance of the terms of the square matrix. Let's compute the expected value of
$$
E[((A^2-A)_{mn})^2] = E[\sum_j A_{mj}A_{jn}A_{mj'}A_{j'n}] + E[(A_{mn})^2] -2E[(\sum_{j}A_{mj}A_{jn})A_{mn}]
$$
The second term is easy, using (7)
$$
E[A_{mn}A_{mn}] = (\frac{N^2}{L^2}+\frac{N}{L^2})\delta_{mn} + \frac{N}{L^2}
$$
With some pain we can compute the last term, since
$$
E[\sum_{j=1}^L\sum_{j'j"j'''}^N W_{j'm}W_{j'j}W_{j"j}W_{j"n}W_{j'''m}W_{j'''n}] = (\frac{N^3}{L^3} + 3\frac{N^2}{L^3} + \frac{N^2}{L^2} + \frac{4N}{L^3} +\frac{N}{L^2})\delta_{mn} + 2\frac{N^2}{L^3} + \frac{N}{L^2}
$$


the first term requires 
$$
E[((A^2)_{mn})^2] = \sum_{jj'}^LE[A_{mj}A_{jn}A_{mj'}A_{j'n}] = \sum_{jj'}\sum_{j"j'''j^{iv}j^v} ^N W_{j"m}W_{j"j}W_{j'''j}W_{j'''n}W_{j^{iv}m}W_{j^{iv}j'}W_{j^{v}j'}W_{j^{v}n}
$$
Note: a good strategy to compute this is ordering terms for N powers, determining how many terms remain in the summation over the 4 indices, and it depends how we form the couples.

We have 1 term of order $N^4$ which arise coupling each first index, and return $ \frac{N^4}{L^4} \delta_{mn}$.

We have 12 terms of order $N^3$ arising coupling two different indices  among them like $j"-j'''$ and leaving the others couples $(6\frac{N^3}{L^4} + 2\frac{N^3}{L^3})\delta_{mn} + 4\frac{N^3}{L^4}$

We have 12 + 32 terms of order $N^2$ $(\frac{N^2}{L^2} + 3\frac{N^2}{L^3} + 5\frac{N^2}{L^4})\delta_{mn} + 3\frac{N^2}{L^4}$  for the one formed by couples and then the one formed by picking one couple and varying others pairing



48 terms of order N which are not important since they scale like $N/L^3$ at most



### Just using the mean of the perturbation

Using the fact that $\sum_j (u'_j(x))^2 \approx L\frac{\sqrt\pi}{\sigma}$, the mean of perturbation in Eq.5 therefore can be computed and gives
$$
<\delta J(x)> = \frac{\xi^2}{Z^2 \tilde\sigma^4}(\frac{N^2}{L} + \frac{N}{L})\frac{\sqrt{\pi}}{\sigma}
$$
The fisher info therefore can be approximated as
$$
J^c \approx J^{i} - \delta J = \frac{N\sqrt\pi}{Z^2\tilde\sigma^2 \sigma} - \frac{\xi^2}{Z^4 \tilde\sigma^4}(\frac{N^2}{L} + \frac{N}{L})\frac{\sqrt{\pi}}{\sigma}
$$
It is interesting to see the difference between this special types of correlation and the one induced by a random matrix with the same statistical structure, but uncorrelated with the synaptic weights. Instead of W, we can consider another wishart matrix but not made by the same W. For example, consider $B = XX^T$ where X is a matrix whose column are random gaussian vectors of variance $\frac{1}{L}$ . We can make the same considerations and arrive to rewrite (5)
$$
J(x) = J^{ind}(x) -\frac{\xi^2}{Z^2\tilde{\sigma}_\eta^4} u'^T(x)(W^TXX^TW - W^TW)u'(x)
$$
In this case one should compute the mean of the matrix
$$
B' = W^TXX^TW
$$
with the same wick theorem we obtain
$$
<B> = \frac{N}{L}\delta_{mn}
$$
and the perturbation has therefore 0 mean.



It is possible go also to further order of expansion, useful to understand  the effect of a random covariance matrix. (4) can be extended as 
$$
\Sigma^{-1} \approx \frac{1}{\tilde{\sigma}_\eta^2} I - \frac{1}{\tilde\sigma_\eta^4}\delta\Sigma + \frac{1}{\tilde\sigma_\eta^6} \delta \Sigma^2
$$
and inserting in the fisher info we obtain for the second order term
$$
W^T(W^TW - I)^2W = A^3 - 2 A^2 + A
$$
the entries of $A^3 = \sum_{jj'}^L\sum_{ii'i"^N}W_{im}W_{ij'}W_{i'j'}W_{i'j}W_{i"j}W_{i"n} = (\frac{N^3}{L^3} + 3\frac{N^2}{L^3} + 3\frac{N^2}{L^2} + 4\frac{N}{L^3} + 3\frac{N}{L^2} + \frac{N}{L} )\delta_{mn}$  and therefore the  mean of second order term is 
$$
<\delta J(x)^2> = \frac{\xi^4}{Z^6 \tilde\sigma_\eta^6}(\frac{N^2}{L^2} + \frac{N^3}{L^3} + 3\frac{N^2}{L^3}+ \frac{N}{L^2} + 4\frac{N}{L^3}) \delta_{mn}
$$
In the case of uncorrelated noise covariance matrix,  the second order term gives also a positive effect  resulting in 
$$
<\delta J(x)^2> = \frac{\xi^4}{Z^6 \tilde\sigma_\eta^6}(\frac{N^2}{L^2} + \frac{N}{L^3}) \delta_{mn}
$$

The fisher information correction therefore is, for the synaptic-noise covariance matrix (using just the higher order terms for each correction)
$$
<J(x)>_W = J^i (1-\frac{\xi^2}{Z^2\tilde\sigma^2}\frac{N}{L} + \frac{\xi^4}{Z^4\tilde\sigma^4}\frac{N}{L} - ...)
$$

$$
\frac{\xi^2}{Z^2\tilde\sigma^2} = \frac{1}{1+\frac{Z^2}{\xi^2}\sigma_\eta^2}
$$



while, in the case of random covariance matrix
$$
<J(x)>_W = J^i (1 + \frac{\xi^4}{Z^4\tilde\sigma^4}\frac{N}{L} - ...)
$$
