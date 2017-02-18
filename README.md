# Spike and Slab model 

Python code for my masters project. 

    spike_n_slab.py

Uses Gibbs sampling to get sample from posterior with 'spike and slab', or Gaussian priors. 

    tests.py

Does some data visualisation on a toy dataset, interpolating between errors with Gaussian processes and showing off some posteriors. Shows how spike and slab
is probably the best model when you have a particular situation of multiple target variables that depend on the same features: further outlined below. 

# Intro

I've got a dataset of peptoids (a peptide mimicking molecule), and how effective they are against a range of different 
microbes. I'm using linear regression with sparsity-inducing priors (spike and slab ones to be precise) to figure out which features of the peptoids make them
good drugs. We want to use information from the microbes where we have a lot of data to help prediction in microbes where we don't have much data. And we want
an overall 'relevance score' for the features that's independent of a particular microbe. One way to get those things is to use 'Spike and slab' priors. 

A more technical/complete description can be found in [these](http://www.jmlr.org/papers/volume14/hernandez-lobato13a/hernandez-lobato13a.pdf) 
[papers](http://link.springer.com/article/10.1007/s10994-014-5475-7) by Hernández-Lobato et. al, but I'll give a basic intro. Let's start with linear regression:

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/linreg.gif)

y is a vector with our target values in, and X is a matrix with the features associated with each target value. Epsilon is Gaussian noise we add on to data point. 
In a Bayesian setting we put a prior on the weights w. The simplest prior is just Gaussian, but 
I'm doing something a bit (okay, maybe a lot!) more complicated, our prior looks like this: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/s_n_s.gif)

We've introduced latent variables z. These zs are either 1, in which case we revert back to a Gaussian prior, or 0, in which case our prior is a 'Dirac delta'
centered on 0, in other words we have 100% probability that those weights are zero, and our model totally ignores them. This is another way of achieving sparsity, 
with the most famous way being L1 regularisation. A nice thing about the spike and slab prior, other than it being fully Bayesian and giving us access to a 
posterior, is that that the posterior of z gives a probability that a certain feature is relevant. This is very useful for my project, because I want to figure out
which features make a difference to drug effectiveness! 

We also have a prior on z which looks like: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/zs.gif)

I.e. a Bernouilli probability distribution. p0 is the a priori proportion of features we think are relevant, so p0 = 0.5 means we think about half of the 
features will end up being used. The posterior of p0 gives us how many features are model thinks are relevant. 

Notice back in the prior on the weights we had vectors in the Dirac delta and Gaussian distributions, not scalars. This is because we have 'group sparsity'. 
We group together the weights for different microbes, for example charge for E.coli and charge for some other bacteria would be jointly zero or Gaussian. 
In this way we can 'share information' between microbes. If one microbe has loads of data points and it's obvious a certain feature isn't useful for prediction, 
that feature won't get used on the other microbes too. And the posterior for z will give us the probability that a certain feature is relevant across every 
microbe, not just one. 

The way we formulate this in linear regression is to have a setup that looks like this: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/formulation.gif)

The 1,2,...,m here are different target variables, so in my project they're the different microbes. In traditional linear regression this would mean the weights 
against each microbe would be completely independent (even if they were all weights for e.g. charge). But spike and slab allows them to be grouped together. 
Another thing you can do here is if you have some domain expertise and think it's likely that, say, number of red balloons, and number of green balloons will
jointly relevant or irrelevant to party enjoyment, you can put that into your prior. But I concentrate on the case where you have multiple target variables. 

This fancy prior means our posterior distribution is completely intractable, so we have to use MCMC (if you aren't familiar, [this](https://www.youtube.com/watch?v=Em6mQQy4wYA&t=2734s) 
is a great intro by Iain Murray of Edinburgh Uni.), specifically Gibbs sampling, since the conditional distributions of our variables are tractable. Details are 
in the appendix of [this](http://www.jmlr.org/papers/volume14/hernandez-lobato13a/hernandez-lobato13a.pdf) paper. 

# Toy Dataset

Okay, enough maths! We can demonstrate the principal using a toy dataset. The number of features is set as five, and we have four target variables, and our 
linear system looks like:

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/system.gif)

Notice that the weights are different for each target, but in each one only the 2nd and 5th features are actually contributing anything. For clarity I've just
included the terms that matter down below:

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/system_small.gif)

This is a pretty simple system, so with enough data points maximum likelihood will give us good results. But if the number of data points is small, the 'nuisance'
variables x0, x2, x3 are going to get some weight attached to them and our estimate will not generalise well to new data. This isn't solved by a regular 
Gaussian prior either. What spike and slab is really good at is utilising the fact that the same features are relevant in every problem. There's going to end up
being a very high probability that x1 and x4 are relevant, and the others are will have a high probability of being exactly zero, 
or equivalently the probability of z being one for x1 and x4 will be high (remember zs are shared between x1 and x4 for every y). 
Don't believe me? Look at some graphs: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/errors.png)

On the y axis is the mean error on 200 test data points, averaged over five runs. The x axis is on a log scale. Since it can share information between the four target variables, spike and slab is the clear winner
for a small number of data points, although all methods converge to the true error (0.01^2 = 0.0001) at around 40 data points. 
Interestingly, a Gaussian prior is at first pretty bad, losing to a MLE, although it eventually overtakes. This is probably because
our prior is not particularly close to the real weights, so when the model doesn't have much data it sticks pretty close to the prior, giving a rubbish 
test error. 40 data points is pretty small, so why bother with spike and slab? Well when you increase the number of features it performs even better against
MLE. In the papers I mention above you can have > 1000 features, and you really need a strong regularizer like a spike and slab prior. 

Below is the same but removing the Gaussian prior values, and fitting a Gaussian process to the error for the other two. Mainly just for fun, but since MCMC is pretty 
expensive in general, you might want the Gaussian process as a good interpolation between two of your sizes.  

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/errors_gp.png)

And we can look at histograms for some of our variables to see if the model can learn them from the data: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/p0.png)

Above are 9000 samples from the posterior for p0, or the percentage of features the model thinks are relevant (for 80 data points). 
The real value is 2/5, but this has a quite broad mode at around
0.6. I found this was typical: when I used one feature it landed on about 0.4. Perhaps it takes a lot of evidence to get p0 to be a low number. 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/sigbig.png)

This is 9000 samples from the posterior for error (again 80 data points). The distribution is pretty much centred on 0.0001 (We added Gaussian 
noise with variance 0.0001 to get the training data.) like we expect. 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/big_weights.png)

Finally here's something showing the spike and slab prior in action: These are the weights for x1 against y0, which has a true value of 0.1, indicated by the 
vertical red line. For a low number of data points, the posterior is very wide (note the scales changing on the x axis), and there's a significant number 
of samples on exactly zero- the 'spike' is most easily visible on the top right. As we get more data, the posterior shifts towards 0.1, stops hitting zero, 
and gets narrower. Pretty cool, huh? 

You can explore what the posterior looks like for certain weights, or for zs by looking in 

    spike_slab_results

Samples from the posterior look like weights(number of data points) or gauss_weight(number of data points) for spike and slab or Gaussian priors respectively. 
Just remember to throw out the first 1000 samples when plotting a posterior or finding a mean. 
