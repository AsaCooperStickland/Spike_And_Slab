# Spike and Slab model 

Python code for my masters project. I've got a dataset of peptoids (a peptide mimicking molecule), and how effective they are against a range of different 
microbes. I'm using linear regression with sparsity-inducing priors (spike and slab ones to be precise) to figure out which features of the peptoids make them
good drugs. We want to use information from the microbes where we have a lot of data to help prediction in microbes where we don't have much data. And we want
an overall 'relevance score' for the features that's independent of a particular microbe. One way to get those things is to use 'Spike and slab' priors. 

A more technical/complete description can be found in these papers by Hernández-Lobato et. al, but I'll give a basic intro. Let's start with linear regression:

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/linreg.gif)

Epsilon is Gaussian noise we add on to each y value. In a Bayesian setting we put a prior on the weights w. The simplest prior is just Gaussian, but 
I'm doing something a bit (okay, maybe a lot!) more complicated, our prior looks like this: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/s_n_s.gif)

We've introduced latent variables z. These zs are either 1, in which case we revert back to a Gaussian prior, or 0, in which case our prior is a 'Dirac delta'
centred on 0, in other words we have 100% probability that those weights are zero, and our model totally ignores them. This is another way of acheiving sparsity, 
with the most famous way being L1 regularisation. A nice thing about the spike and slab prior, other than it being fully Bayesian and giving us access to a 
posterior, is that thet posteriror of z gives a probabiltiy that a certain feature is relevant. This is very useful for my project, because I want to figure out
which features make a difference to drug effectiveness! 

We also have a prior on z which looks like: 

![Five Adam runs](https://github.com/AsaCooperStickland/Spike_And_Slab/blob/master/figures/zs.gif)

I.e. a Bernouilli probability distribution. p0 is the a priori proportion of features we think are relevant, so p0 = 0.5 means we think about half of the 
features will end up being used. The posterior of p0 gives us how many features are model thinks are relevant. 

# Toy Dataset

I'm not putting the real dataset on Github since it's not mine! But we can demonstrate the principal using a toy dataset. 



