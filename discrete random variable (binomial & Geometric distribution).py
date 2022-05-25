#!/usr/bin/env python
# coding: utf-8

# In[4]:


#  No of possible probability of arranging number of set ( sample size= n, number of selection= r) with order respect, without repetitation
def factorial(n):
    if n==1: return 1
    else: return n * factorial(n-1)
# No of possible probability without order, without repetitation
def combination_no_rep (n,r):
      return(((factorial(n)/(factorial(r))*(factorial (n-r)))))
#r = number of trials
#p = probability of success in each trial
#x = number of success in n trials
def binomial_Probability(n, r, p):
  
    return (combination_no_rep(n, r) * pow(p, x) * 
                        pow(1 - p, n - x))
def variance_B (p):
    return (n * p * (1 - p))

#x= number of trials until first success
def geometrical_Probability(x,p):
    return (p * (pow(1 - p, x - 1)))

def variance_G (p):
    return ((1 - p) /p ** 2)


# In[ ]:




