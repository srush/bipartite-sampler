import numpy as np
import numpy.random as rand
import random
import math
rand.seed(0)

def kl(a,b):
  if a < 1e-4: return 0.0
  if b < 1e-4: return 0.0
  else: return a * np.log(a / b )
v_kl = np.vectorize(kl)

def kl_div(dist_a,dist_b):
  return (v_kl(dist_a, dist_b)).sum()
    
def shift(v):
  return v + 1e-10
v_shift = np.vectorize(shift)
def kl_tables(table_a, table_b):
  return -sum([kl_div(table_a[key],table_b[key]) for key in table_a.keys()])

types = {"A":0,"B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10 }


def normalize(dist):
  k = np.sum(dist)
  return dist / k if k > 0 else dist

def rand_dist(over):
  n = len(over)
  return normalize(rand.random(n) + 10*np.ones(n))

def rand_dists(over):
  return dict([(k, rand_dist(over)) for k in over.keys()])


def maximum_likelihood(counts):
  dists = {}
  for k in counts.keys():
    n = counts[k].sum()
    dists[k] = counts[k] / float(n)
  return dists

def add(a,b):
  ret ={}
  for k,v in a.iteritems():
    ret[k] = v
  for k,v in b.iteritems():
    if ret.has_key(k):
      ret[k] += v 
    else:
      ret[k] = v
  return ret 




def gen_mat(dists, instance):
  us, them = instance
  n = len(us)
  A = np.zeros((n,n))
  tsym_c = {}
  for tsym in them:
    tsym_c.setdefault(tsym,0)
    tsym_c[tsym] +=1
  
  for i, sym in enumerate(us):
    for j, tsym in enumerate(them):      
      A[i,j] = dists[sym][types[tsym]] / float(tsym_c[tsym])
  return A
