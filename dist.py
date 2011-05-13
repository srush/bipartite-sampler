
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

types = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10 }


def normalize(dist):
  k = np.sum(dist)
  return dist / k if k > 0 else dist

def normalize_sparse(dist):
  norm = sum([ v for v in dist.itervalues()])
  return dict((k, v/norm) for k,v in dist.iteritems()) if norm > 0.0 else dist 

def rand_dist(over):
  n = len(over)
  return normalize(rand.random(n) + 1*np.ones(n))

def rand_dist_sparse(over):
  return {}


def maximum_likelihood_add_n(counts, extra =0.5):
  dists = {}
  for k in counts.keys():
    n = counts[k].sum()
    dists[k] = (counts[k] + extra) / (float(n) + 100*extra)
  return dists


def maximum_likelihood(counts):
  dists = {}
  for k in counts.keys():
    n = counts[k].sum()
    dists[k] = counts[k] / float(n+1)
  return dists

def maximum_likelihood_sparse(counts):
  dists = {}
  for k in counts.keys():
    norm = sum( v for v in counts[k].itervalues()) 
    dists[k] = dict((k, v/(norm+1)) for k,v in counts[k].iteritems())
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


def add_sparse(a,b):
  ret ={}
  for k,v in a.iteritems():
    ret[k] = v
  for k,v in b.iteritems():
    if ret.has_key(k):
      ret[k] = dict((k2, a[k].get(k2, 0.0) +b[k].get(k2,0.0))for k2 in (set(a[k].keys())| set(b[k].keys())))
    else:
      ret[k] = v
  return ret 


class ProbModel:
  def __init__(self, f_types, e_types):
    self.f_types = f_types
    self.e_types = e_types

  def rand_dists(self):
    return dict([(k, rand_dist(self.e_types)) for k in self.f_types.keys()])

  def rand_dists_sparse(self):
    return dict([(k, rand_dist_sparse(self.e_types)) for k in self.f_types.keys()])

  def types(self):
    return self.f_types, self.e_types

  def prob(self, dists, f_type, e_type):
    if dists.has_key(f_type):
      if self.e_types.has_key(e_type):
        #return dists[f_type][self.e_types[e_type]]
        return dists[f_type].get(self.e_types[e_type], 1e-10)
      else:
        return 1e-10
    else:
      return 1e-10
    
  def gen_mat(self, dists, instance):
    us, them = instance.split()
    n = len(us)
    A = np.zeros((n,n))
    tsym_c = {}
    for tsym in them:
      tsym_c.setdefault(tsym,0)
      tsym_c[tsym] +=1
    for i, sym in enumerate(us):
      for j, tsym in enumerate(them):      
        A[i,j] = self.prob(dists,sym,tsym) / float(tsym_c[tsym])
    return A
