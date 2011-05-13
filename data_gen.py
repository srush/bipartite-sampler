import itertools
import copy
import random

random.seed(0)
data = {"A": "A", "B": "B"}
types = {"A":0,"B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10 }

rev_types = dict([(v,k) for k, v in types.iteritems()])



def real_dist():
  dist = {}
  n = len(types)
  for k in types.keys():
    def pick(i):
      if i ==types[k]: return 0.8
      if i ==types[k]+1 % n: return 0.2
      return 1e-10
    dist[k] = [pick(i) for i in range(n)]
  return dist

def generate_data():

  instances = []
  for q, comb in enumerate(itertools.combinations(range(11), 6)):
    if q > 100: break
    d = []
    
    for j in comb:
        
      d.append(types.keys()[j])
      # if random.random() > 0.5:
      #   d.append("A")
      # else:
      #   d.append("B")
    f = copy.copy(d)
    random.shuffle(f)
    for i in range(len(f)):
      if random.random() > 0.8:
        f[i] = rev_types[(types[f[i]] + 1) % 11]
      
    instances.append((d,f))
  return instances

