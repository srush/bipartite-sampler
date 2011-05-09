import time
from main import *
import numpy as np
import numpy.random as nprandom
import random
random.seed(2)

def gen_mat(n, dense):
  """
  Generate dense binary matrices
  """
  z = np.zeros((n,n))
  for row in range(n):
    r = []
    total = 0
    for col in range(n):
      if z[row,col] == 0:
        r.append(col)
      else:
        total +=1
    random.shuffle(r)
    need = int((n* dense) - total )
    for col in r[:need]:
      z[row, col] = 1

  for col in range(n):
    r = []
    total = 0
    for row in range(n):
      if z[row,col] == 0:
        r.append(row)
      else:
        total +=1
    random.shuffle(r)
    need = int((n* dense) - total )
    for row in r[:need]:
      z[row, col] = 1
  return z
  
RANDOMZO = 1
RANDOM = 2

mode = RANDOM
def main():
  for size in range(2,30):
    if mode == RANDOMZO:
      m = gen_mat(size, 0.8)
    else:
      m = nprandom.random((size,size))
      #print m
    #print m
    #print m

    # start1 = time.time()
    # r_perm = ryser(m)
    # end1  = time.time()

    #print "Exact Time:", end - start
    #print r_perm
    
    #print m
    start2 = time.time()
    samp_perm, samples =  PermanentSampler(0.5,0.1, mode == RANDOMZO).estimate_permanent(m)
    end2 = time.time()
    #print "Samp Time:", end - start
    #print samp_perm

    # start3 = time.time()
    # n_perm,_ = 0.0, []#naive_permanent(m)
    # end3  = time.time()

    #print "END %s %3f %3f %3f %3f %3f %3f %3f"%(size, end1-start1, end2-start2, end3-start3, r_perm, n_perm, samp_perm, r_perm/samp_perm) 
    print "END %s %3f "%(size, end2-start2)
  
  #n_perm = naive_permanent(m)
  
  #print n_perm

if __name__== "__main__":
  main()
