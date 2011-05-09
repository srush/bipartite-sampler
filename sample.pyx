
# cython: profile=True
cimport sample
cimport cython
cdef extern from "math.h":
    double log(double theta)
import math

M_E = math.e
import numpy as np
import numpy.random as rand
cimport numpy as np
DTYPE =np.double
ctypedef np.double_t DTYPE_t
np.import_array()


@cython.boundscheck(False)

def bregmans_upper(double r):  
  if r >= 1:
    return r + 0.5 *log(r) + M_E - 1
  elif r >=0 and r < 1: 
    return 1  +(M_E -1) *r

# def upper_bound(np.ndarray A):
#   cdef np.ndarray [DTYPE_t, ndim=1] row_sums  = A.sum(axis=1)    
#   cdef double total = 1.0
#   cdef double local
#   for r in row_sums:   
#     if r >= 1:
#       local = r + 0.5 *log(r) + M_E - 1
#     elif r >=0 and r < 1: 
#       local = 1  +(M_E -1) *r
#     total *= local / M_E
#   #bvec = np.vectorize(temp, otypes=[np.double])
#   return total #(bvec(row_sums)/ M_E).prod() 

def rem(np.ndarray M, int i, int j):
  """
  Replace row i, column j with 0, set M[i,j] =1
  """
  cdef int n = len(M)
  cdef np.ndarray[DTYPE_t, ndim=2] D = M.copy()
  cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(n)
  D[i,:] = z
  D[:,j] = z
  D[i,j] = np.ones(1)
  return D

# @cython.boundscheck(False)
# def rem_row_sums_upper(np.ndarray [DTYPE_t, ndim=2] original_mat not None, 
#                        np.ndarray [DTYPE_t, ndim=1] row_sums not None, 
#                        int i, int j, int n, set picked_rows):
#   cdef np.ndarray[DTYPE_t, ndim=1] t = row_sums - original_mat[:,j]
#   cdef double total = 1.0
#   cdef double local
#   cdef double r
#   cdef int row
#   for row in picked_rows:
#     t[row] = 1
#   t[i] = 1

#   for r in t: 
#     if r >= 1:
#       local = r + 0.5 *log(r) + M_E - 1
#     elif r >=-1e-5 and r < 1: 
#       local = 1  +(M_E -1) *r
#     total *= local / M_E
#   return t, total

# @cython.boundscheck(False)
# def inner_sample(np.ndarray [DTYPE_t, ndim=2] C not None, int n, double start_ubD):
#   cdef np.ndarray [DTYPE_t, ndim=1]  D_row_sums  =C.sum(axis=1)  
#   cdef double ubD = start_ubD
#   cdef set chosen_row = set()
#   # have we choosen this row or col yet
#   cdef list sigma = [None] * n 
#   cdef int j
#   #cdef np.ndarray [DTYPE_t, ndim=1] weights
#   cdef int ci
#   cdef double ws
#   cdef list res
#   cdef double fail_prob
#   cdef double r
#   for j in range(n):
#     r = rand.random()
#     cum = 0.0
#     choice = None
#     for i in range(n):
#       newrow, score = rem_row_sums_upper(C, D_row_sums, i, j, n, chosen_row) if i not in chosen_row else (None,0.0)
#       weight = (C[i,j] * score)/ ubD
#       cum += weight
#       if r < cum:
#         choice = i, newrow, score 
#         break
#     if choice == None:
#       break
#     else:
#       # choose an element based on weights
#       #ci = weighted_choice(weights/ws, 1)
#       ci, newrow, score = choice
#       #choose i
#       sigma[ci] = j 
#       chosen_row.add(ci)
#       D_row_sums = newrow
#       ubD = score
#   return sigma


# cdef int weighted_choice(np.ndarray [DTYPE_t, ndim=1] weights , int number):
#   cdef np.ndarray [DTYPE_t, ndim=1] c = np.cumsum(weights)
#   cdef double y = rand.random(number)
#   return np.searchsorted(c, y)

cdef inner_sample_wrap(double Cc[50][50], double Dc_row_sums[50], int n, double start_ubD):
  cdef int [50] sigma
  for i in range(n):
    sigma[i] = -1
    
  cdef int d 
  d = inner_sample(<double (*)[50]>Cc, Dc_row_sums, n, start_ubD, sigma)
  cdef list ret = []
  for i in range(n):
    ret.append(sigma[i])
  return d, ret

def outer_sample(np.ndarray [DTYPE_t, ndim=2] C not None, int k, int n, double start_ubD):
  cpdef double Cc[50][50]
  for i in range(n):
    for j in range(n):
      Cc[i][j] = C[i,j]
  cdef np.ndarray [DTYPE_t, ndim=1]  D_row_sums  =C.sum(axis=1)  
  cpdef double Dc_row_sums[50]
  for i in range(n):
    Dc_row_sums[i] = D_row_sums[i]

  cdef int d = 0
  samples = []
  for c in range(k):
    little_d, sigma  = inner_sample_wrap(Cc,Dc_row_sums, n, start_ubD)
    d += little_d
    samples.append(sigma)
  return samples, d
