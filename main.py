import pickle
import time
import math
from munkres import Munkres, print_matrix
import numpy as np
import numpy.random as rand
import itertools
from sample import *
rand.seed(1)

eng_vocab = ["A", "B", "C"]

examples = {"simple": np.array([[1.0,0.0],[0.0,1.0]]),
            "bigger": np.array([[1.0,1.0,0.0,0.0],
                                [1.0,1.0,0.0,0.0],
                                [0.0,1.0,1.0,0.0],
                                [0.0,0.1,0.0,1.0]]),
            "huge": np.eye(20),
            "split" : np.array([[0.5,0.5],[0.5,0.5]]),
            "diff" : np.array([[0.25,0.35],[0.45,0.55]])}


def powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))



def ryser(mat):
  """
  Given a square matrix, compute the permanent exactly
  slow-tastic
  """
  n = len(mat)
  total = 0.0
  for S in powerset(range(0,n)): 
    # \product_i \sum_{j\in C} A_{ij}
    size =  len(S)
    total += ((-1)**size) * mat.take(S, axis=1).sum(axis=1).prod()
  return ((-1)**n) * total

def naive_permanent(mat):
  """
  Compute the permanent by naive enumeration
  """
  n = len(mat)
  total = 0.0
  samples = []
  for S in itertools.permutations(range(0,n)): 
    #print S
    match_score = 1.0

    for i in range(n):
      match_score *= mat[i, S[i]] 
    samples.append((match_score, [S[i] for i in range(n)]))
    total += match_score
  return total, samples 


def sinkhorn(A, accuracy):
  """
  Find matrices X and Y such that |XAY1 - 1|_inf < accuracy and |YA^TX1 - 1| < accuracy
  """
  n = len(A) 
  X = np.diag(np.ones(n))
  Y = np.diag(np.ones(n))
  while scale_accuracy(X,Y,A) > accuracy:    
    cur_mat = np.dot(np.dot(X, A), Y)
    row_sums = cur_mat.sum(axis=1)
    X = np.diag(1.0 / row_sums) * X
    cur_mat = np.dot(np.dot(X, A), Y)
    col_sums = cur_mat.sum(axis=0)
    Y = np.diag(1.0 / col_sums) * Y
    #print scale_accuracy(X,Y,A)
  return (X,Y)

def my_log(r):
  if r ==0:
    return -1e20
  else:
    return math.log(r)

my_log_vec = np.vectorize(my_log)

def munkres(mat):
  m = Munkres()
  indexes = m.compute((-my_log_vec(mat)).tolist())  
  return np.array([mat[i,j] for i,j in indexes ]).prod(), [ind[1] for ind in indexes]


def bregmans_upper(r):
  if r >= 1:
    return r + 0.5 *math.log(r) + math.e - 1
  elif r >=-1e-5 and r < 1: 
    return 1  +(math.e -1) *r
  else:
    assert False, "bad r"

bvec = np.vectorize(bregmans_upper, otypes=[np.double])

def upper_bound(A):
  row_sums  = A.sum(axis=1)  
  return (bvec(row_sums)/ math.e).prod() 

def rem_row_sums(original_mat, row_sums, i, j, picked_rows):
  t = row_sums - original_mat[:,j]
  for r in picked_rows:
    t[r] = 1
  t[i] = 1
  #print t
  return t

def fast_upper_bound(row_sums):
  return (bvec(row_sums)/ math.e).prod() 

def matrix_density(mat):
  gamma = min(mat.sum(axis=1).min(), mat.sum(axis=0).min())
  n = len(mat)
  return gamma/ float(n)


def scale_accuracy(X, Y, A):
  n = len(X)
  O = np.ones((n,1))
  
  T1 = (np.dot(np.dot(np.dot(X, A), Y) ,O)  - O)
  T2 = (np.dot(np.dot(np.dot(Y, A.transpose()), X) ,O)  - O)
  return max(abs(T1).max(), abs(T2).max())


def factorial(k):
  return reduce(lambda i,j : i*j, range(1,k+1))

def rem(M, i, j):
  """
  Replace row i, column j with 0, set M[i,j] =1
  """
  n = len(M)
  D = M.copy()
  z = np.zeros(n)
  D[i,:] = z
  D[:,j] = z
  D[i,j] = 1.0
  return D

class PermanentSampler:
  def __init__(self, delta, epsilon, zeroone =True):
    self.delta = delta
    self.epsilon = epsilon
    self.zeroone = zeroone
    self.debug = False
  def estimate_permanent(self, mat):
    # lower bound (just one)
    n = len(mat)
    if not self.zeroone:
      alpha3,_ = munkres(mat)
      largest = mat.max()
      mat2 = mat / largest
      mat3 = self.prescale_matrix(mat2, alpha3)
      dens = matrix_density(mat2)
      #print dens
      #print mat3
    else:
      largest = 1.0
      mat3 = self.simple_prescale_matrix(mat)
    (X,Y,Z,C) = self.scale_matrix(mat3)

    perm, samples = self.sample(C,X,Y,Z)
    #print samples
    return (largest** n) * perm, samples
 


  def simple_prescale_matrix(self, mat):
    A = mat.copy()
    n = len(A)
    dens = matrix_density(A)
    #print dens
    if not dens > .5:
      alpha1 = (self.delta/3.0) / factorial(n) 
    else:
      alpha1 = (self.delta/3.0) / (factorial(n)**3)

    for i in range(n):
      for j in range(n):
        if A[i,j] == 0:
          A[i,j] = alpha1
    return A
 
  def prescale_matrix(self, mat, alpha3):
    A = mat.copy()
    n = len(A)
    alpha1 = (self.delta/3.0)*alpha3 / math.factorial(n) 

    for i in range(n):
      for j in range(n):
        if A[i,j] < alpha1:
          A[i,j] = alpha1
    return A
    
  def scale_matrix(self, mat):
    """
    Assume 0,1 for now
    """
    A = mat.copy()
    n = len(A)
 
    alpha2 = .1 / (n ** 2)

    # use sinkhorn or ellipsoid
    #print "Sinkhorn"
    X, Y = sinkhorn(A, alpha2)
    B = np.dot(np.dot(X , A) , Y) 
    #print "B",B
    Z = np.diag([(1.0/B).take([i], axis=0).min() for i in range(n)])
    #print Z
    C = np.dot(Z,B)
    return (X,Y, Z, C)

  def sample(self, C, X, Y, Z):
    d = 0
    n = len(C)
    k = 14 * (1.0 / (self.delta ** 2)) * math.log(2.0/ self.epsilon)
    #print "k", k
    samples = []

    start_ubD = fast_upper_bound(C.sum(axis=1))

    samples,d = outer_sample(C, round(k), n, start_ubD)

    # for c in range(round(k)):
    #   sigma = [None]
    #   while any(s == None for s in sigma):
    #     d += 1
    #     sigma = inner_sample_wrap(C, n, start_ubD)
    #     print sigma
        # D_row_sums  = C.sum(axis=1)  
        # ubD = fast_upper_bound(D_row_sums)
        # d += 1
        # chosen_row = set()
        # # have we choosen this row or col yet
        # sigma = [None] * n 
        # for j in range(n):
        #   res = [rem_row_sums_upper(C, D_row_sums, i, j, n, chosen_row) if i not in chosen_row else (None,0.0) 
        #          for i in range(n)]

        #   weights = np.array([ (C[i,j] * score)/ ubD  for i, (_,score) in enumerate(res)])
          
        #   ws = weights.sum()
        #   fail_prob = 1.0 - weights.sum()
          
        #   if rand.random() < fail_prob:
        #     # fail case need to restart
        #     break 
        #   else:
        #     # choose an element based on weights
        #     ci = weighted_choice(weights/ws, 1)[0]
            
        #     #choose i
        #     sigma[ci] = j 
        #     chosen_row.add(ci)
        #     D_row_sums = res[ci][0] 
        #     ubD = res[ci][1]

      #samples.append(sigma)
    s = np.array([ X[i,i] *Z[i,i] *Y[i,i] for i in range(n)]).prod()
    if self.debug:
      print "M:", upper_bound(C)
      print "K:",k
      print "d:",d
      print "s:", s
    perm = (upper_bound(C) * k)/ (d * s)

    return perm, samples

import data_gen,dist
NAIVE = 0
SAMPLE = 1 
VITERBI = 2
MANYTOONE = 3

model_names =  {
  "naive" : NAIVE,
  "sample" : SAMPLE,
  "viterbi" : VITERBI,
  "manyone": MANYTOONE
  }

def model1(dists, instances, prob_model):
  f, e = instances.split()
  e_counts = {}
  e_ret = {}
  perm = 1.0 
  for e_type in e:
    for f_type in f:
      t_model = prob_model.prob(dists,f_type,e_type)
      e_counts.setdefault(e_type, 0.0)
      e_counts[e_type] += t_model
    for f_type in f:
      t_model = prob_model.prob(dists,f_type,e_type)
      e_ret.setdefault(e_type, [])
      e_ret[e_type].append((t_model/ e_counts[e_type], f_type) )
    perm *= e_counts[e_type]
  return perm, e_ret
      
def em(instances, dists, mode, prob_model):
  f_types, e_types = prob_model.types()
  psamp = PermanentSampler(0.5,0.1,zeroone=True)
  n=len(e_types.keys())

  counts = {}
  total_perm = 0.0
  #print dists
  for num,instance in enumerate(instances):
    #print num, mode
    mat = prob_model.gen_mat(dists, instance) 
    if mode == NAIVE:
      _, samples = naive_permanent(mat)
    elif mode == SAMPLE:
      perm, samples_tmp = psamp.estimate_permanent(mat)
      samples = [ (1.0,s) for s in samples_tmp]
    elif mode == VITERBI:
      perm, _ = naive_permanent(mat)
      _, s = munkres(mat)
      samples = [ (1.0,s)]
    elif mode == MANYTOONE:
      #start = time.time()
      perm, samples = model1(dists, instance, prob_model)
      #print time.time() - start
    #perm, _ = naive_permanent(mat)
    #print perm, math.log(perm)
    total_perm += math.log(perm)

    ins_counts = {}
    for k in instance.f:
      #ins_counts[k] = np.zeros(n)
      ins_counts[k] = {}

    a,b = instance.split()

    if mode == MANYTOONE:
      for eng, eng_counts in samples.iteritems():
        total_counts = 0.0
        for exp_count, fre in eng_counts:
          #print fre, eng
          ins_counts[fre].setdefault(e_types[eng], 0.0)
          ins_counts[fre][e_types[eng]] += exp_count
    else:
      for p, s in samples:
        for j, to in enumerate(s):
          ins_counts[a[j]].setdefault(e_types[b[to]], 0.0)
          if mode == NAIVE:
            ins_counts[a[j]][e_types[b[to]]] += p/float(perm)
          elif mode == SAMPLE:
            ins_counts[a[j]][e_types[b[to]]] += p/float(len(samples))
          elif mode == VITERBI:
            ins_counts[a[j]][e_types[b[to]]] += p

    for k in instance.f:
      ins_counts[k] = dist.normalize_sparse(ins_counts[k])
    counts = dist.add_sparse(counts, ins_counts)


  dists = dist.maximum_likelihood_sparse(counts)
  #kl = dist.kl_tables(dists, data_gen.real_dist())
  #print "PERM: %s %s %s"% (i, total_perm, kl)
  print "PERM:  %s"% ( total_perm)
  return dists


def viterbi_align_oneone(instance, dists, prob_model):
  mat = prob_model.gen_mat(dists, instance) 
  _, s = munkres(mat)
  return [(i,other) for i,other in enumerate(s) if instance.e[other]<>"*EPS*" and instance.f[i]<>"*EPS*" ]


def viterbi_align_manyone(instance, dists, prob_model):
  a, b = instance.split()
  ret = []
  for j, e_type in enumerate(b):
    m = -100000
    best = None
    for i, f_type in enumerate(a):
      score = prob_model.prob(dists, f_type, e_type)
      if score > m:
        m = score
        best = (i,j)
        if f_type == '*EPS*':
          best = -1
        if e_type == '*EPS*':
          best = -1
    if best <> -1:
      ret.append(best)
  return ret

def dev_assess(instances, alignments, dists, prob_model, viterbi_fn):
  all = []
  test_alignments = []
  for j, ins in enumerate(instances):
    
    res = viterbi_fn(ins, dists, prob_model)    
    align = alignments[ins.num]
    score = align.aer(res)
    test_alignments.append(res)
    #if j ==0:
      #print ins.e, ins.f
      #print res
    
  
    all.append(score)
  return sum(all) / float(len(all)), test_alignments

def main():
  # m = examples["diff"]
  # print PermanentSampler(0.5,0.1).estimate_permanent(m)

  # r_perm = ryser(m)
  # #n_perm = naive_permanent(m)
  
  # print r_perm
  # #print n_perm




  from align import Align
  from align import Alignment
  import sys
  mode = model_names[sys.argv[3]]

  align = Align.from_files("data/eng-fr.full.fr","data/eng-fr.full.en")
  e_types, f_types = align.types()

  prob_model = dist.ProbModel(f_types, e_types)
  
  test_align = Align.from_files("data/eng-fr.dev.fr","data/eng-fr.dev.en", True)
  gold_align= Alignment.read_alignments("data/eng-fr.dev.align")
  if sys.argv[1] == "train":
  
    dists = prob_model.rand_dists_sparse()  
    instances = align.instances()
    test_instances = test_align.instances()
    for r in range(20):
      
      if mode == MANYTOONE:
        score,_ = dev_assess(test_instances, gold_align, dists, prob_model, viterbi_align_manyone)
      else:
        score,_ = dev_assess(test_instances, gold_align, dists, prob_model, viterbi_align_oneone)
      dists = em(instances, dists, mode, prob_model)
      #print "Dist 'le'"
      #for i, lscore in enumerate(dists['le']):
      #  if lscore > 1e-4:
      #    print i, align.eng_to_ind(i), lscore
      print "Score is:", score
    final_dist = dists
    pickle.dump(final_dist, open(sys.argv[2], 'wb'))

  elif sys.argv[1] == "test":
    dists = pickle.load(open(sys.argv[2], 'rb'))
    
    instances = test_align.instances()
    

    if mode == MANYTOONE:
      score,alignments = dev_assess(instances, gold_align, dists, prob_model, viterbi_align_manyone)
    else:
      score,alignments = dev_assess(instances, gold_align, dists, prob_model, viterbi_align_oneone)
    print score
    f_out = open("out.f", 'w')
    e_out = open("out.e", 'w')
    a_out = open("out.a", 'w')
    gold_out = open("out.gold.a", 'w')
    for ins,align in zip(instances,alignments):
      print >>f_out, " ".join( ins.f)
      print >>e_out, " ".join(ins.e)
      print >>a_out, " ".join([str(e)+"-"+str(f) for e, f in align])
      print >>gold_out, " ".join([str(e)+"-"+str(f) for e, f in gold_align[ins.num]])
    
#   print data_gen.real_dist()
#   instances = data_gen.generate_data()
#   mode = SAMPLE

#   em(instances, mode,dist.types)
 

if __name__== "__main__":
  main()
