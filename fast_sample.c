#include "math.h"
#include "fast_sample.h"
#include <stdlib.h>


double rem_row_sums_upper(double original_mat[MAX_ROW][MAX_ROW], 
                          double row_sums[MAX_ROW], 
                          int i, int j, int n, int picked_rows[MAX_ROW]) {
  
  double total = 1.0;
  double local;
  int k;

  for ( k =0; k < n; k++) {
    if (picked_rows[k] || k == i) continue;
    
    double r = row_sums[k] - original_mat[k][j];  
    if (r >= 1) {
      local = r + 0.5 *log(r) + M_E - 1;
    }
    else if (r >=-1e-5 && r < 1)  {
      local = 1  +(M_E -1) *r;
    }
    total *= local / M_E;
  }
  return total;
}

// sample until success
int inner_sample(double C[MAX_ROW][MAX_ROW] , double Dc_row_sums[MAX_ROW], int n, double start_ubD, int sigma[MAX_ROW]) {
  double D_row_sums[MAX_ROW], new_row_sums[MAX_ROW];
  int d = 0;
  int j, i;
  int success = 0;

  while (success == 0) {
    success = 1;
    for (i =0; i< n; i++) {
      D_row_sums[i] = Dc_row_sums[i];
    }

    double ubD = start_ubD;

    //have we choosen this row or col yet
    int chosen_row[MAX_ROW];
  
    for ( j=0; j < n; j++) {
      chosen_row[j] = 0;
    }
    
    for ( j=0; j < n; j++) {
      double rand_num = (double)rand()/(double)RAND_MAX;
      //printf("%f\n",r);
      double cum = 0.0;
      int choice = 0;

      int ci;

      double cache_score = 1.0;
      int k;
      for ( k =0; k < n; k++) {
        if (chosen_row[k]) continue;      
        double r = D_row_sums[k] - C[k][j];  
        if (r >= 1) {
          cache_score *= (r + 0.5 *log(r) + M_E - 1) / M_E;
        } else if (r >=-1e-5 && r < 1)  {
          cache_score *= (1  +(M_E -1) *r) / M_E;
        }
      }
      
      double score;
      for ( i =0; i <n; i++) {
        if (chosen_row[i]) continue;
      
        double r = D_row_sums[i] - C[i][j];
        if (r >= 1) {
          score = cache_score / ((r + 0.5 *log(r) + M_E - 1) / M_E);
        }
        else if (r >=-1e-5 && r < 1)  {
          score = cache_score / ((1  +(M_E -1) *r) / M_E);
        }
    
        //score = rem_row_sums_upper(C, D_row_sums, i, j, n, chosen_row); 
        cum +=  (C[i][j] * score)/ ubD;;
        //printf("%d %f %f %f %f %f %d %d\n",i, C[i][j], cache_score, score, cum, rand_num, i, j);
        if (rand_num < cum) {
          choice = 1;
          ci = i;
          break;
        }
      }

      if (choice==0) {
        success = 0;
        break;
      } else {
        sigma[ci] = j;
        //printf("%d\n",ci);
        //printf("%d\n",j);
        chosen_row[ci] = 1;

        for (i =0; i< n; i++) {        
          if (chosen_row[i])
            D_row_sums[i] = 1;
          else
            D_row_sums[i] = D_row_sums[i] - C[i][j];
        }
        ubD = score;
      }
    }
    d +=1;
  }
  return d;
}
