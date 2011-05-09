cdef extern from "fast_sample.h":
    int inner_sample(double C[50][50], double Dc[50], int n, double start_ubD, int sigma[50])