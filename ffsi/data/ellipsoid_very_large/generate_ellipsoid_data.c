/*
 * Generate Intensity Data and Distributions for the SAS Ellipsoid
 * (takes about four hours on 16 cores for the very large example)
 *
 * Compile: gcc -fopenmp -O2 generate_ellipsoid_data.c -o gen_elp -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


double G_ellipsoid(double qx, double qy, double rp, double re, double theta, double phi, double drho) {

    // ellipsoid volume
    double V = 4/3 * M_PI * rp * re * re;

    // ellipsoid scattering amplitude
    double qc = qx * sin(theta) * cos(phi) + qy * sin(theta) * sin(phi);
    double qa = sqrt(qx * qx + qy * qy - qc * qc);
    double qr = sqrt(pow(qa * re, 2) + pow(qc * rp, 2));
    double F = 3.0 * V * drho * (sin(qr) - qr * cos(qr)) / pow(qr, 3);

    return F * F;
}


typedef struct {
    double factor;
    double mean;
    double stddev;
} Gaussian;

void crazy_distribution(const double *x, int n, const Gaussian *gauss, int ngauss,
                        double noise_level, int fade_start, int fade_end, double *out) {
    for (int i = 0; i < n; ++i) out[i] = 0.0;
    for (int g = 0; g < ngauss; ++g) {
        double factor = gauss[g].factor;
        double mean = gauss[g].mean;
        double stddev = gauss[g].stddev;
        if (stddev == 0.0) continue;
        for (int i = 0; i < n; ++i) {
            double arg = (x[i] - mean) / stddev;
            out[i] += factor * exp(- (arg * arg));
        }
    }
    srand(0);
    for (int i = 0; i < n; ++i) {
        double r1 = (double)rand() / (double)RAND_MAX;
        double r2 = (double)rand() / (double)RAND_MAX;
        out[i] += noise_level * r1 * r2;
    }
    if (n >= 3) {
        int m = fade_end - fade_start;
        for (int i = 0; i < fade_start; ++i) out[i] = 0.0;
        for (int k = 0; k < m; ++k) {
            double t = (m == 1) ? 1.0 : ((double)k / (double)(m - 1));
            out[fade_start + k] *= t;
        }
        for (int i = n - fade_start; i < n; ++i) out[i] = 0.0;
        for (int k = 0; k < m; ++k) {
            double t = (m == 1) ? 1.0 : (1.0 - (double)k / (double)(m - 1));
            out[n - fade_end + k] *= t;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += out[i];
    for (int i = 0; i < n; ++i) out[i] /= sum;
}

void linspace(double start, double end, int n, double *out) {
    if (n == 1) {
        out[0] = start;
        return;
    }
    double step = (end - start) / (double)(n - 1);
    for (int i = 0; i < n; ++i) out[i] = start + step * i;
}

void logspace(double startExp, double endExp, int n, double *out) {
    double *exps = (double *)malloc(n * sizeof(double));
    linspace(startExp, endExp, n, exps);
    for (int i = 0; i < n; ++i) out[i] = pow(10.0, exps[i]);
    free(exps);
}


int main() {

    const int nqx = 120;
    const int nqy = 120;
    const int side_len = 50;
    const int center_len = 20;

    double *q_side = (double *)malloc(side_len * sizeof(double));
    double *q_center = (double *)malloc(center_len * sizeof(double));
    double *qx = (double *)malloc(nqx * sizeof(double));
    double *qy = (double *)malloc(nqy * sizeof(double));

    logspace(-2.0, 0.0, side_len, q_side);
    linspace(-0.0095, 0.0095, center_len, q_center);

    for (int i = 0; i < side_len; ++i) qx[i] = -q_side[side_len - 1 - i];
    for (int i = 0; i < center_len; ++i) qx[side_len + i] = q_center[i];
    for (int i = 0; i < side_len; ++i) qx[side_len + center_len + i] = q_side[i];
    for (int i = 0; i < nqy; ++i) qy[i] = qx[i];

    free(q_side);
    free(q_center);

    const int rpl = 50;
    const int rpu = 90;
    const int nrp = 100;

    const int rel = 200;
    const int reu = 600;
    const int nre = 100;

    const int thetal = 20;
    const int thetau = 75;
    const int ntheta = 100;

    const int phil = 150;
    const int phiu = 240;
    const int nphi = 100;

    double *rp = (double *)malloc(nrp * sizeof(double));
    double *re = (double *)malloc(nre * sizeof(double));
    double *theta = (double *)malloc(ntheta * sizeof(double));
    double *phi = (double *)malloc(nphi * sizeof(double));

    linspace((double)rpl, (double)rpu, nrp, rp);
    linspace((double)rel, (double)reu, nre, re);
    linspace((double)thetal, (double)thetau, ntheta, theta);
    linspace((double)phil, (double)phiu, nphi, phi);

    double *w_rp_true = (double *)malloc(nrp * sizeof(double));
    double *w_re_true = (double *)malloc(nre * sizeof(double));
    double *w_theta_true = (double *)malloc(ntheta * sizeof(double));
    double *w_phi_true = (double *)malloc(nphi * sizeof(double));

    // generate w distributions
    int ng = 3;
    Gaussian grp[] = {{1, 60, 3}, {2, 70, 4}, {2, 80, 3}};
    crazy_distribution(rp, nrp, grp, ng, 0, 1, 1, w_rp_true);
    Gaussian gre[] = {{1.5, 300, 20}, {1, 400, 20}, {2, 500, 20}};
    crazy_distribution(re, nre, gre, ng, 0, 1, 1, w_re_true);
    Gaussian gtheta[] = {{4, 30, 5}, {2, 50, 5}, {2, 65, 5}};
    crazy_distribution(theta, ntheta, gtheta, ng, 0, 1, 1, w_theta_true);
    Gaussian gphi[] = {{2, 170, 10}, {2, 200, 10}, {4, 220, 10}};
    crazy_distribution(phi, nphi, gphi, ng, 0, 1, 1, w_phi_true);

    // convert degrees to radians
    for (int i = 0; i < ntheta; ++i) theta[i] = theta[i] * M_PI / 180.0;
    for (int i = 0; i < nphi; ++i) phi[i] = phi[i] * M_PI / 180.0;

    // save distributions
    FILE *fl = fopen("c_w_rp_true.txt","w");
    for (int irp = 0; irp < nrp; ++irp) {
            fprintf(fl, "%.18e\n", w_rp_true[irp]);
    }
    fclose(fl);
    FILE *fr = fopen("c_w_re_true.txt","w");
    for (int ire = 0; ire < nre; ++ire) {
            fprintf(fr, "%.18e\n", w_re_true[ire]);
    }
    fclose(fr);
    FILE *ft = fopen("c_w_theta_true.txt","w");
    for (int it = 0; it < ntheta; ++it) {
            fprintf(ft, "%.18e\n", w_theta_true[it]);
    }
    fclose(ft);
    FILE *fp = fopen("c_w_phi_true.txt","w");
    for (int ip = 0; ip < nphi; ++ip) {
            fprintf(fp, "%.18e\n", w_phi_true[ip]);
    }
    fclose(fp);

    // compute xi and b
    double scale_true = 0.15;
    double b_true = 2.2e-4;
    printf("b_true: %.2e\n", b_true);
    double *V = (double *)malloc((size_t)nrp * nre * sizeof(double));
    for (int irp = 0; irp < nrp; ++irp) {
        for (int ire = 0; ire < nre; ++ire) {
            V[irp * nre + ire] = 4/3 * M_PI * rp[irp] * re[ire] * re[ire];
        }
    }
    double V_ave = 0.0;
    for (int irp = 0; irp < nrp; ++irp) {
        for (int ire = 0; ire < nre; ++ire) {
            V_ave += w_rp_true[irp] * V[irp * nre + ire] * w_re_true[ire];
        }
    }
    double xi_true = 1e-4 * scale_true / V_ave;
    printf("xi_true: %.2e\n", xi_true);

    printf("\nComputing I=xi*Gw+b to simulate the intensities...\n");

    double *I_true = (double *)calloc((size_t)nqx * nqy, sizeof(double));
    if (!I_true) {
        fprintf(stderr, "Failed to allocate I_true\n");
        return 1;
    }

    double drho = 1.0;

    // parallelize outer qx loop with OpenMP
    #pragma omp parallel for schedule(dynamic) default(none) shared(I_true, xi_true, b_true, qx, qy, rp, re, theta, phi, w_rp_true, w_re_true, w_theta_true, w_phi_true, nqx, nqy, nrp, nre, ntheta, nphi, drho)
    for (int iqx = 0; iqx < nqx; ++iqx) {
        // print progress once per iqx (protected to avoid garbled output)
        #pragma omp critical
        {
            printf("  progress at iqx %d out of %d\n", iqx + 1, nqx);
        }
        for (int iqy = 0; iqy < nqy; ++iqy) {
            double sum_for_point = 0.0;
            for (int irp = 0; irp < nrp; ++irp) {
                for (int ire = 0; ire < nre; ++ire) {
                    for (int it = 0; it < ntheta; ++it) {
                        for (int ip = 0; ip < nphi; ++ip) {
                            double g = G_ellipsoid(qx[iqx], qy[iqy], rp[irp], re[ire], theta[it], phi[ip], drho);
                            double wprod = w_rp_true[irp] * w_re_true[ire] * w_theta_true[it] * w_phi_true[ip];
                            sum_for_point += g * wprod;
                        }
                    }
                }
            }
            I_true[iqx * nqy + iqy] = xi_true * sum_for_point + b_true;
        }
    }

    // write intensities to file
    FILE *f = fopen("c_intensities.txt","w");
    for (int iqx = 0; iqx < nqx; ++iqx) {
        for (int iqy = 0; iqy < nqy; ++iqy) {
            fprintf(f, "%.18e ", I_true[iqx * nqy + iqy]);
        }
        fprintf(f,"\n");
    }
    fclose(f);

    // cleanup
    free(qx);
    free(qy);
    free(rp);
    free(re);
    free(theta);
    free(phi);
    free(w_rp_true);
    free(w_re_true);
    free(w_theta_true);
    free(w_phi_true);
    free(V);
    free(I_true);

    return 0;
}
