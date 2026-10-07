/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Ethan L. Edmunds
 *	eledmunds1@sheffield.ac.uk
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_DIFFUSION_H
#define EXADIS_DIFFUSION_H

#include "system.h"
#include "force_fft.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Diffusion
 *
 *-------------------------------------------------------------------------*/
class Diffusion {
public:

    ForceFFT* fft;
    int Ngrid;
    typedef Kokkos::View<complex***, Kokkos::LayoutRight, T_memory_space> T_grid;
    T_grid c;
    double L[3];
    double D = 1.0;
    FFTPlan<> plan;
    bool initialized = false;

    void initialize(System* system) {
        Mat33 H = system->get_serial_network()->cell.H;
        L[0] = H.xx(); L[1] = H.yy(); L[2] = H.zz();
        printf("box L = %g %g %g\n", L[0], L[1], L[2]);

        Kokkos::resize(c, Ngrid, Ngrid, Ngrid);

        plan.initialize(Ngrid, Ngrid, Ngrid);
        initialized = true;
    }

    Diffusion(ForceFFT* _fft, int _Ngrid) : fft(_fft), Ngrid(_Ngrid) {
        if (!fft) ExaDiS_fatal("Diffusion: requires an FFT force model\n");
    }

    void set_gaussian(double s0) {
        auto c = this->c;
        int N = Ngrid;
        double Lx = L[0], Ly = L[1], Lz = L[2];

        Kokkos::parallel_for("Diffusion::SetGaussian",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            double x = i*Lx/N - 0.5*Lx;
            double y = j*Ly/N - 0.5*Ly;
            double z = k*Lz/N - 0.5*Lz;
            double r2 = x*x + y*y + z*z;
            c(i,j,k) = complex(exp(-r2/(2.0*s0*s0)), 0.0);
        });
        Kokkos::fence();
    }

    void compute(System* system) { 
        
        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].start();

        if (!initialized) initialize(system);

        std::vector<Mat33> stress = fft->export_stress_gridval();

        double smin = 1e300, smax = -1e300, ssum = 0.0;
        for (size_t i=0; i < stress.size(); i++) {
            double sh = stress[i].trace() / 3.0;
            if (sh < smin) smin = sh;
            if (sh > smax) smax = sh;
            ssum += sh;
        }
        double smean = ssum / stress.size();

        double s0 = 0.1*L[0], h = L[0]/Ngrid;
        double expect = pow(2.0*M_PI, 1.5) * s0*s0*s0 / (h*h*h);

        set_sine();
        solve_poisson();
        printf("poisson sine test: rel. max error %.2e (amplitude %.4e)\n",
        check_sine(), pow(L[0]/(2.0*M_PI), 2) / D);
    
        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].stop();

    }

    double total() {
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        double sum = 0.0;
        for (int i = 0; i < Ngrid; i++)
        for (int j = 0; j < Ngrid; j++)
        for (int k = 0; k < Ngrid; k++)
            sum += h_c(i,j,k).real();
        return sum;
    }

    void fft_roundtrip() {
        FFT3DTransform(plan, c, c, FFT_FORWARD);
        Kokkos::fence();
        FFT3DTransform(plan, c, c, FFT_BACKWARD);
        Kokkos::fence();

        auto c = this->c;
        double invN3 = 1.0 / ((double)Ngrid*Ngrid*Ngrid);
        int N = Ngrid;
        Kokkos::parallel_for("Diffusion::Normalize",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                c(i,j,k) *= invN3;
            });
        Kokkos::fence();
    }

    void set_sine() {
        auto c = this->c;
        int N = Ngrid;
        double Lz = L[2];
        Kokkos::parallel_for("Diffusion::SetSine",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                double z = k*Lz/N;
                c(i,j,k) = complex(sin(2.0*M_PI*z/Lz), 0.0);
            });
        Kokkos::fence();
    }

    double check_sine() {
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        double amp = (L[0]/(2.0*M_PI)) * (L[0]/(2.0*M_PI)) / D;
        double maxerr = 0.0;
        for (int i = 0; i < Ngrid; i++)
        for (int j = 0; j < Ngrid; j++)
        for (int k = 0; k < Ngrid; k++) {
            double z = k*L[2]/Ngrid;
            double exact = -amp * sin(2.0*M_PI*z/L[2]);
            maxerr = fmax(maxerr, fabs(h_c(i,j,k).real() - exact));
        }
        return maxerr / amp;
    }

    void solve_poisson() {
        FFT3DTransform(plan, c, c, FFT_FORWARD);
        Kokkos::fence();

        auto c = this->c;
        int N = Ngrid;
        double invN3 = 1.0 / ((double)N*N*N);
        double tx = 2.0*M_PI/L[0], ty = 2.0*M_PI/L[1], tz = 2.0*M_PI/L[2];
        double D1 = D;

        Kokkos::parallel_for("Diffusion::Poisson",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int kmax = N/2 + N%2;
                int ii = (i >= kmax) ? i - N : i;
                int jj = (j >= kmax) ? j - N : j;
                int kk = (k >= kmax) ? k - N : k;

                double kx = tx*ii, ky = ty*jj, kz = tz*kk;
                double k2 = kx*kx + ky*ky + kz*kz;

                if (k2 == 0.0) {
                    c(i,j,k) = complex(0.0, 0.0);
                } else {
                    c(i,j,k) *= -invN3 / (D1*k2);
                }
            });
        Kokkos::fence();

        FFT3DTransform(plan, c, c, FFT_BACKWARD);
        Kokkos::fence();
    }

};

} // namespace ExaDiS

#endif