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
    Kokkos::View<double***, Kokkos::LayoutRight, T_memory_space> wk;
    Cell cell;
    bool initialized = false;

    void initialize(System* system) {
        Mat33 H = system->get_serial_network()->cell.H;
        L[0] = H.xx(); L[1] = H.yy(); L[2] = H.zz();
        printf("box L = %g %g %g\n", L[0], L[1], L[2]);

        cell = system->get_serial_network()->cell;

        Kokkos::resize(c, Ngrid, Ngrid, Ngrid);

        plan.initialize(Ngrid, Ngrid, Ngrid);
        init_spreading(2.0*L[0]/Ngrid);
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

        double s0 = 0.05*L[0];
        double h = L[0]/Ngrid;

        double hval = 1.0, a = 2.0*L[0]/Ngrid;
        set_line_source(hval, 8);
        solve_poisson();
        printf("line source poisson (a=2h): rel. error %.2e\n", check_line(hval, a));
    
        printf("line + short-range correction: rel. error %.2e\n",
               check_line_corrected(hval, a));

        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].stop();

    }

    void init_spreading(double a) {
        Kokkos::resize(wk, Ngrid, Ngrid, Ngrid);
        auto h_wk = Kokkos::create_mirror_view(wk);
        int N = Ngrid, kmax = N/2 + N%2;
        double tx = 2.0*M_PI/L[0], ty = 2.0*M_PI/L[1], tz = 2.0*M_PI/L[2];

        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++) {
            int ii = (i >= kmax) ? i - N : i;
            int jj = (j >= kmax) ? j - N : j;
            int kk = (k >= kmax) ? k - N : k;

            double kx = tx*ii, ky = ty*jj, kz = tz*kk;
            double ka = sqrt(kx*kx + ky*ky + kz*kz) * a;
            h_wk(i,j,k) = (ka > 0.0) ? 0.5*ka*ka*std::cyl_bessel_k(2.0, ka) : 1.0;
        }
        Kokkos::deep_copy(wk, h_wk);
    }

    void deposit_segment(T_grid::HostMirror& h_s, Vec3 r1, Vec3 r2, double hval) {
        Vec3 Hs(1.0/Ngrid, 1.0/Ngrid, 1.0/Ngrid);
        double Vs = cell.H.det() / ((double)Ngrid*Ngrid*Ngrid);

        r2 = cell.pbc_position(r1, r2);
        Vec3 t = r2 - r1;
        double L = t.norm();
        if (L < 1e-10) return;

        r1 = cell.scaled_position(r1);
        r2 = cell.scaled_position(r2);

        double Ls = (r2 - r1).norm();
        double scale = L / Ls;
        t = t.normalized();

        int imin = floor((fmin(r1.x, r2.x) - 0.5*Hs.x)/Hs.x);
        int imax = floor((fmax(r1.x, r2.x) - 0.5*Hs.x)/Hs.x) + 1;
        int jmin = floor((fmin(r1.y, r2.y) - 0.5*Hs.y)/Hs.y);
        int jmax = floor((fmax(r1.y, r2.y) - 0.5*Hs.y)/Hs.y) + 1;
        int kmin = floor((fmin(r1.z, r2.z) - 0.5*Hs.z)/Hs.z);
        int kmax = floor((fmax(r1.z, r2.z) - 0.5*Hs.z)/Hs.z) + 1;

        for (int ib = imin; ib <= imax; ib++)
        for (int jb = jmin; jb <= jmax; jb++)
        for (int kb = kmin; kb <= kmax; kb++) {
            Vec3 bc((ib+0.5)*Hs.x, (jb+0.5)*Hs.y, (kb+0.5)*Hs.z);
            double W = scale * fft->alpha_box_segment(r1, t, Ls, bc, Hs);

            int kx = ib % Ngrid; if (kx < 0) kx += Ngrid;
            int ky = jb % Ngrid; if (ky < 0) ky += Ngrid;
            int kz = kb % Ngrid; if (kz < 0) kz += Ngrid;

            h_s(kx, ky, kz) += complex(W/Vs * hval, 0.0);   // serial → plain += is safe
        }
    }

    void set_line_source(double hval, int nseg) {
        auto h_s = Kokkos::create_mirror_view(c);
        Kokkos::deep_copy(h_s, complex(0.0, 0.0));

        Vec3 o = cell.origin;
       
        double h = L[0]/Ngrid;
        int m = Ngrid/2;
        double x0 = o.x + (m + 0.5)*h, y0 = o.y + (m + 0.5)*h;

        for (int s = 0; s < nseg; s++) {
            Vec3 r1(x0, y0, o.z + s    *L[2]/nseg);
            Vec3 r2(x0, y0, o.z + (s+1)*L[2]/nseg);
            deposit_segment(h_s, r1, r2, hval);
        }
        Kokkos::deep_copy(c, h_s);
    }

    double check_line (double lam, double a) {
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        int N = Ngrid, m = N/2;
        double h = L[0]/N, A = L[0]*L[1];

        int iref = m + 3;
        double rref = 3.0*h;
        double cref = h_c(iref, m, 0).real();

        auto f = [&](double r) {
            double E1 = (a > 0) ? -std::expint(-r*r/(2.0*a*a)) : 0.0;
            return lam/(4.0*M_PI*D) * (log(r*r) + E1);
        };

        double maxerr = 0.0, scale = 0.0;
        for (int i = iref + 1; i <= m + N/4; i++) {
            double r = (i-m)*h;
            double expect = f(r) - f(rref) -lam*(r*r - rref*rref)/(4.0*D*A);
            double got = h_c(i, m, 0).real() - cref;
            maxerr = fmax(maxerr, fabs(got - expect));
            scale = fmax(scale, fabs(expect));
        }
        return maxerr / scale;
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

    double check_gaussian(double s0, double Q) {
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        int N = Ngrid, m = N/2;
        double h = L[0]/N, V = L[0]*L[1]*L[2];
        double c0 = h_c(m,m,m).real();
        double ciso0 = -Q/(4.0*M_PI*D) * sqrt(2.0/M_PI)/s0;

        double maxerr = 0.0, scale = 0.0;
        for (int i = m+1; i <= m + N/4; i++) {
            double r = (i - m)*h;
            double ciso = -Q/(4.0*M_PI*D*r) * erf(r/(sqrt(2.0)*s0));
            double expect = (ciso - ciso0) - Q*r*r/(6.0*D*V);
            double got = h_c(i,m,m).real() - c0;
            maxerr = fmax(maxerr, fabs(got - expect));
            scale = fmax(scale, fabs(expect));
        }
        return maxerr / scale;
    }

    double check_line_corrected(double lam, double a) {
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        int N = Ngrid, m = N/2;
        double h = L[0]/N, A = L[0]*L[1];

        // Short range correction for an infitnite straight line
        auto delta = [&](double r) {
            double R2 = r*r + a*a;
            return lam/(4.0*M_PI*D) * (log(r*r) - log(R2) + a*a/R2);
        };

        int iref = m + 8;
        double rref = 8.0*h;
        double cref = h_c(iref, m, 0).real() + delta(rref);

        double maxerr = 0.0, scale = 0.0;
        for (int i = m + 1; i < iref; i++) {
            double r = (i - m)*h;
            
            double c_line = lam/(2.0*M_PI*D) * log(r/rref);
            double c_bg   = lam*(r*r - rref*rref)/(4.0*D*A);
            double expect = c_line - c_bg;

            double got = h_c(i, m, 0).real() + delta(r) - cref;
            printf("  r/h=%d  got %.6f  expect %.6f\n", i - m, got, expect);
            maxerr = fmax(maxerr, fabs(got - expect));
            scale = fmax(scale, fabs(expect));
        }
        return maxerr / scale;
    }

    void solve_poisson() {
        FFT3DTransform(plan, c, c, FFT_FORWARD);
        Kokkos::fence();

        auto c = this->c;
        int N = Ngrid;
        double invN3 = 1.0 / ((double)N*N*N);
        double tx = 2.0*M_PI/L[0], ty = 2.0*M_PI/L[1], tz = 2.0*M_PI/L[2];
        double D1 = D;
        
        auto wk = this->wk;

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
                    c(i,j,k) *= -invN3 * wk(i,j,k) / (D1*k2);
                }
            });
        Kokkos::fence();

        FFT3DTransform(plan, c, c, FFT_BACKWARD);
        Kokkos::fence();
    }

};

} // namespace ExaDiS

#endif