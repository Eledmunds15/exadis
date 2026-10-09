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

        SerialDisNet* net = system->get_serial_network();
        
        double Vs = cell.H.det() / ((double)Ngrid*Ngrid*Ngrid);

        deposit_network_serial(net, 1.0);
        auto c_serial = snapshot();
        deposit_network(system->get_device_network(), 1.0);
        auto c_par = snapshot();
        
        double maxdiff = 0.0;
        for (int i = 0; i < Ngrid; i++)
        for (int j = 0; j < Ngrid; j++)
        for (int k = 0; k < Ngrid; k++)
            maxdiff = fmax(maxdiff, fabs(c_serial(i,j,k).real() - c_par(i,j,k).real()));
        printf("max voxel difference %.3e\n", maxdiff);

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
            double W = scale * ForceFFT::alpha_box_segment(r1, t, Ls, bc, Hs);

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

    void deposit_network_serial(SerialDisNet* net, double hval) {
        auto h_s = Kokkos::create_mirror_view(c);
        Kokkos::deep_copy(h_s, complex(0.0, 0.0));

        for (int i = 0; i < net->segs.size(); i++) {
            Vec3 r1 = net->nodes[net->segs[i].n1].pos;
            Vec3 r2 = net->nodes[net->segs[i].n2].pos;
            deposit_segment(h_s, r1, r2, hval);
        }
        Kokkos::deep_copy(c, h_s);
    }

    double total_length(SerialDisNet* net) {
        double sum = 0.0;
        for (int i = 0; i < net->segs.size(); i++) {
            Vec3 r1 = net->nodes[net->segs[i].n1].pos;
            Vec3 r2 = net->nodes[net->segs[i].n2].pos;
            r2 = cell.pbc_position(r1, r2);
            sum += (r2 - r1).norm();
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    static void deposit_one(const T_grid& c, const Cell& cell, int N, Vec3 r1, Vec3 r2, double hval) {
        Vec3 Hs(1.0/N, 1.0/N, 1.0/N);
        double Vs = cell.H.det() / ((double)N*N*N);

        r2 = cell.pbc_position(r1, r2);
        Vec3 t = r2 - r1;
        double len = t.norm();
        if (len < 1e-10) return;

        r1 = cell.scaled_position(r1);
        r2 = cell.scaled_position(r2);

        double Ls = (r2 - r1).norm();
        double scale = len / Ls;
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
            double W = scale * ForceFFT::alpha_box_segment(r1, t, Ls, bc, Hs);

            int kx = ib % N; if (kx < 0) kx += N;
            int ky = jb % N; if (ky < 0) ky += N;
            int kz = kb % N; if (kz < 0) kz += N;

            Kokkos::atomic_add(&c(kx, ky, kz).real(), W/Vs * hval);
        }
    }

    void deposit_network(DeviceDisNet* net, double hval) {
        Kokkos::deep_copy(c, complex(0.0, 0.0));

        auto c      = this->c;
        Cell cell   = this->cell;
        int N       = Ngrid;
        auto nodes  = net->get_nodes();
        auto segs   = net->get_segs();

        Kokkos::parallel_for("Diffusion::DepositNetwork", net->Nsegs_local, KOKKOS_LAMBDA(const int i) {
            Vec3 r1 = nodes[segs[i].n1].pos;
            Vec3 r2 = nodes[segs[i].n2].pos;
            deposit_one(c, cell, N, r1, r2, hval);
        });
        Kokkos::fence();
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

    T_grid::HostMirror snapshot() {
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
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

        
        Vec3 o = cell.origin;
        double hh = L[0]/Ngrid;
        auto delta = [&](double r) {
            // grid point (i, m, 0): voxel centre, r = (i - m)*h from the line along x
            Vec3 x(o.x + (m + 0.5)*hh + r, o.y + (m + 0.5)*hh, o.z + 0.5*hh);
            return line_correction(x, lam, a, 8);
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

    double segment_correction(Vec3 x, Vec3 r1, Vec3 r2, double lam, double a) {
        r2 = cell.pbc_position(r1, r2);
        Vec3 mid = 0.5*(r1 + r2);
        Vec3 shift = cell.pbc_position(x, mid) - mid;
        r1 = r1 + shift;
        r2 = r2 + shift;

        Vec3 seg = r2 - r1;
        double len = seg.norm();
        if (len < 1e-10) return 0.0;
        Vec3 t = (1.0/len) * seg;

        double z1 = dot(r1 - x, t);
        double z2 = dot(r2 - x, t);
        double d2 = fmax((r1 - x).norm2() - z1*z1, 0.0);
        double d = sqrt(d2);
        double ba = sqrt(d2 + a*a);

        auto I1 = [](double z1, double z2, double b) {
            return asinh(z2/b) - asinh(z1/b);
        };

        auto I3 = [](double z1, double z2, double b) {
            return z2/(b*b*sqrt(z2*z2 + b*b)) - z1/(b*b*sqrt(z1*z1 + b*b));
        };

        return -lam/(4.0*M_PI*D)*(I1(z1, z2, d) - I1(z1, z2, ba) - 0.5*a*a*I3(z1, z2, ba));
    }

    double line_correction(Vec3 x, double lam, double a, int nseg) {
        Vec3 o = cell.origin;
        double h = L[0]/Ngrid;
        int m = Ngrid/2;
        double x0 = o.x + (m + 0.5)*h, y0 = o.y + (m + 0.5)*h;
        double sum = 0.0;

        for (int s = 0; s < nseg; s++) {
            Vec3 r1(x0, y0, o.z + s*L[2]/nseg);
            Vec3 r2(x0, y0, o.z + (s+1)*L[2]/nseg);
            sum += segment_correction(x, r1, r2, lam, a);
        }
        return sum;
    }

};

} // namespace ExaDiS

#endif