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
    double D = 1e8;
    FFTPlan<> plan;
    Kokkos::View<double***, Kokkos::LayoutRight, T_memory_space> wk;
    Cell cell;
    bool initialized = false;
    Kokkos::View<double*, T_memory_space> cnode;
    
    typedef Kokkos::View<double*, T_memory_space> T_nodeval;
    double burgmag = 2.48e-10;               // m
    double Omega   = 0.7698*pow(burgmag, 3); // atomic volume, m^3 (bcc: 0.7698 b^3)
    double kT      = 1.380649e-23*800.0;     // J (T = 800 K)
    double c0      = 1.0;                    // equilibrium vacancy fraction (scales v linearly)
    double rd      = 1.0;                    // core radius r_d, in b
    double a0      = sqrt(M_E)*rd;          // Cai core width = sqrt(e)*r_d  (decision #1)

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

    double dipole_series(Vec3 x, Vec3 x1, Vec3 x2, double lam, double a, int K) {
        double A = L[0]*L[1];
        double sum = 0.0;
        for (int ii = -K; ii <= K; ii++)
        for (int jj = -K; jj <= K; jj++) {
            if (ii == 0 && jj == 0) continue;            // k = 0 dropped, as in solve_poisson
            double kx = 2.0*M_PI*ii/L[0], ky = 2.0*M_PI*jj/L[1];
            double k2 = kx*kx + ky*ky;
            double ka = sqrt(k2)*a;
            double w  = 0.5*ka*ka*std::cyl_bessel_k(2.0, ka);   // same ŵ as init_spreading
            sum += w/k2 * ( cos(kx*(x.x - x1.x) + ky*(x.y - x1.y))
                          + cos(kx*(x.x - x2.x) + ky*(x.y - x2.y)) );
        }
        return -lam/(D*A) * sum;
    }

    void compute(System* system) {

        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].start();

        if (!initialized) initialize(system);

        // solve for h and the climb velocity at every node
        std::vector<Vec3> vcl; std::vector<double> hcl; int cit;
        climb_velocity(system, vcl, hcl, cit);

        // add climb on top of the glide velocity from mobility.
        // Re-fetch: apply_A switched the active copy, so make the serial one live before writing.
        SerialDisNet* net = system->get_serial_network();
        for (int i = 0; i < (int)vcl.size(); i++)
            net->nodes[i].v += vcl[i];

        // diagnostic: separation of the two C1 lines and climb speed
        double y_low = 0.0, y_high = 0.0, v_low = 0.0;
        int n_low = 0, n_high = 0;
        for (int i = 0; i < (int)net->nodes.size(); i++) {
            double y = net->nodes[i].pos.y;
            if (y < 0.5*L[1]) { y_low += y; v_low += vcl[i].y; n_low++; }
            else              { y_high += y; n_high++; }
        }
        if (n_low > 0 && n_high > 0)
            printf("climb: CG %d it   separation %.3f   v_y(lower) %.4e\n",
                   cit, y_high/n_high - y_low/n_low, v_low/n_low);

        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].stop();
    }


    void climb_velocity(System* system, std::vector<Vec3>& vcl, std::vector<double>& hsol, int& iters) {
        SerialDisNet* net = system->get_serial_network();
        int n = net->nodes.size();

        std::vector<Vec3> ecl; std::vector<double> be, fcl;
        climb_force(net, ecl, be, fcl);

        T_nodeval b ("Diffusion::rhs", n);
        auto hb = Kokkos::create_mirror_view(b);
        for (int i = 0; i < n; i++) {
            double g = (be[i] > 1e-10) ? fcl[i]/be[i] : 0.0;
            hb(i) = c0*(exp(-g*Omega/kT) - 1.0);
        }
        Kokkos::deep_copy(b, hb);

        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solve_cg(system, b, 100, 1e-10, iters));

        vcl.assign(n, Vec3(0.0));
        hsol.assign(n, 0.0);
        for (int i = 0; i < n; i++) {
            hsol[i] = h(i);
            if (be[i] > 1e-10) vcl[i] = (h(i)/be[i])*ecl[i];
        }

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

    KOKKOS_INLINE_FUNCTION
    static double interpolate_c(const T_grid& c, const Cell& cell, int N, const Vec3& p) {
        Vec3 s = cell.scaled_position(p);

        double  q[3] = { s.x*N - 0.5, s.y*N - 0.5, s.z*N - 0.5 };
        int     g[3];
        double  f[3];
        for (int d = 0; d < 3; d++) {
            g[d] = (int)floor(q[d]);
            f[d] = q[d] - g[d];
        }

        double sum = 0.0;
        for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
        for (int e = 0; e < 2; e++) {
            double w = (a ? f[0] : 1.0 - f[0])
                     * (b ? f[1] : 1.0 - f[1])
                     * (e ? f[2] : 1.0 - f[2]);
            
            int i = (g[0] + a) % N; if (i < 0) i += N;
            int j = (g[1] + b) % N; if (j < 0) j += N;
            int k = (g[2] + e) % N; if (k < 0) k += N;

            sum += w * c(i, j, k).real();
        }
        return sum;
    }

    double check_interp(int M) {
        auto c = this->c;
        Cell cell = this->cell;
        int N = Ngrid;
        double maxerr = 0.0;

        Kokkos::parallel_reduce("Diffusion::CheckInterp", M,
        KOKKOS_LAMBDA(const int n, double& err) {
            // spread M points around the box (fractional parts of irrational multiples)
            Vec3 s(fmod(n*0.6180339887, 1.0),
                   fmod(n*0.4142135624, 1.0),
                   fmod(n*0.7320508076, 1.0));
            Vec3 p = cell.real_position(s);

            double got   = interpolate_c(c, cell, N, p);
            double exact = sin(2.0*M_PI*s.z - M_PI/N);
            err = fmax(err, fabs(got - exact));
        }, Kokkos::Max<double>(maxerr));

        return maxerr;
    }

    void interpolate_nodes(DeviceDisNet* net) {
        Kokkos::resize(cnode, net->Nnodes_local);

        auto c     = this->c;
        auto cnode = this->cnode;
        Cell cell  = this->cell;
        int N      = Ngrid;
        auto nodes = net->get_nodes();

        Kokkos::parallel_for("Diffusion::InterpolateNodes", net->Nnodes_local,
        KOKKOS_LAMBDA(const int i) {
            cnode(i) = interpolate_c(c, cell, N, nodes[i].pos);
        });
        Kokkos::fence();
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
            double W = scale * diffusion_alpha_box_segment(r1, t, Ls, bc, Hs);

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
            double W = scale * diffusion_alpha_box_segment(r1, t, Ls, bc, Hs);

            int kx = ib % N; if (kx < 0) kx += N;
            int ky = jb % N; if (ky < 0) ky += N;
            int kz = kb % N; if (kz < 0) kz += N;

            Kokkos::atomic_add(&c(kx, ky, kz).real(), W/Vs * hval);
        }
    }

    void deposit_network(DeviceDisNet* net, T_nodeval hnode) {
        Kokkos::deep_copy(c, complex(0.0, 0.0));

        auto c      = this->c;
        Cell cell   = this->cell;
        int N       = Ngrid;
        auto nodes  = net->get_nodes();
        auto segs   = net->get_segs();

        Kokkos::parallel_for("Diffusion::DepositNetwork", net->Nsegs_local, KOKKOS_LAMBDA(const int i) {
            Vec3 r1 = nodes[segs[i].n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[segs[i].n2].pos);
            Vec3 mid = 0.5*(r1 + r2);
            deposit_one(c, cell, N, r1, mid, hnode(segs[i].n1));
            deposit_one(c, cell, N, mid, r2, hnode(segs[i].n2));
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

    // Copied from ForceFFT::alpha_box_segment (force_fft.h)
    KOKKOS_INLINE_FUNCTION
    static double diffusion_alpha_box_segment(const Vec3 &p1, const Vec3 &t, const double &L,
                             const Vec3 &bc, const Vec3 &H)
    {
        double eps = 1e-10;
        Vec3 tinv(1.0/(t.x+eps), 1.0/(t.y+eps), 1.0/(t.z+eps));
        
        Vec3 t1 = bc - H - p1;
        t1.x *= tinv.x; t1.y *= tinv.y; t1.z *= tinv.z;
        Vec3 t2 = bc + H - p1;
        t2.x *= tinv.x; t2.y *= tinv.y; t2.z *= tinv.z;
        
        Vec3 tmin(fmin(t1.x, t2.x), fmin(t1.y, t2.y), fmin(t1.z, t2.z));
        Vec3 tmax(fmax(t1.x, t2.x), fmax(t1.y, t2.y), fmax(t1.z, t2.z));
        
        double cmin = fmax(fmax(tmin.x, tmin.y), tmin.z);
        double cmax = fmin(fmin(tmax.x, tmax.y), tmax.z);
        
        cmin = fmax(cmin, 0.0);
        cmax = fmin(cmax, L);
        
        Vec3 x1 = p1 + cmin * t;
        Vec3 x2 = p1 + cmax * t;
        double lx = (x2-x1).norm();
        
        double W = 0.0;
        if (cmin <= cmax && lx >= eps) {
            // Parametrize segment
            Vec3 R = bc-x1;
            double dr = dot(R, t);
            Vec3 drt = dr * t;
            Vec3 d = R-drt;
            Vec3 x0 = x1+drt;
            double s1 = dot(x1-x0, t);
            double s2 = dot(x2-x0, t);
            
            Vec3 s;
            s.x = (fabs(t.x) < eps) ? s2+1 : d.x/t.x;
            s.y = (fabs(t.y) < eps) ? s2+1 : d.y/t.y;
            s.z = (fabs(t.z) < eps) ? s2+1 : d.z/t.z;
            
            Vec3 sk;
            sk.x = fmin(fmax(s.x, s1), s2);
            sk.y = fmin(fmax(s.y, s1), s2);
            sk.z = fmin(fmax(s.z, s1), s2);
            
            // First term
            W = s2-s1;
            
            // Second term Ai
            for (int k = 0; k < 3; k++) {
                W -= 1.0/H[k]*fabs(0.5*(sk[k]-s1)*(2.0*d[k]-t[k]*(sk[k]+s1)));
                W -= 1.0/H[k]*fabs(0.5*(s2-sk[k])*(2.0*d[k]-t[k]*(s2+sk[k])));
            }
    
            // Third term Bij
            for (int k = 0; k < 3; k++) {
                int i1 = k;
                int i2 = (k+1) % 3;
                double sm1 = fmin(sk[i1], sk[i2]);
                double sm2 = fmax(sk[i1], sk[i2]);
                double ss[4] = {s1, sm1, sm2, s2};
                double B = 0.0;
                for (int l = 0; l < 3; l++) {
                    double ss1 = ss[l];
                    double ss2 = ss[l+1];
                    B += fabs(d[i1]*d[i2]*(ss2-ss1)
                    -0.5*(d[i1]*t[i2]+d[i2]*t[i1])*(ss2*ss2-ss1*ss1)
                    +1.0/3.0*t[i1]*t[i2]*(ss2*ss2*ss2-ss1*ss1*ss1));
                }
                W += 1.0/H[i1]/H[i2]*B;
            }
            
            // Forth term Cijk
            double sm1 = fmin(sk[0], sk[1]);
            double sm2 = fmax(sk[0], sk[1]);
            double sn1 = fmin(sm1, sk[2]);
            double si2 = fmax(sm1, sk[2]);
            double sn2 = fmin(sm2, si2);
            double sn3 = fmax(sm2, si2);
            double ss[5] = {s1, sn1, sn2, sn3, s2};
            double C = 0.0;
            for (int l = 0; l < 4; l++) {
                double ss1 = ss[l];
                double ss2 = ss[l+1];
                C += fabs(d[0]*d[1]*d[2]*(ss2-ss1)
                -0.5*(t[0]*d[1]*d[2]+d[0]*t[1]*d[2]+d[0]*d[1]*t[2])*(ss2*ss2-ss1*ss1)
                +1.0/3.0*(d[0]*t[1]*t[2]+t[0]*d[1]*t[2]+t[0]*t[1]*d[2])*(ss2*ss2*ss2-ss1*ss1*ss1)
                -0.25*(t[0]*t[1]*t[2])*(ss2*ss2*ss2*ss2-ss1*ss1*ss1*ss1));
            }
            W -= 1.0/H[0]/H[1]/H[2]*C;
        }
        
        return W;
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
        auto h_c = Kokkos::create_mirror_view(c);
        Kokkos::deep_copy(h_c, c);
        return h_c;
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

    double segment_correction(Vec3 x, Vec3 r1, Vec3 r2, double lam, double a, double a0 = 0.0) {
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
        double b0 = sqrt(d2 + a0*a0);

        auto I1 = [](double z1, double z2, double b) {
            return asinh(z2/b) - asinh(z1/b);
        };

        auto I3 = [](double z1, double z2, double b) {
            return z2/(b*b*sqrt(z2*z2 + b*b)) - z1/(b*b*sqrt(z1*z1 + b*b));
        };

        return -lam/(4.0*M_PI*D)*( I1(z1, z2, b0) + 0.5*a0*a0*I3(z1, z2, b0) - I1(z1, z2, ba) - 0.5*a*a*I3(z1, z2, ba) );
    }

    double node_correction(SerialDisNet* net, T_nodeval::HostMirror hn, Vec3 x, double a, double a0) {
        double sum = 0.0;
        for (int i = 0; i < (int)net->segs.size(); i++) {
            int n1 = net->segs[i].n1, n2 = net->segs[i].n2;
            Vec3 r1 = net->nodes[n1].pos;
            Vec3 r2 = cell.pbc_position(r1, net->nodes[n2].pos);
            Vec3 mid = 0.5*(r1 + r2);
            sum += segment_correction(x, r1,  mid, hn(n1), a, a0);
            sum += segment_correction(x, mid, r2,  hn(n2), a, a0);
        }
        return sum;
    }

        T_nodeval apply_A(System* system, T_nodeval hnode) {
        DeviceDisNet* d_net = system->get_device_network();
        SerialDisNet* net   = system->get_serial_network();

        deposit_network(d_net, hnode);
        solve_poisson();
        interpolate_nodes(d_net);                 // grid part → cnode

        // short-range part, on the host for now
        auto hn = Kokkos::create_mirror_view(hnode);
        Kokkos::deep_copy(hn, hnode);
        auto cn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cnode);
        double a = 2.0*L[0]/Ngrid;
        for (int i = 0; i < (int)net->nodes.size(); i++)
            cn(i) += node_correction(net, hn, net->nodes[i].pos, a, a0);

        T_nodeval out("Diffusion::Ah", cn.extent(0));
        Kokkos::deep_copy(out, cn);
        return out;
    }

    T_nodeval solve_cg(System* system, T_nodeval b, int maxit, double tol,int& iters) {
        int n = b.extent(0);
        auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

        std::vector<double> x(n, 0.0), r(n), p(n);
        for (int i = 0; i < n; i++) { r[i] = -hb(i); p[i] = r[i]; }

        auto dot = [&](const std::vector<double>& u, const std::vector<double>& v) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += u[i]*v[i];
            return s;
        };
        double rr = dot(r, r), rr0 = rr;

        T_nodeval pv("Diffusion::p", n);
        auto hp = Kokkos::create_mirror_view(pv);

        for (iters = 0; iters < maxit && sqrt(rr/rr0) > tol; iters++) {
            for (int i =0; i < n; i++) hp(i) = p[i];
            Kokkos::deep_copy(pv, hp);
            auto Ap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), apply_A(system, pv));

            double pAp = 0.0;
            for (int i = 0; i < n; i++) pAp += p[i]*(-Ap(i));
            double alpha = rr/pAp;

            for (int i = 0; i < n; i++) {
                x[i] += alpha*p[i];
                r[i] -= alpha*(-Ap(i));
            }

            double rr_new = dot(r, r);
            double beta = rr_new/rr;
            for (int i = 0; i < n; i++) p[i] = r[i] + beta*p[i];
            rr = rr_new;
        }

        T_nodeval xv("Diffusion::h", n);
        auto hx = Kokkos::create_mirror_view(xv);
        for (int i = 0; i < n; i++) hx(i) = x[i];
        Kokkos::deep_copy(xv, hx);
        return xv;
    }

    void climb_force(SerialDisNet* net, std::vector<Vec3>& ecl, std::vector<double>& be, std::vector<double>& fcl) {
        int n = net->nodes.size();
        std::vector<double> Lh(n, 0.0);
        ecl.assign(n, Vec3(0.0));
        be.assign(n, 0.0);
        fcl.assign(n, 0.0);

        for (int i = 0; i < (int)net->segs.size(); i++) {
            int n1 = net->segs[i].n1, n2 = net->segs[i].n2;
            Vec3 r1 = net->nodes[n1].pos;
            Vec3 r2 = cell.pbc_position(r1, net->nodes[n2].pos);
            Vec3 t = r2 - r1;
            double len = t.norm();
            if(len < 1e-10) continue;
            Vec3 cb = cross((1.0/len)*t, net->segs[i].burg);

            for (int n_ : {n1, n2}) {
                Lh[n_]      += 0.5*len;
                ecl[n_]     += 0.5*len*cb;
                be[n_]      += 0.5*len*cb.norm();
            }
        }

        for (int i = 0; i < n; i++) {
            if (Lh[i] <= 0.0) continue;
            be[i] /= Lh[i];
            double en = ecl[i].norm();
            if (en > 1e-10) {
                ecl[i] = (1.0/en)*ecl[i];
                fcl[i] = dot(net->nodes[i].f, ecl[i]) / Lh[i];
            } else {
                ecl[i] = Vec3(0.0);
            }
        }
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