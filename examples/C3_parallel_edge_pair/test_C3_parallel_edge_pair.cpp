/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Author: Ethan L. Edmunds
 *
 *
 *-------------------------------------------------------------------------*/

#include <iostream>

#include "exadis.h"
#include "driver.h"

using namespace ExaDiS;

ForceFFT* get_force_fft(Force* force)
{
    return static_cast<ForceType::DDD_FFT_MODEL*>(force)->get_force2()->get_flong();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_C1_edge_dipole
 *
 *-------------------------------------------------------------------------*/

struct CaseResult {
    double ratio;    // h1 * ln(r12/rd) / (-2 pi D rhs1): should be 1
    double antisym;  // (h1 + h2) / |h1|: should be ~0
    double g1, g1_ref; // climb force / b_e on line 1 (Pa), code vs Liu Eq. 4.1
    int iters;
};

CaseResult run_case(double Lbox, int Ngrid, double a0fac = 1)
{
    // Units of b throughout; r_d = diffusion->rd = 1 b
    double burgmag = 2.48e-10;
    double MU = 82e9;
    double NU = 0.29;
    double a = 1.0;
    double maxseg = 200.0;  // no remeshing happens (we never step), just keep Params consistent
    double minseg = 50.0;

    SerialDisNet* config = new SerialDisNet(Lbox);
    
    Vec3 ldir(0.0, 0.0, 1.0);
    Vec3 plane(0.0, 1.0, 0.0);
    
    Vec3 c(0.5*Lbox);
    Vec3 p1 = c - Vec3(150.0, 200.0, 0.0);
    Vec3 p2 = c + Vec3(150.0, 200.0, 0.0);

    insert_infinite_line(config, Vec3( 1.0, 0.0, 0.0), plane, ldir, p1, Mat33().eye(), maxseg);
    int n1 = config->number_of_nodes();
    
    insert_infinite_line(config, Vec3( 1.0, 0.0, 0.0), plane, ldir, p2, Mat33().eye(), maxseg);

    Params params(burgmag, MU, NU, a, maxseg, minseg);
    System* system = make_system(config, Crystal(), params);

    // Diffusion only needs the FFT module to exist; no force is evaluated here
    Force* force = exadis_new<ForceType::DDD_FFT_MODEL>(system,
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid)
    );
    ForceFFT* fft = static_cast<ForceType::DDD_FFT_MODEL*>(force)->get_force2()->get_flong();
    Diffusion* diffusion = new Diffusion(fft, Ngrid);

    diffusion->a0 *= a0fac;

    force->pre_compute(system);
    force->compute(system);

    diffusion->initialize(system);  // climb_velocity does not do this itself

    std::vector<Vec3> vcl;
    std::vector<double> h;
    int iters;
    diffusion->climb_velocity(system, vcl, h, iters);

    // Per-line average of h (constant along each straight line up to roundoff)
    int n = h.size();
    double h1 = 0.0, h2 = 0.0;
    for (int i = 0; i < n1; i++) h1 += h[i];
    for (int i = n1; i < n; i++) h2 += h[i];
    h1 /= n1;
    h2 /= (n - n1);

    // RHS on line 1, rebuilt exactly as climb_velocity does
    SerialDisNet* net = system->get_serial_network();
    std::vector<Vec3> ecl; std::vector<double> be, fcl;
    diffusion->climb_force(net, ecl, be, fcl);
    double g1 = fcl[0]/be[0];
    double rhs1 = diffusion->c0*(exp(-g1*diffusion->Omega/diffusion->kT) - 1.0);

    double g2 = fcl[n1]/be[n1];
    double rhs2 = diffusion->c0*(exp(-g2*diffusion->Omega/diffusion->kT) - 1.0);

    double h_a = 0.5*(h1 - h2);
    double rhs_a = 0.5*(rhs1 - rhs2);

    // Two-line infinite-medium solution: h1 = -2 pi D rhs1 / ln(r12/rd)
    double r12 = (p2 - p1).norm();
    double D = diffusion->D;

    // Liu Eq. 4.1, y-component, in Pa (lengths in b, so the b^2/b factors drop out)
    double x = p2.x - p1.x, y = p2.y - p1.y, r2 = x*x + y*y;
    double g1_ref = MU/(2.0*M_PI*(1.0 - NU)) * y*(3.0*x*x + y*y)/(r2*r2);

    CaseResult res;
    res.ratio   = h_a*log(r12/diffusion->rd) / (-2.0*M_PI*D*rhs_a);
    res.antisym = (h1 + h2)/fabs(h1);
    res.g1      = g1;
    res.g1_ref  = g1_ref;
    res.iters   = iters;

    delete diffusion;
    exadis_delete(force);
    exadis_delete(system);
    return res;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);

    printf("\n== L sweep, fixed spacing (r12 = 500, L/Ngrid = 62.5) ==\n");
    for (double L : {2000.0, 4000.0, 8000.0}) {
        int Ng = (int)(64 * L / 4000.0);
        CaseResult r = run_case(L, Ng);
        printf("L=%6.0f Ng=%4d  ratio=%.6f  antisym=% .2e  g1=% .4e  g1_ref=% .4e  CG=%d\n",
               L, Ng, r.ratio, r.antisym, r.g1, r.g1_ref, r.iters);
    }

    printf("\n== a0 sweep (L = 8000, Ng = 128) ==\n");
    for (double f : {0.607, 0.8, 0.9, 1.0, 1.1}) {
        CaseResult r = run_case(8000.0, 128, f);
        printf("a0/(sqrt(e) rd)=%.3f  ratio=%.6f\n", f, r.ratio);
    }    

    return 0;
}

