/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Author: Ethan L. Edmunds
 *
 *  C2: circular prismatic loop, single climb solve with a prescribed uniform
 *  right-hand side. Reproduces Gu et al. (2015) Sec. 5.2 / Fig. 8:
 *      v_cl / (2 pi D / b) = 1/ln(8R/r_d)   (= 0.1355425 for R = 200 r_d)
 *
 *-------------------------------------------------------------------------*/

#include <iostream>

#include "exadis.h"
#include "driver.h"

using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Function:     insert_circular_prismatic_loop
 *                  N nodes on a circle of radius R in the xy plane, b = +z.
 *                  Parametrised as in Gu et al.: (R cos t, -R sin t, 0),
 *                  i.e. clockwise seen from +z.
 *
 *-------------------------------------------------------------------------*/
void insert_circular_prismatic_loop(SerialDisNet* network, double R, Vec3 center, int N)
{
    Vec3 burg(0.0, 0.0, 1.0);

    int istart = network->number_of_nodes();
    for (int i = 0; i < N; i++) {
        double t = 2.0*M_PI*i/N;
        Vec3 p = center + Vec3(R*cos(t), -R*sin(t), 0.0);
        network->add_node(network->cell.pbc_fold(p), UNCONSTRAINED);
    }

    // Each segment's plane contains its line direction and b (prismatic: varies per segment)
    for (int i = 0; i < N; i++) {
        int n1 = istart + i, n2 = istart + (i+1)%N;
        Vec3 t = network->nodes[n2].pos - network->nodes[n1].pos;
        Vec3 plane = cross(t, burg).normalized();
        network->add_seg(n1, n2, burg, plane);
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     print_aij
 *                  Recover the paper's coefficient matrix a_ij (Eq. 52-58)
 *                  from Diffusion::apply_A, column by column.
 *
 *-------------------------------------------------------------------------*/
void print_aij(System* system, Diffusion* diffusion, int N)
{
    // TODO(you): for each j, build a T_nodeval with h = 1 at node j and 0
    // elsewhere, call diffusion->apply_A(system, e_j), copy the result to the
    // host, and convert column j to a_ij (work out the -4 pi D factor and sign
    // from Eq. 22 and how solve_cg uses -Ap). Print the N x N matrix and
    // compare with Eq. 58 (N = 6, R = 200 r_d).
    printf("  a_ij: not implemented yet\n");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     run_case
 *                  Build one loop, solve once, report h and v_cl.
 *
 *-------------------------------------------------------------------------*/
struct CaseResult {
    double vnorm;   // mean(h)/b_e in units of 2 pi D / b
    double spread;  // max |h - mean| / |mean|
    int iters;
};

CaseResult run_case(double R, double Lbox, int N, int Ngrid, bool show_aij)
{
    // Units of b throughout; r_d = diffusion->rd = 1 b
    double burgmag = 2.48e-10;
    double MU = 82e9;
    double NU = 0.29;
    double a = 1.0;
    double seglen = 2.0*R*sin(M_PI/N);
    double maxseg = 2.0*seglen;  // no remeshing happens (we never step), just keep Params consistent
    double minseg = 0.25*seglen;

    SerialDisNet* config = new SerialDisNet(Lbox);
    insert_circular_prismatic_loop(config, R, Vec3(0.5*Lbox), N);

    Params params(burgmag, MU, NU, a, maxseg, minseg);
    System* system = make_system(config, Crystal(), params);

    // Diffusion only needs the FFT module to exist; no force is evaluated here
    Force* force = exadis_new<ForceType::DDD_FFT_MODEL>(system,
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid)
    );
    ForceFFT* fft = static_cast<ForceType::DDD_FFT_MODEL*>(force)->get_force2()->get_flong();
    Diffusion* diffusion = new Diffusion(fft, Ngrid);

    // Prescribed RHS: Delta c = c_inf - c0 exp(-g Omega/kT) = 1 at every node.
    // TODO(you): add these two members to Diffusion and use them in climb_velocity.
    diffusion->use_fixed_rhs = true;
    diffusion->fixed_rhs = -1.0;

    diffusion->initialize(system);  // climb_velocity does not do this itself

    std::vector<Vec3> vcl;
    std::vector<double> h;
    int iters;
    diffusion->climb_velocity(system, vcl, h, iters);

    double mean = 0.0;
    for (double hi : h) mean += hi;
    mean /= h.size();
    double spread = 0.0;
    for (double hi : h) spread = fmax(spread, fabs(hi - mean));
    spread /= fabs(mean);

    // Prismatic loop: b_e = |b| = 1, so v_cl = h. Normalise by 2 pi D / b.
    CaseResult res;
    res.vnorm = mean / (2.0*M_PI*diffusion->D);
    res.spread = spread;
    res.iters = iters;

    if (show_aij) print_aij(system, diffusion, N);

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

    int Ngrid = 64;
    double rd = 1.0;

    auto expect = [&](double R) { return 1.0/log(8.0*R/rd); };
    auto report = [&](const char* tag, double R, double L, int N, CaseResult r) {
        double e = expect(R);
        printf("%-6s R=%6.0f L=%6.0f N=%4d  v/(2piD/b)=% .7f  expect=%.7f  relerr=% .3e  spread=%.1e  CG=%d\n",
               tag, R, L, N, r.vnorm, e, (r.vnorm - e)/e, r.spread, r.iters);
    };

    // 1. Discretisation: error vs N (paper: slope -2, until the box/core floor)
    printf("\n== N sweep (R = 200, L = 4000) ==\n");
    for (int N : {6, 12, 25, 50, 100}) {
        CaseResult r = run_case(200.0, 4000.0, N, Ngrid, N == 6);
        report("N", 200.0, 4000.0, N, r);
    }

    // 2. Periodic box: error should scale like R/L (net source + images)
    printf("\n== L sweep (R = 200, N = 50) ==\n");
    for (double L : {2000.0, 4000.0, 8000.0}) {
        CaseResult r = run_case(200.0, L, 50, Ngrid, false);
        report("L", 200.0, L, 50, r);
    }

    // 3. Loop size: a constant core offset shows up as a drift vs 1/ln(8R/r_d)
    printf("\n== R sweep (L = 20 R, N = 50) ==\n");
    for (double R : {50.0, 100.0, 200.0, 400.0}) {
        CaseResult r = run_case(R, 20.0*R, 50, Ngrid, false);
        report("R", R, 20.0*R, 50, r);
    }

    return 0;
}
