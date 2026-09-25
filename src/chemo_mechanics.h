/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Ethan L. Edmunds
 *	eledmunds1@sheffield.ac.uk
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CHEMO_MECHANICS_H
#define EXADIS_CHEMO_MECHANICS_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        ChemoMechanics
 *
 *-------------------------------------------------------------------------*/
class ChemoMechanics {
public:
    ChemoMechanics() {}
    ChemoMechanics(System *system) {}
    virtual void step(System *system) = 0;
    virtual ~ChemoMechanics() {}
    virtual const char* name() { return "ChemoMechanicsNone"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ChemoMechanicsVacancyDiffusion
 *                  Base class for chemo-mechanical model driven by stress fields 
 *                  extracted from DDD
 *
 *-------------------------------------------------------------------------*/
class ChemoMechanicsVacancyDiffusion : public ChemoMechanics {
public:
    std::vector<int> Ngrid;
    std::string outputdir = ".";
    int compute_freq = 10; // only actually build/update the field every N calls to step()
    int write_freq = 10;   // write a .vtk file every N calls to step() (should be a multiple of compute_freq)
    int step_count = 0;

    double D = 1.0;         // vacancy diffusion coefficient (placeholder value/units for now)
    double dt_diff = 100.0; // diffusion internal timestep (explicit Euler - must satisfy dt <= dx^2/(6D))
    double c0 = 1.0;        // initial (uniform) normalized vacancy concentration

    // Stress coupling: vacancy relaxation volume and temperature.
    // dOmega < 0 for vacancies (they drift toward compressive regions).
    // Default: alpha-Fe, atomic volume a^3/2 with a = 2.86e-10 m, dOmega = -0.3*Omega_atom.
    double Omega_atom = 0.5*2.86e-10*2.86e-10*2.86e-10; // m^3
    double dOmega = -0.3*Omega_atom;                     // m^3
    double T = 800.0;                                    // K
    static constexpr double kB = 1.380649e-23;           // J/K

    // Persistent concentration field - unlike the stress field (rebuilt from
    // the network every call), this is real evolving state that must survive
    // between calls to step().
    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace> concentration;

    ChemoMechanicsVacancyDiffusion(System* system, std::vector<int> _Ngrid, std::string _outputdir = ".")
        : Ngrid(_Ngrid), outputdir(_outputdir)
    {
        Kokkos::resize(concentration, Ngrid[0], Ngrid[1], Ngrid[2]);
        // Uniform initial condition: pure diffusion leaves this unchanged,
        // so any structure that develops is due to the stress-driven drift.
        Kokkos::deep_copy(concentration, c0);
    }

    // Explicit-Euler update of the stress-assisted diffusion equation
    //   dc/dt = -div(J),  J = -D*grad(c) + D*c*(dOmega/kT)*grad(sigma_h)
    // with sigma_h = tr(sigma)/3 (tension positive). Discretized in
    // conservative flux form on cell faces, so the total number of
    // vacancies is conserved exactly under periodic boundary conditions.
    // Note: assumes a fully periodic Cell (wraps indices at the boundary).
    template<class SF>
    void diffusion_step(Cell& cell, SF& sfield)
    {
        int Nx = Ngrid[0], Ny = Ngrid[1], Nz = Ngrid[2];
        double dx = cell.H.xx()/Nx, dy = cell.H.yy()/Ny, dz = cell.H.zz()/Nz;

        // Hydrostatic stress on the grid (Pa)
        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace> sh;
        Kokkos::resize(sh, Nx, Ny, Nz);
        auto sgrid = sfield.gridval;
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx, Ny, Nz}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                sh(i,j,k) = sgrid(i,j,k).trace() / 3.0;
            });
        Kokkos::fence();

        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace> c_new;
        Kokkos::resize(c_new, Nx, Ny, Nz);

        auto c = concentration;
        double Dloc = D, dtloc = dt_diff;
        double beta = dOmega / (kB*T); // 1/Pa

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx, Ny, Nz}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int ip = (i+1)%Nx, im = (i-1+Nx)%Nx;
                int jp = (j+1)%Ny, jm = (j-1+Ny)%Ny;
                int kp = (k+1)%Nz, km = (k-1+Nz)%Nz;

                // Flux-divergence form: sum over faces of D*(grad c - c_face*beta*grad sigma_h)
                auto face = [&](double cn, double sn, double h) {
                    double cf = 0.5*(c(i,j,k) + cn);
                    return Dloc * ((cn - c(i,j,k)) - cf*beta*(sn - sh(i,j,k))) / (h*h);
                };

                double dcdt = face(c(ip,j,k), sh(ip,j,k), dx) + face(c(im,j,k), sh(im,j,k), dx)
                            + face(c(i,jp,k), sh(i,jp,k), dy) + face(c(i,jm,k), sh(i,jm,k), dy)
                            + face(c(i,j,kp), sh(i,j,kp), dz) + face(c(i,j,km), sh(i,j,km), dz);

                c_new(i,j,k) = c(i,j,k) + dtloc * dcdt;
            });
        Kokkos::fence();

        Kokkos::deep_copy(concentration, c_new);
    }

    double total_concentration()
    {
        double sum = 0.0;
        auto c = concentration;
        Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Ngrid[0], Ngrid[1], Ngrid[2]}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& s) {
                s += c(i,j,k);
            }, sum);
        return sum;
    }

    void step(System* system) {
        ExaDiS_log("ChemoMechanics::step()\n");

        if (step_count % compute_freq == 0) {
            Kokkos::fence();
            system->timer[system->TIMER_STRESSFIELD].start();

            auto net = system->get_device_network();
            fields::StressFieldGrid<DeviceDisNet> sfield(
                net,
                fields::StressIso::Params(system->params.MU, system->params.NU, system->params.a),
                Ngrid
            );

            Kokkos::fence();
            system->timer[system->TIMER_STRESSFIELD].stop();

            Kokkos::fence();
            system->timer[system->TIMER_CHEMOMECH].start();

            diffusion_step(net->cell, sfield);
            ExaDiS_log("ChemoMechanics: total concentration = %.12e\n", total_concentration());

            if (step_count % write_freq == 0) {
                std::string filename = outputdir + "/chemomech." + std::to_string(step_count) + ".vtk";
                FILE* fp = fopen(filename.c_str(), "w");
                if (fp == NULL)
                    ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());

                fields::write_vtk_header(fp, sfield.cell, Ngrid);

                fields::write_vtk_scalar(fp, "Sxx", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).xx(); });
                fields::write_vtk_scalar(fp, "Syy", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).yy(); });
                fields::write_vtk_scalar(fp, "Szz", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).zz(); });
                fields::write_vtk_scalar(fp, "Syz", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).yz(); });
                fields::write_vtk_scalar(fp, "Sxz", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).xz(); });
                fields::write_vtk_scalar(fp, "Sxy", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).xy(); });
                fields::write_vtk_scalar(fp, "SigmaH", Ngrid, [&](int kx, int ky, int kz) { return sfield.gridval(kx,ky,kz).trace()/3.0; });

                auto c = concentration;
                fields::write_vtk_scalar(fp, "Concentration", Ngrid, [&](int kx, int ky, int kz) { return c(kx,ky,kz); });

                fclose(fp);
                ExaDiS_log("Writing chemo-mechanics VTK file: %s\n", filename.c_str());
            }

            system->timer[system->TIMER_CHEMOMECH].stop();
        }
        step_count++;
    }

    const char* name() { return "ChemoMechanicsVacancyDiffusion"; }
};

} // namespace ExaDiS

#endif
