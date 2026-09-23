/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Ethan L. Edmunds
 *	eledmunds1@sheffield.ac.uk
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_PHASE_FIELD_H
#define EXADIS_PHASE_FIELD_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        PhaseField
 *
 *-------------------------------------------------------------------------*/
class PhaseField {
public:
    PhaseField() {}
    PhaseField(System *system) {}
    virtual void step(System *system) = 0;
    virtual ~PhaseField() {}
    virtual const char* name() { return "PhaseFieldNone"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        PhaseFieldLocal
 *                  Base class for phase field model driven by stress fields 
 *                  extracted from DDD
 *
 *-------------------------------------------------------------------------*/
class PhaseFieldLocal : public PhaseField {
public:
    std::vector<int> Ngrid;
    std::string outputdir = ".";
    int compute_freq = 10; // only actually build/update the field every N calls to step()
    int write_freq = 10;   // write a .vtk file every N calls to step() (should be a multiple of compute_freq)
    int step_count = 0;

    double D = 1.0;       // vacancy diffusion coefficient (placeholder value/units for now)
    double dt_pf = 100.0; // phase field internal timestep (explicit Euler - must satisfy dt <= dx^2/(6D))

    // Persistent concentration field - unlike the stress field (rebuilt from
    // the network every call), this is real evolving state that must survive
    // between calls to step().
    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace> concentration;

    PhaseFieldLocal(System* system, std::vector<int> _Ngrid, std::string _outputdir = ".")
        : Ngrid(_Ngrid), outputdir(_outputdir)
    {
        Kokkos::resize(concentration, Ngrid[0], Ngrid[1], Ngrid[2]);

        // Initial condition: a Gaussian blob at the center of the grid (in
        // index space), just so diffusion is visibly spreading outward.
        // Not physical yet - purely to validate the update kernel works.
        double cx = 0.5*Ngrid[0], cy = 0.5*Ngrid[1], cz = 0.5*Ngrid[2];
        double sigma = 0.125*std::min({Ngrid[0], Ngrid[1], Ngrid[2]});
        auto c = concentration;
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Ngrid[0], Ngrid[1], Ngrid[2]}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                double r2 = (i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz);
                c(i,j,k) = exp(-r2/(2.0*sigma*sigma));
            });
        Kokkos::fence();
    }

    // Explicit-Euler update of the diffusion equation dc/dt = D*Laplacian(c)
    // Note: assumes a fully periodic Cell (wraps indices at the boundary).
    void diffusion_step(Cell& cell)
    {
        int Nx = Ngrid[0], Ny = Ngrid[1], Nz = Ngrid[2];
        double dx = cell.H.xx()/Nx, dy = cell.H.yy()/Ny, dz = cell.H.zz()/Nz;

        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace> c_new;
        Kokkos::resize(c_new, Nx, Ny, Nz);

        auto c = concentration;
        double Dloc = D, dtloc = dt_pf;

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx, Ny, Nz}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int ip = (i+1)%Nx, im = (i-1+Nx)%Nx;
                int jp = (j+1)%Ny, jm = (j-1+Ny)%Ny;
                int kp = (k+1)%Nz, km = (k-1+Nz)%Nz;

                double lap = (c(ip,j,k) - 2.0*c(i,j,k) + c(im,j,k)) / (dx*dx)
                           + (c(i,jp,k) - 2.0*c(i,j,k) + c(i,jm,k)) / (dy*dy)
                           + (c(i,j,kp) - 2.0*c(i,j,k) + c(i,j,km)) / (dz*dz);

                c_new(i,j,k) = c(i,j,k) + dtloc * Dloc * lap;
            });
        Kokkos::fence();

        Kokkos::deep_copy(concentration, c_new);
    }

    void step(System* system) {
        ExaDiS_log("PhaseField::step()\n");

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
            system->timer[system->TIMER_PHASEFIELD].start();

            diffusion_step(net->cell);

            if (step_count % write_freq == 0) {
                std::string filename = outputdir + "/phasefield." + std::to_string(step_count) + ".vtk";
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

                auto c = concentration;
                fields::write_vtk_scalar(fp, "Concentration", Ngrid, [&](int kx, int ky, int kz) { return c(kx,ky,kz); });

                fclose(fp);
                ExaDiS_log("Writing phase field VTK file: %s\n", filename.c_str());
            }

            system->timer[system->TIMER_PHASEFIELD].stop();
        }
        step_count++;
    }

    const char* name() { return "PhaseFieldLocal"; }
};

} // namespace ExaDiS

#endif
