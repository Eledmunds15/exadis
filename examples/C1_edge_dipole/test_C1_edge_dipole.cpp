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

/*---------------------------------------------------------------------------
 *
 *    Function:     get_force_fft
 *                  Return the long-range FFT module holding the stress grid
 *
 *-------------------------------------------------------------------------*/
ForceFFT* get_force_fft(Force* force)
{
    return static_cast<ForceType::DDD_FFT_MODEL*>(force)->get_force2()->get_flong();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     write_sigmah
 *                  Write sigma_h = tr(sigma)/3 from the FFT stress grid, as a
 *                  legacy VTK file and as a line cut along y.
 *                  Grid values sit at voxel centres (k+0.5)*L/N.
 *
 *-------------------------------------------------------------------------*/
void write_sigmah(System* system, Force* force, int Ngrid, double xcut,
                  std::string outputdir, int step)
{
    ForceFFT* fft = get_force_fft(force);
    std::vector<Mat33> stress = fft->export_stress_gridval(); // kx slowest, kz fastest
    auto idx = [&](int kx, int ky, int kz) { return (kx*Ngrid + ky)*Ngrid + kz; };

    Cell cell = system->get_serial_network()->cell;
    double h = cell.H.xx() / Ngrid; // cubic box: same spacing in x, y, z
    Vec3 origin = cell.origin + Vec3(0.5*h);

    // VTK: STRUCTURED_POINTS expects x fastest, then y, then z
    std::string vtkfile = outputdir + "/sigmah." + std::to_string(step) + ".vtk";
    FILE* fp = fopen(vtkfile.c_str(), "w");
    if (fp == NULL) ExaDiS_fatal("Error: cannot open %s\n", vtkfile.c_str());
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "sigma_h from ExaDiS ForceFFT (Pa)\n");
    fprintf(fp, "ASCII\nDATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", Ngrid, Ngrid, Ngrid);
    fprintf(fp, "ORIGIN %e %e %e\n", origin.x, origin.y, origin.z);
    fprintf(fp, "SPACING %e %e %e\n", h, h, h);
    fprintf(fp, "POINT_DATA %d\n", Ngrid*Ngrid*Ngrid);
    fprintf(fp, "SCALARS sigma_h float 1\nLOOKUP_TABLE default\n");
    for (int kz = 0; kz < Ngrid; kz++)
        for (int ky = 0; ky < Ngrid; ky++)
            for (int kx = 0; kx < Ngrid; kx++)
                fprintf(fp, "%e\n", stress[idx(kx, ky, kz)].trace() / 3.0);
    fclose(fp);

    // Line cut along y through the voxel column nearest x = xcut (z = mid-plane;
    // the field is independent of z for straight lines along z)
    int kxcut = (int)floor((xcut - cell.origin.x) / h);
    int kzcut = Ngrid / 2;
    std::string cutfile = outputdir + "/sigmah_cut." + std::to_string(step) + ".dat";
    fp = fopen(cutfile.c_str(), "w");
    if (fp == NULL) ExaDiS_fatal("Error: cannot open %s\n", cutfile.c_str());
    fprintf(fp, "# x y z sigma_h(Pa)   positions in units of b, voxel centres\n");
    for (int ky = 0; ky < Ngrid; ky++) {
        Vec3 p = origin + h*Vec3(kxcut, ky, kzcut);
        fprintf(fp, "%e %e %e %e\n", p.x, p.y, p.z, stress[idx(kxcut, ky, kzcut)].trace() / 3.0);
    }
    fclose(fp);

    ExaDiS_log("Wrote %s and %s\n", vtkfile.c_str(), cutfile.c_str());
}

void write_network_vtk(System* system, std::string filename)
{
    SerialDisNet* net = system->get_serial_network();
    int ns = net->segs.size();

    FILE* fp = fopen(filename.c_str(), "w");
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "ExaDiS network\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET POLYDATA\n");

    fprintf(fp, "POINTS %d double\n", 2*ns);
    for (int i = 0; i < ns; i++) {
        Vec3 r1 = net->nodes[net->segs[i].n1].pos;
        Vec3 r2 = net->cell.pbc_position(r1, net->nodes[net->segs[i].n2].pos);
        fprintf(fp, "%f %f %f\n%f %f %f\n", r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);
    }

    fprintf(fp, "LINES %d %d\n", ns, 3*ns);
    for (int i = 0; i < ns; i++)
        fprintf(fp, "2 %d %d\n", 2*i, 2*i + 1);

    // Burgers vector per segment, so you can colour the lines by it
    fprintf(fp, "CELL_DATA %d\n", ns);
    fprintf(fp, "VECTORS burgers double\n");
    for (int i = 0; i < ns; i++) {
        Vec3 b = net->segs[i].burg;
        fprintf(fp, "%f %f %f\n", b.x, b.y, b.z);
    }
    fclose(fp);
}

class C1App : public ExaDiSApp {
public:
    using ExaDiSApp::ExaDiSApp;                       // reuse the (argc, argv) constructor
    void output(Control& ctrl) override {
        ExaDiSApp::output(ctrl);                      // the usual .data / restart files
        if (istep % ctrl.outfreq == 0)
            write_network_vtk(system, outputdir + "/network." + std::to_string(istep) + ".vtk");
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     test_C1_edge_dipole
 *
 *-------------------------------------------------------------------------*/
void test_C1_edge_dipole(ExaDiSApp* exadis)
{
    // Simulation parameters (alpha-Fe, isotropic)
    double burgmag = 2.48e-10;
    double MU = 82e9;
    double NU = 0.29;
    double a = 1.0;
    double Mob = 1.0;
    int Ngrid = 64;

    double Lbox = 1000.0;
    double maxseg = 0.04*Lbox;
    double minseg = 0.01*Lbox;
    double dt = 1.0e-8;
    double rann = 2.0;
    int nsteps = 2000;

    ExaDiSApp::Control ctrl;
    ctrl.nsteps = nsteps;
    ctrl.loading = ExaDiSApp::STRESS_CONTROL;
    ctrl.appstress = Mat33().zero();
    ctrl.printfreq = 10;
    ctrl.outfreq = 10;
    std::string outputdir = "output_C1_edge_dipole";

    // Initialization: edge dipole, both lines along +z, opposite b
    SerialDisNet* config = new SerialDisNet(Lbox); // fully periodic, origin at 0
    Vec3 ldir(0.0, 0.0, 1.0);
    Vec3 plane(0.0, 1.0, 0.0);
    
    double d = 0.1*Lbox;
    Vec3 p1(0.5*Lbox, 0.5*Lbox - 0.5*d, 0.0);
    Vec3 p2(0.5*Lbox, 0.5*Lbox + 0.5*d, 0.0);

    insert_infinite_line(config, Vec3( 1.0, 0.0, 0.0), plane, ldir, p1, Mat33().eye(), maxseg);
    insert_infinite_line(config, Vec3(-1.0, 0.0, 0.0), plane, ldir, p2, Mat33().eye(), maxseg);

    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.nextdt = dt;
    params.rann = rann;

    System* system = exadis->system;
    system->initialize(params, Crystal(), config);

    // Modules. Force and mobility are dereferenced inside device kernels, so
    // they must live in unified memory (exadis_new), not plain host memory.
    exadis->force = exadis_new<ForceType::DDD_FFT_MODEL>(system,
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid)
    );
    exadis->diffusion = new Diffusion(get_force_fft(exadis->force), Ngrid);
    exadis->mobility = exadis_new<MobilityType::GLIDE>(system, MobilityType::GLIDE::Params(Mob));
    exadis->integrator = new IntegratorEuler(system);
    exadis->collision = new CollisionRetroactive(system);
    exadis->topology = new TopologySerial(system, exadis->force, exadis->mobility);
    exadis->remesh = new RemeshSerial(system);

    // Simulation setup (creates the output directory)
    exadis->outputdir = outputdir;
    exadis->set_simulation();

    // Stress grid for the initial configuration
    exadis->force->pre_compute(system);
    write_sigmah(system, exadis->force, Ngrid, p1.x, outputdir, 0);

    // Simulation: the dipole is in glide equilibrium, so it should barely move
    exadis->run(ctrl);

    // Stress grid from the last step's pre_compute
    write_sigmah(system, exadis->force, Ngrid, p1.x, outputdir, nsteps);

    // Free unified-memory modules here: ~ExaDiSApp() uses plain delete
    exadis_delete(exadis->force); exadis->force = nullptr;
    exadis_delete(exadis->mobility); exadis->mobility = nullptr;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);

    C1App exadis(argc, argv);
    test_C1_edge_dipole(&exadis);

    return 0;
}
