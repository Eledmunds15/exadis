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

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Diffusion
 *
 *-------------------------------------------------------------------------*/
class Diffusion {
public:

    void compute(System* system) { 
        
        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].start();

        printf("diffusion comput\n"); 
    
        Kokkos::fence();
        system->timer[system->TIMER_DIFFUSION].stop();

    }
};

} // namespace ExaDiS

#endif