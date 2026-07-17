/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the CutE-FSI authors
 *
 * This file is part of CutE-FSI.
 *
 * -----------------------------------------------------------------------------
 *
 * Authors: Stefan Frei, University of Konstanz, 2026
 *          Tobias Knoke, Leibniz University Hannover, 2026
 *          Marc Steinbach, Leibniz University Hannover, 2026
 *          Anne-Kathrin Wenske, Leibniz University Hannover, 2026
 *          Thomas Wick, Leibniz University hannover, 2026
 *
 * Contributers: Marc Fehling, Charles University, Prague, 2026
 *
 * -----------------------------------------------------------------------------
 * Contact:
 * Anne-Kathrin Wenske
 * Leibniz Universität Hannover (LUH)
 * Institut für Angewandte Mathematik (IfAM)
 * E-mail: wenske@ifam.uni-hannover.de
 */

#include "include/cute-fsi.h"
#include "include/parameters.h"

int main(int argc, char **argv)
{
  try
  {
    dealii::deallog.depth_console(0);
    const unsigned int dim = 2;
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const Parameters::AllParameters parameters("cute-fsi.prm");
    Stokes::StokesFSI<dim> stokes_fsi(parameters.fe_degree_fluid,
                                      parameters.fe_degree_solid,
                                      parameters);
    stokes_fsi.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;

}
