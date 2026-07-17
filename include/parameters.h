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

#pragma once
#ifndef INCLUDE_PARAMETERS_H
#define INCLUDE_PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace Parameters
{
  struct GlobalParameters
  {
    unsigned int fe_degree_fluid;
    unsigned int fe_degree_solid;
    unsigned int n_refinements;
    unsigned int n_refinement_cycles;
    bool         do_spatial_analysis;
    bool         do_temporal_analysis;
    unsigned int output_skip;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

  };

  struct PhysicalConstants
  {
    double v_f_in;
    double nu_f;
    double rho_f;
    double rho_s;
    double mu;
    double lambda;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  struct Time
  {
    double end_time;
    double timestep_size;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  struct GhostParameters
  {
    double ghost_prm_v_f;
    double ghost_prm_v_s;
    double ghost_prm_p;
    double ghost_prm_u_v;
    double max_ghost_weight;
    double nitsche_parameter;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  struct AllParameters
      : public GlobalParameters,
      public PhysicalConstants,
      public Time,
      public GhostParameters
  {
    AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

} // namespace Parameters

#endif // INCLUDE_PARAMETERS_H
