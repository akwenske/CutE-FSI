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

#include "../include/parameters.h"

using namespace dealii;

namespace Parameters
{
    void
    GlobalParameters::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Global parameters");
        {
            prm.declare_entry("fe_degree_fluid", "2",
                              Patterns::Integer(2,2),
                              "Fluid finite element degree m_f");
            prm.declare_entry("fe_degree_solid", "1",
                              Patterns::Integer(1,2),
                              "Solid finite element degree m_s");
            prm.declare_entry("n_refinements", "0",
                              Patterns::Integer(0),
                              "Number of global refinements");
            prm.declare_entry("n_refinement_cycles", "0",
                              Patterns::Integer(0),
                              "Number of refinment cycles");
            prm.declare_entry("do_spatial_analysis", "true",
                              Patterns::Bool(),
                              "Conduct convergence analysis in space");
            prm.declare_entry("do_temporal_analysis", "false",
                              Patterns::Bool(),
                              "Conduct convergence analysis in time");
            prm.declare_entry("output_skip", "1",
                              Patterns::Integer(1),
                              "Output skip for solution values");
        }
        prm.leave_subsection();
    }

    void
    GlobalParameters::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Global parameters");
        {
            fe_degree_fluid              = prm.get_integer("fe_degree_fluid");
            fe_degree_solid              = prm.get_integer("fe_degree_solid");
            n_refinements                = prm.get_integer("n_refinements");
            n_refinement_cycles          = prm.get_integer("n_refinement_cycles");
            do_spatial_analysis          = prm.get_bool("do_spatial_analysis");
            do_temporal_analysis         = prm.get_bool("do_temporal_analysis");
            output_skip                  = prm.get_integer("output_skip");
        }
        prm.leave_subsection();
    }

    void
    PhysicalConstants::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Physical constants");
        {
            prm.declare_entry("inflow_velocity", "0.0",
                              Patterns::Double(0),
                              "inflow velocity");
            prm.declare_entry("viscosity_fluid", "1.0",
                              Patterns::Double(0),
                              "fluid viscosity");
            prm.declare_entry("density_fluid", "1.0",
                              Patterns::Double(0),
                              "fluid density");
            prm.declare_entry("density_structure", "1.0",
                              Patterns::Double(0),
                              "structure density");
            prm.declare_entry("mu", "1.0",
                              Patterns::Double(0),
                              "mu");
            prm.declare_entry("lambda", "1.0",
                              Patterns::Double(0),
                              "lambda");
        }
        prm.leave_subsection();
    }

    void
    PhysicalConstants::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Physical constants");
      {
        v_f_in     = prm.get_double("inflow_velocity");
        nu_f       = prm.get_double("viscosity_fluid");
        rho_f      = prm.get_double("density_fluid");
        rho_s      = prm.get_double("density_structure");
        mu         = prm.get_double("mu");
        lambda     = prm.get_double("lambda");
      }
      prm.leave_subsection();
    }

    void
    Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Timestepping");
      {
        prm.declare_entry("end_time", "1.0",
                          Patterns::Double(0),
                          "end time");
        prm.declare_entry("timestep_size", "1.0e-2",
                          Patterns::Double(0),
                          "timestep size");
      }
      prm.leave_subsection();
    }

    void
    Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Timestepping");
      {
        end_time      = prm.get_double("end_time");
        timestep_size = prm.get_double("timestep_size");
      }
      prm.leave_subsection();
    }

    void
    GhostParameters::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Ghost Parameters");
      {
        prm.declare_entry("ghost_parameter_velocity_fluid", "1.0e-3",
                          Patterns::Double(0),
                          "Ghost parameter velocity fluid");
        prm.declare_entry("ghost_parameter_velocity_structure", "1.0e-3",
                          Patterns::Double(0),
                          "Ghost parameter velocity structure");
        prm.declare_entry("ghost_parameter_pressure", "1.0e-3",
                          Patterns::Double(0),
                          "Ghost parameter pressure");
        prm.declare_entry("ghost_parameter_displacement_velocity", "1.0e-3",
                          Patterns::Double(0),
                          "Ghost parameter displacement velocity");
        prm.declare_entry("max_ghost_weight", "3.0",
                          Patterns::Double(0),
                          "Max ghost weight");
        prm.declare_entry("nitsche_parameter", "10.0",
                          Patterns::Double(0),
                          "Nitsche parameter");
      }
      prm.leave_subsection();
    }

    void
    GhostParameters::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Ghost Parameters");
      {
        ghost_prm_v_f        = prm.get_double("ghost_parameter_velocity_fluid");
        ghost_prm_v_s        = prm.get_double("ghost_parameter_velocity_structure");
        ghost_prm_p          = prm.get_double("ghost_parameter_pressure");
        ghost_prm_u_v        = prm.get_double("ghost_parameter_displacement_velocity");
        max_ghost_weight     = prm.get_double("max_ghost_weight");
        nitsche_parameter    = prm.get_double("nitsche_parameter");
      }
      prm.leave_subsection();
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      GlobalParameters::declare_parameters(prm);
      PhysicalConstants::declare_parameters(prm);
      Time::declare_parameters(prm);
      GhostParameters::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      GlobalParameters::parse_parameters(prm);
      PhysicalConstants::parse_parameters(prm);
      Time::parse_parameters(prm);
      GhostParameters::parse_parameters(prm);
    }
}
