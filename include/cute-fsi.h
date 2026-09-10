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
#ifndef CUTE_FSI_H
#define CUTE_FSI_H

#include "parameters.h"
#include "functions.h"

#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_full_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <vector>
#include <math.h>
#include <tuple>

namespace Stokes
{
    using namespace dealii;

    template <int dim>
    class StokesFSI
    {
    public:
      StokesFSI(const unsigned int m_f,
                const unsigned int m_s,
                const Parameters::AllParameters & parameters);

      void run();

      /** The ActiveFEIndex enum.
       *
       * Classifies cells into purely fluid cells,
       * purely solid cells and cells at the interface.
       */
      enum ActiveFEIndex
      {
        fluid = 0,
        solid  = 1,
        interface = 2,
      };

      /** The SolutionType enum.
       *
       * Helper enum for the graphical output
       * and computation of L2-norms.
       */
      enum SolutionType
      {
        current_solution = 0,
        error  = 1
      };

    private:
      void set_runtime_parameters();

      void make_grid();

      void setup_discrete_level_sets(
          DoFHandler<dim> & level_set_dof_handler,
          PETScWrappers::MPI::Vector  & level_set_fluid);

      void distribute_dofs(
          DoFHandler<dim> & dof_handler,
          NonMatching::MeshClassifier<dim> & mesh_classifier_fluid);

      void initialize_matrices();

      void assemble_system();

      void set_bc();

      void solve();

      void output_results(
          const unsigned int cycle,
          const unsigned int timestep_no,
          SolutionType solution_type);

      void L2_space_norm (
          double & v_f,
          double & v_s,
          double & symgrad_u,
          SolutionType solution_type);

      void L2_space_time_norm(
          double & sum_grad_v_f,
          double & sum_p,
          SolutionType solution_type);

      void read_in_solution(
          std::string filename,
          PETScWrappers::MPI::Vector &solution_vector);

      void compute_reference_solution();

      bool face_has_ghost_penalty(
          const typename DoFHandler<dim>::active_cell_iterator &cell,
          const unsigned int face_index,
          const ActiveFEIndex subdomain) const;

      MPI_Comm                                  mpi_communicator;

      const unsigned int                        n_mpi_processes;
      const unsigned int                        this_mpi_process;

      ConditionalOStream                        pcout;

      const unsigned int                        fe_degree_level_set;
      const unsigned int                        fe_degree_fluid;
      const unsigned int                        fe_degree_solid;
      const unsigned int                        quadrature_degree;

      // We need two separte triangulations. The first is for the current solution and the
      // second for the reference solution at the finest refinement level.
      parallel::distributed::Triangulation<dim> triangulation;
      parallel::distributed::Triangulation<dim> ref_triangulation;

      // We need two pairs of separate DofHandlers: One for the reference solution and one
      // for the current solution. For each solution type there is one DofHandler that manages
      // the solution and one for the level set function.
      // The geometry of our problem will always be described from the fluid point of view.
      const FE_Q<dim>                           fe_level_set;
      DoFHandler<dim>                           level_set_dof_handler;
      DoFHandler<dim>                           ref_level_set_dof_handler;
      PETScWrappers::MPI::Vector                level_set_fluid;
      PETScWrappers::MPI::Vector                ref_level_set_fluid;

      FESystem<dim>                             fe_fluid;
      FESystem<dim>                             fe_solid;
      FESystem<dim>                             fe_interface;

      hp::FECollection<dim>                     fe_collection;
      DoFHandler<dim>                           dof_handler;
      DoFHandler<dim>                           ref_dof_handler;

      NonMatching::MeshClassifier<dim>          mesh_classifier_fluid; /** Mesh classifier from the fluid pov */
      NonMatching::MeshClassifier<dim>          ref_mesh_classifier_fluid;

      // Indices for the extraction of the components, since the FEValuesExtractors
      // doesn't work for the used NonMatching::FEValues
      const unsigned int                        velocity_fluid_index;
      const unsigned int                        pressure_index;
      const unsigned int                        velocity_solid_index;
      const unsigned int                        displacement_index;

      // We have five different vectors handling the solutions.
      // The first stores the solution at the current timestep and
      // the second the solution at the previous timestep. Then, we have
      // the reference solution at the current timestep. To compute the error,
      // we need to project the current solution onto the reference mesh
      // and then store the error in the final Vector object.
      PETScWrappers::MPI::Vector                solution;
      PETScWrappers::MPI::Vector                old_timestep_solution;
      PETScWrappers::MPI::Vector                ref_solution;
      PETScWrappers::MPI::Vector                coarse_solution_on_fine_grid;
      PETScWrappers::MPI::Vector                err;

      AffineConstraints<double>                 constraints;

      SparsityPattern                           sparsity_pattern;
      PETScWrappers::MPI::SparseMatrix          system_matrix;
      PETScWrappers::MPI::Vector                rhs;

      Parameters::AllParameters                 parameters;

      // Number of refinemenets for the initial mesh and number of refinement
      // cycles for the convergence analysis. The latter corresponds to the number
      // of refinements for the reference solution
      unsigned int n_refinements;
      unsigned int n_refinement_cycles;

      // Conduct convergence analysis in space or in time
      bool         do_spatial_analysis;
      bool         do_temporal_analysis;

      // Only print every output_skip-th solution to .vtu
      unsigned int output_skip;

      // Defines the interface geometry
      std::string interface_type;

      // Physcial parameters
      double       v_f_in; // inflow velocity
      double       nu_f;   // fluid viscosity
      double       rho_f;  // fluid density
      double       rho_s;  // solid density
      double       mu;     // lamee parameter mu
      double       lambda; // lamee parameter lambda

      // Temporal Parameters
      double       time;
      double       k; // timestep size
      double       end_time;
      unsigned int timestep_no;

      // Ghost Penalty Parameters
      double       ghost_prm_v_f;
      double       ghost_prm_v_s;
      double       ghost_prm_p;
      double       ghost_prm_u_v;
      double       max_ghost_weight;
      double       nitsche_parameter;

      // Norms of the reference solution
      // ||v_f(T)||, ||v_s(T)||, ||grad u(T)|| as space norms and
      // sqrt(sum_k ||grad v_f||^2), sqrt(sum_k h^2||grad p||^2) as space-time norms
      double       v_f_T_ref;
      double       v_s_T_ref;
      double       grad_u_T_ref;
      double       sum_grad_v_f_ref;
      double       sum_grad_p_ref;

      class Postprocessor;
    };
}

#endif // CUTE_FSI_H
