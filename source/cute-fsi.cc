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

#include "../include/cute-fsi.h"

using namespace dealii;

namespace GhostPenalty
{
  /** The Component enum
     *
     * Identifies the physical component for which
     * the ghost penalty will be computed.
     */
  enum Component {
    v_f,
    p,
    v_s,
    u
  };

  /** Computes the ghost penalty.
     *
     * @param component Component for which the ghost penalty will be computed
     * @param gp_prm Ghost penalty paramater gamma
     * @param weight Weight \f$\omega(\kappa)\f$ of the current cell
     * @param h cell size
     * @param jumps_i Vector of jumps of normal derivatives of
     *        test functions in ascending order
     * @param jumps_j Vector of jumps of normal derivates of
     *        ansatz functions in ascending order
     */
  template <typename T>
  double get_gp(GhostPenalty::Component component,
                double gp_prm,
                double weight,
                double h,
                std::vector<T> jumps_i,
                std::vector<T> jumps_j)
  {
    AssertDimension(jumps_i.size(), jumps_j.size());

    // Precompute the set of coefficients of the jump terms
    // depending on the solution variable
    std::vector<double> coefficients;
    double ghost_penalty = 0.0;

    switch(component) {
      case v_f:
        coefficients = {h,
                        Utilities::fixed_power<3>(h)};
        break;
      case p:
        coefficients = {Utilities::fixed_power<3>(h),
                        Utilities::fixed_power<5>(h)/4.};
        break;
      case v_s:
        coefficients = {Utilities::fixed_power<3>(h),
                        Utilities::fixed_power<5>(h)/4.};
        break;
      case u:
        coefficients = {h,
                        Utilities::fixed_power<3>(h)};
        break;
      }

    for (unsigned int l = 0; l < jumps_i.size(); ++l)
      {
        ghost_penalty += coefficients[l] * (jumps_i[l] * jumps_j[l]);
      }

    return 0.5 * gp_prm * weight * ghost_penalty;
  }
} // end namespace GhostPenalty

/**
  * Standard constructor that initializes all the DofHandlers, triangulations,
  * FECollections etc. By default, the degree for the level set function is set to 2.
  * For the fluid, only quadratic elements are implemented, while for the solid
  * linear and quadratic elements are possible.
  * @param input_file the parameter file cute-fsi.prm.
  */
template <int dim>
Stokes::StokesFSI<dim>::StokesFSI(const unsigned int m_f,
                                  const unsigned int m_s,
                                  const Parameters::AllParameters & prm)
  : mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , pcout(std::cout, (this_mpi_process == 0))
  , fe_degree_level_set(2)
  , fe_degree_fluid(m_f) // m_f = 2
  , fe_degree_solid(m_s) // 1 <= m_s <= 2
  , quadrature_degree(std::max(fe_degree_fluid, fe_degree_solid) + 1)
  , triangulation(mpi_communicator,
                  dealii::Triangulation<dim, dim>::none,
                  parallel::distributed::Triangulation<dim>::no_automatic_repartitioning)
  , ref_triangulation(mpi_communicator)
  , fe_level_set(fe_degree_level_set)
  , level_set_dof_handler(triangulation)
  , ref_level_set_dof_handler(ref_triangulation)
  , fe_fluid(FE_Q<dim>(fe_degree_fluid), dim, // velocity fluid
             FE_Q<dim>(fe_degree_fluid-1), 1, // pressure
             FE_Nothing<dim>(), dim,
             FE_Nothing<dim>(), dim)
  , fe_structure(FE_Nothing<dim>(), dim,
                 FE_Nothing<dim>(), 1,
                 FE_Q<dim>(fe_degree_solid), dim, // velocity structure
                 FE_Q<dim>(fe_degree_solid), dim) // displacement
  , fe_interface(FE_Q<dim>(fe_degree_fluid), dim, // velocity fluid
                 FE_Q<dim>(fe_degree_fluid-1), 1, // pressure
                 FE_Q<dim>(fe_degree_solid), dim, // velocity structure
                 FE_Q<dim>(fe_degree_solid), dim) // displacement
  , dof_handler(triangulation)
  , ref_dof_handler(ref_triangulation)
  , mesh_classifier_fluid(level_set_dof_handler,
                          level_set_fluid)
  , ref_mesh_classifier_fluid(ref_level_set_dof_handler,
                              ref_level_set_fluid)
  , velocity_fluid_index(0)
  , pressure_index(dim)
  , velocity_structure_index(dim+1)
  , displacement_index(2*dim+1)
  , parameters(prm)
  , time(0.0)
  , timestep_no(0)
{
  fe_collection.push_back(fe_fluid);
  fe_collection.push_back(fe_structure);
  fe_collection.push_back(fe_interface);
}

/** Set physical and computational parameters.
   *
   * The parameters are given in the file cute-fsi.prm.
   */
template <int dim>
void Stokes::StokesFSI<dim>::set_runtime_parameters()
{
  //Global parameters
  n_refinements                = parameters.n_refinements;
  n_refinement_cycles          = parameters.n_refinement_cycles;
  do_spatial_analysis          = parameters.do_spatial_analysis;
  do_temporal_analysis         = parameters.do_temporal_analysis;
  output_skip                  = parameters.output_skip;
  interface_type               = parameters.interface_type;

  //Fluid parameters
  v_f_in                       = parameters.v_f_in;
  nu_f                         = parameters.nu_f;
  rho_f                        = parameters.rho_f;

  //Structure parameters
  rho_s                        = parameters.rho_s;
  mu                           = parameters.mu;
  lambda                       = parameters.lambda;

  //Timestepping parameters
  end_time                     = parameters.end_time;
  k                            = parameters.timestep_size;

  //Ghost penalty and Nitsche parameters
  ghost_prm_v_f                = parameters.ghost_prm_v_f;
  ghost_prm_v_s                = parameters.ghost_prm_v_s;
  ghost_prm_p                  = parameters.ghost_prm_p;
  ghost_prm_u_v                = parameters.ghost_prm_u_v;
  max_ghost_weight             = parameters.max_ghost_weight;
  nitsche_parameter            = parameters.nitsche_parameter;
}

/** Set up the background mesh.
   *  The mesh is a square domain with side length 2.
  */
template <int dim>
void Stokes::StokesFSI<dim>::make_grid()
{
  pcout << "Creating background mesh" << std::endl;

  Triangulation<dim> triangulation_construction;
  GridGenerator::hyper_cube(triangulation_construction,-1.0,1.0);

  for (auto face : triangulation_construction.active_face_iterators())
    {
      if (face->center()[1] == 1.0)
        face->set_boundary_id(0);
      else
        face->set_boundary_id(1);
    }

  triangulation_construction.refine_global(n_refinements);
  GridGenerator::flatten_triangulation(triangulation_construction,
                                       triangulation);
}

/** Setup the discrete level set. The level set function is given from the fluid
   * point of  view. Depending on the given level_set_dof_handler and level_set_fluid,
   * the level set function is set up either for the current or for the reference solution.
   */
template <int dim>
void Stokes::StokesFSI<dim>::setup_discrete_level_sets(
    DoFHandler<dim> & level_set_dof_handler,
    PETScWrappers::MPI::Vector  & level_set_fluid)
{
  pcout << "Setting up discrete level set function" << std::endl;

  level_set_dof_handler.distribute_dofs(fe_level_set);

  const IndexSet locally_owned_dofs =
      level_set_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(level_set_dof_handler);

  level_set_fluid.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);

  PETScWrappers::MPI::Vector
      competely_distributed_level_set_fluid(locally_owned_dofs,
                                            mpi_communicator);

  const SignedDistanceInterface<dim>
      signed_distance_interface(interface_type);

  VectorTools::interpolate(level_set_dof_handler,
                           signed_distance_interface,
                           competely_distributed_level_set_fluid);

  level_set_fluid = competely_distributed_level_set_fluid;
}

/** Distributes the dofs for the current or reference solution.
   * The location of each cell is determined by the given mesh_classifier_fluid object
   * to either fluid, structure or (cut) interface cells.
   */
template <int dim>
void Stokes::StokesFSI<dim>::distribute_dofs(
    DoFHandler<dim> & dof_handler,
    NonMatching::MeshClassifier<dim> & mesh_classifier_fluid)
{
  pcout << "Distributing degrees of freedom" << std::endl;

  for (const auto &cell :
       dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const NonMatching::LocationToLevelSet cell_location =
            mesh_classifier_fluid.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::intersected)
          cell->set_active_fe_index(ActiveFEIndex::interface);
        else if (cell_location == NonMatching::LocationToLevelSet::inside)
          cell->set_active_fe_index(ActiveFEIndex::fluid);
        else
          cell->set_active_fe_index(ActiveFEIndex::structure);
      }

  dof_handler.distribute_dofs(fe_collection);

  pcout << "Number of dofs: " << dof_handler.n_dofs() << std::endl;
}

/** Initialize the system matrix, rhs- and solution vectors as usual. */
template <int dim>
void Stokes::StokesFSI<dim>::initialize_matrices()
{
  pcout << "Initializing matrices" << std::endl;

  const auto face_has_flux_coupling = [&](const auto & cell,
      const unsigned int face_index)
  {
      return (this->face_has_ghost_penalty(cell, face_index,
                                           ActiveFEIndex::fluid) ||
              this->face_has_ghost_penalty(cell, face_index,
                                           ActiveFEIndex::structure));
    };

  const unsigned int n_components = fe_collection.n_components();
  Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
  Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);

  // determine the face and cell coupling of the discrete form
  for (unsigned int i = 0; i < n_components; i++)
    for (unsigned int j = 0; j < n_components; j++)
      {
        if (i < n_components - dim && j < n_components - dim) // v_f, p, v_s
          {
            face_coupling[i][j] = DoFTools::always;
            cell_coupling[i][j] = DoFTools::always;
          }
        else if (i >= dim + 1 && j >= dim + 1) // v_s, u
          {
            face_coupling[i][j] = DoFTools::always;
            cell_coupling[i][j] = DoFTools::always;
          }
        else
          {
            face_coupling[i][j] = DoFTools::none;
            cell_coupling[i][j] = DoFTools::none;
          }
      }

  const IndexSet locally_owned_dofs =
      dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  set_bc();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  const bool keep_constrained_dofs = true;

  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       keep_constrained_dofs,
                                       cell_coupling,
                                       face_coupling,
                                       numbers::invalid_subdomain_id,
                                       face_has_flux_coupling);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       sparsity_pattern,
                       mpi_communicator);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                  mpi_communicator);

  old_timestep_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                               mpi_communicator);

  rhs.reinit(locally_owned_dofs, mpi_communicator);
}

/** Decides whether a given face lies in the set \f$\mathcal{F}_G^i\f$ for \f$ i\in\{f,s\}\f$.
   * @param cell The current cell on which the face lies.
   * @param face_index The index of the face on the current cell.
   * @param subdomain The subdomain (either fluid or structure) for which the necessity
   *  of penalization should be checked.
   */
template <int dim>
bool Stokes::StokesFSI<dim>::face_has_ghost_penalty(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int face_index,
    const ActiveFEIndex subdomain) const
{
  if (cell->at_boundary(face_index))
    return false;
  const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier_fluid.location_to_level_set(cell);
  const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier_fluid.location_to_level_set(cell->neighbor(face_index));
  switch (subdomain)
    {
    case fluid:
      if ((cell_location == NonMatching::LocationToLevelSet::intersected &&
           neighbor_location != NonMatching::LocationToLevelSet::outside) ||
          (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
           cell_location != NonMatching::LocationToLevelSet::outside))
        return true;
      break;
    case structure:
      if ((cell_location == NonMatching::LocationToLevelSet::intersected &&
           neighbor_location != NonMatching::LocationToLevelSet::inside) ||
          (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
           cell_location != NonMatching::LocationToLevelSet::inside))
        return true;
      break;
    case interface:
      break;
    }
  return false;
}

/** Assemble the system matrix and right hand side vector. */
template <int dim>
void Stokes::StokesFSI<dim>::assemble_system()
{
  pcout << "Assembling" << std::endl;

  FullMatrix<double> local_matrix;
  Vector<double>     local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  const RightHandSideFluid<dim>     rhs_function_fluid;
  const RightHandSideStructure<dim> rhs_function_structure;

  // Assemble the ghost penalty terms via an FEInterfaceValues object
  const QGauss<dim - 1> face_quadrature(quadrature_degree);
  const hp::QCollection<dim - 1> quadrature_collection(face_quadrature);

  FEInterfaceValues<dim> fe_interface_values(fe_collection,
                                             quadrature_collection,
                                             update_gradients |
                                             update_hessians |
                                             update_JxW_values |
                                             update_normal_vectors);

  // Assemble the bulk and interface terms via a NonMatching object
  // that is given from the fluid point of view. Hence, "inside" refers
  // to fluid terms, "outside" to structure terms and "surface" to
  // interface terms.
  const QGauss<1> quadrature_1D(quadrature_degree);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
      update_JxW_values | update_quadrature_points;
  region_update_flags.outside = update_values | update_gradients |
      update_JxW_values | update_quadrature_points;
  region_update_flags.surface = update_values | update_gradients |
      update_JxW_values | update_quadrature_points |
      update_normal_vectors;

  NonMatching::FEValues<dim> non_matching_fe_values_fluid(fe_collection,
                                                          quadrature_1D,
                                                          region_update_flags,
                                                          mesh_classifier_fluid,
                                                          level_set_dof_handler,
                                                          level_set_fluid);

  const FEValuesExtractors::Vector velocity_fluid (velocity_fluid_index);
  const FEValuesExtractors::Scalar pressure (pressure_index);
  const FEValuesExtractors::Vector velocity_structure (velocity_structure_index);
  const FEValuesExtractors::Vector displacement (displacement_index);

  system_matrix = 0;
  rhs = 0;

  const auto symgrad = [&] (Tensor<2,dim> grad_w)
  {
    return 0.5*(grad_w + transpose(grad_w));
  };
  const auto stress_fluid = [&] (Tensor<2,dim> grad_v_f, double p)
  {
    return 2.0 * rho_f * nu_f * symgrad(grad_v_f) -
        p * unit_symmetric_tensor<dim>();
  };
  const auto stress_structure = [&] (Tensor<2,dim> grad_u)
  {
    return 2.0 * mu * symgrad(grad_u) +
        lambda * trace(symgrad(grad_u)) * unit_symmetric_tensor<dim>();
  };


  solution.update_ghost_values();
  old_timestep_solution.update_ghost_values();

  for (const auto &cell :
       dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const double h = cell->diameter();

        local_matrix.reinit (cell->get_fe().dofs_per_cell,
                             cell->get_fe().dofs_per_cell);

        local_rhs.reinit (cell->get_fe().dofs_per_cell);

        non_matching_fe_values_fluid.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values_fluid =
            non_matching_fe_values_fluid.get_inside_fe_values();

        const std::optional<FEValues<dim>> &inside_fe_values_structure =
            non_matching_fe_values_fluid.get_outside_fe_values();

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values_fluid =
            non_matching_fe_values_fluid.get_surface_fe_values();

        // fluid bulk terms
        if (inside_fe_values_fluid)
          {
            std::vector<Vector<double>>
                old_timestep_solution_values (inside_fe_values_fluid->n_quadrature_points,
                                              Vector<double> (3*dim+1));

            inside_fe_values_fluid->get_function_values(old_timestep_solution,
                                                        old_timestep_solution_values);

            for (const unsigned int q :
                 inside_fe_values_fluid->quadrature_point_indices())
              {
                const Point<dim> &point =
                    inside_fe_values_fluid->quadrature_point(q);

                Tensor<1,dim> v_f_old_timestep_solution;

                for (unsigned int k = 0; k < dim; k++)
                  {
                    v_f_old_timestep_solution[k] =
                        old_timestep_solution_values[q](k+velocity_fluid_index);
                  }

                for (const unsigned int i : inside_fe_values_fluid->dof_indices())
                  {
                    // fluid test functions
                    Tensor<1,dim> v_f_i;
                    Tensor<2,dim> grad_v_f_i;


                    for (unsigned int k = 0; k < dim; k++)
                      {
                        v_f_i[k] =
                            inside_fe_values_fluid->shape_value_component(i, q, k+velocity_fluid_index);
                        grad_v_f_i[k] =
                            inside_fe_values_fluid->shape_grad_component(i, q, k+velocity_fluid_index);
                      }

                    const double p_i =
                        inside_fe_values_fluid->shape_value_component(i, q, pressure_index);

                    for (const unsigned int j : inside_fe_values_fluid->dof_indices())
                      {
                        // fluid ansatz functions
                        Tensor<1,dim> v_f_j;
                        Tensor<2,dim> grad_v_f_j;

                        for (unsigned int k = 0; k < dim; k++)
                          {
                            v_f_j[k] =
                                inside_fe_values_fluid->shape_value_component(j, q, k+velocity_fluid_index);
                            grad_v_f_j[k] =
                                inside_fe_values_fluid->shape_grad_component(j, q, k+velocity_fluid_index);
                          }

                        const double p_j =
                            inside_fe_values_fluid->shape_value_component(j, q, pressure_index);

                        Tensor<2,dim> stress = stress_fluid(grad_v_f_j, p_j);


                        local_matrix(i, j) += (rho_f * v_f_j * v_f_i +
                                               k * scalar_product(stress, grad_v_f_i) +
                                               k * trace(grad_v_f_j) * p_i) *
                            inside_fe_values_fluid->JxW(q);
                      }

                    local_rhs(i) += (rho_f * (k * rhs_function_fluid.vector_value(point) +
                                              v_f_old_timestep_solution) * v_f_i) *
                        inside_fe_values_fluid->JxW(q);
                  }
              }
          }
        // In deal.ii v9.7.0, cells that intersect the interface in only one vertex count as
        // cut cells. As the integral over a single point is zero, on these cells only the
        // ghost penalties contribute to the matrix and right hand side. Because the solid
        // deformation appears only in terms of the ansatz function in the penalization, the resulting
        // system matrix will be singular. As a workaround for that, one can add an additional penalty
        // term g_u_u that serves to extend the test function of the deformation onto these faces.
        // To this end, we check whether the cell with faces for which we add ghost penalties in the
        // structure intersects the solid domain in more than one point. It is recommended to use a
        // very small ghost penalty parameter for g_u_u, e.g. 1.0e-15.
        // Alternatively, one could manually reclassify interface cells, for which either
        // !inside_fe_values_fluid or !inside_fe_values_structure holds and use a
        // cell's active fe index to determine whether or not ghost penalty terms should be applied.
        // In deal.ii v.9.8.0, this classification is amended and no workaround need to be applied.
        // Hence, we will only add g_u_u depending on the package version.
#if defined(DEAL_II_LESS_9_8_0)
        bool cell_contains_structure = false;
#endif

        // structure bulk terms
        if (inside_fe_values_structure)
          {
#if defined(DEAL_II_LESS_9_8_0)
            cell_contains_structure = true;
#endif

            std::vector<Vector<double>>
                old_timestep_solution_values (inside_fe_values_structure->n_quadrature_points,
                                              Vector<double> (3*dim+1));

            inside_fe_values_structure->get_function_values(old_timestep_solution,
                                                            old_timestep_solution_values);

            for (const unsigned int q :
                 inside_fe_values_structure->quadrature_point_indices())
              {
                const Point<dim> &point =
                    inside_fe_values_structure->quadrature_point(q);

                Tensor<1,dim> v_s_old_timestep_solution;
                Tensor<1,dim> u_old_timestep_solution;

                for (unsigned int k = 0; k < dim; k++)
                  {
                    v_s_old_timestep_solution[k] =
                        old_timestep_solution_values[q](k+velocity_structure_index);
                    u_old_timestep_solution[k] =
                        old_timestep_solution_values[q](k+displacement_index);
                  }

                for (const unsigned int i : inside_fe_values_structure->dof_indices())
                  {
                    // solid test functions
                    Tensor<1,dim> v_s_i;
                    Tensor<2,dim> grad_v_s_i;
                    Tensor<1,dim> u_i;

                    for (unsigned int k = 0; k < dim; k++)
                      {
                        v_s_i[k] =
                            inside_fe_values_structure->shape_value_component(i, q, k+velocity_structure_index);
                        grad_v_s_i[k] =
                            inside_fe_values_structure->shape_grad_component(i, q, k+velocity_structure_index);
                        u_i[k] =
                            inside_fe_values_structure->shape_value_component(i, q, k+displacement_index);
                      }


                    for (const unsigned int j : inside_fe_values_structure->dof_indices())
                      {
                        // solid ansatz functions
                        Tensor<1,dim> v_s_j;
                        Tensor<1,dim> u_j;
                        Tensor<2,dim> grad_u_j;

                        for (unsigned int k = 0; k < dim; k++)
                          {
                            v_s_j[k] =
                                inside_fe_values_structure->shape_value_component(j, q, k+velocity_structure_index);
                            u_j[k] =
                                inside_fe_values_structure->shape_value_component(j, q, k+displacement_index);
                            grad_u_j[k] =
                                inside_fe_values_structure->shape_grad_component(j, q, k+displacement_index);
                          }

                        Tensor<2,dim> stress = stress_structure(grad_u_j);

                        local_matrix(i, j) += (rho_s * v_s_j * v_s_i +
                                               k * scalar_product(stress, grad_v_s_i) +
                                               (u_j - k * v_s_j) * u_i) *
                            inside_fe_values_structure->JxW(q);
                      }

                    local_rhs(i) += (rho_s *
                                     (k * rhs_function_structure.vector_value(point) +
                                      v_s_old_timestep_solution) * v_s_i +
                                     u_old_timestep_solution * u_i) *
                        inside_fe_values_structure->JxW(q);
                  }
              }
          }

        // interface terms
        if (surface_fe_values_fluid)
          {
            for (unsigned int q = 0;
                 q < surface_fe_values_fluid->n_quadrature_points; q++)
              {
                const Tensor<1,dim> &normal_fluid =
                    surface_fe_values_fluid->normal_vector(q);

                for (const unsigned int i : surface_fe_values_fluid->dof_indices())
                  {
                    // test functions
                    Tensor<1,dim> v_f_i;
                    Tensor<2,dim> grad_v_f_i;
                    Tensor<1,dim> v_s_i;

                    for (unsigned int k = 0; k < dim; k++)
                      {
                        v_f_i[k] =
                            surface_fe_values_fluid->shape_value_component(i, q, k+velocity_fluid_index);
                        grad_v_f_i[k] =
                            surface_fe_values_fluid->shape_grad_component(i, q, k+velocity_fluid_index);
                        v_s_i[k] =
                            surface_fe_values_fluid->shape_value_component(i, q, k+velocity_structure_index);
                      }

                    double p_i =
                        surface_fe_values_fluid->shape_value_component(i, q, pressure_index);

                    Tensor<2,dim> stress_i = stress_fluid(grad_v_f_i, -1.0 * p_i);

                    for (const unsigned int j :
                         surface_fe_values_fluid->dof_indices())
                      {
                        // ansatz functions
                        Tensor<1,dim> v_f_j;
                        Tensor<2,dim> grad_v_f_j;
                        Tensor<1,dim> v_s_j;

                        for (unsigned int k = 0; k < dim; k++)
                          {
                            v_f_j[k] =
                                surface_fe_values_fluid->shape_value_component(j, q, k+velocity_fluid_index);
                            grad_v_f_j[k] =
                                surface_fe_values_fluid->shape_grad_component(j, q, k+velocity_fluid_index);
                            v_s_j[k] =
                                surface_fe_values_fluid->shape_value_component(j, q, k+velocity_structure_index);
                          }

                        double p_j =
                            surface_fe_values_fluid->shape_value_component(j, q, pressure_index);

                        Tensor<2,dim> stress_j = stress_fluid(grad_v_f_j, p_j);

                        const double nitsche_term = rho_f * nu_f * nitsche_parameter / h *
                            (v_f_j - v_s_j) * (v_f_i - v_s_i);

                        local_matrix(i, j) += k *
                            (nitsche_term - (stress_j * normal_fluid) * (v_f_i - v_s_i)
                             - (v_f_j - v_s_j) * (stress_i * normal_fluid)) *
                            surface_fe_values_fluid->JxW(q);
                      }
                  }
              }
          }

        local_dof_indices.resize (cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(local_matrix, local_rhs,
                                               local_dof_indices,
                                               system_matrix, rhs);

        // Assembly of the ghost penalty terms:
        // First, compute the relatve weight function for the fluid and solid parts.
        double cut_cell_measure_fluid = 0.0;

        if (inside_fe_values_fluid)
          for (const unsigned int q : inside_fe_values_fluid->quadrature_point_indices())
            cut_cell_measure_fluid += inside_fe_values_fluid->JxW(q);

        const double relative_cell_measure_fluid =
            cut_cell_measure_fluid / cell->measure();
        const double relative_cell_measure_structure =
            1.0 - relative_cell_measure_fluid;

        const double weight_fluid =
            max_ghost_weight * std::exp(-2.0 * std::log(max_ghost_weight) * relative_cell_measure_fluid);
        const double weight_structure =
            max_ghost_weight * std::exp(-2.0 * std::log(max_ghost_weight) * relative_cell_measure_structure);

        for (unsigned int f : cell->face_indices())
          {
            const bool face_has_fluid_gp =
                face_has_ghost_penalty(cell, f, ActiveFEIndex::fluid);
            const bool face_has_solid_gp =
                face_has_ghost_penalty(cell, f, ActiveFEIndex::structure);

            const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;

            if (face_has_fluid_gp || face_has_solid_gp)
              {
                fe_interface_values.reinit(cell,
                                           f,
                                           invalid_subface,
                                           cell->neighbor(f),
                                           cell->neighbor_of_neighbor(f),
                                           invalid_subface);

                const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                const unsigned int n_interface_q_points =
                    fe_interface_values.n_quadrature_points;
                FullMatrix<double> local_stabilization(n_interface_dofs,
                                                       n_interface_dofs);
                Vector<double> local_rhs_stabilization(n_interface_dofs);

                std::vector<Tensor<2,dim>> jumps_v_s_old_timestep_solution_grads(n_interface_q_points);
                std::vector<Tensor<3,dim>> jumps_v_s_old_timestep_solution_hess(n_interface_q_points);

                fe_interface_values[velocity_structure].get_jump_in_function_gradients(
                      old_timestep_solution,
                      jumps_v_s_old_timestep_solution_grads);
                if (fe_degree_solid > 1)
                  {
                    fe_interface_values[velocity_structure].get_jump_in_function_hessians(
                          old_timestep_solution,
                          jumps_v_s_old_timestep_solution_hess);
                  }

                for (unsigned int q = 0;
                     q < n_interface_q_points;
                     ++q)
                  {
                    const Tensor<1,dim> normal = fe_interface_values.normal(q);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      {
                        Tensor<1,dim> jumps_v_f_grad_i;
                        Tensor<1,dim> jumps_v_f_hess_i;

                        double jumps_p_grad_i;

                        Tensor<1,dim> jumps_v_s_grad_i;
                        Tensor<1,dim> jumps_v_s_hess_i;

                        jumps_v_f_grad_i =
                            fe_interface_values[velocity_fluid].jump_in_gradients(i, q) * normal;
                        jumps_v_f_hess_i =
                            (fe_interface_values[velocity_fluid].jump_in_hessians(i, q) * normal) * normal;

                        jumps_p_grad_i =
                            fe_interface_values[pressure].jump_in_gradients(i, q) * normal;

                        jumps_v_s_grad_i =
                            fe_interface_values[velocity_structure].jump_in_gradients(i, q) * normal;

                        // jumps in test functions to be used in get_gp
                        std::vector<Tensor<1,dim>> jumps_v_f_i {jumps_v_f_grad_i, jumps_v_f_hess_i};
                        std::vector<double>        jumps_p_i   {jumps_p_grad_i};
                        std::vector<Tensor<1,dim>> jumps_v_s_i {jumps_v_s_grad_i};

                        if (fe_degree_solid > 1)
                          {
                            jumps_v_s_hess_i =
                                (fe_interface_values[velocity_structure].jump_in_hessians(i, q) * normal) * normal;

                            jumps_v_s_i.push_back(jumps_v_s_hess_i);
                          }

                        for (unsigned int j = 0; j < n_interface_dofs; ++j)
                          {
                            Tensor<1,dim> jumps_v_f_grad_j;
                            Tensor<1,dim> jumps_v_f_hess_j;

                            double jumps_p_grad_j;

                            Tensor<1,dim> jumps_u_grad_j;
                            Tensor<1,dim> jumps_u_hess_j;

                            Tensor<1,dim> jumps_v_s_grad_j;
                            Tensor<1,dim> jumps_v_s_hess_j;

                            jumps_v_f_grad_j =
                                fe_interface_values[velocity_fluid].jump_in_gradients(j, q) * normal;
                            jumps_v_f_hess_j =
                                (fe_interface_values[velocity_fluid].jump_in_hessians(j, q) * normal) * normal;

                            jumps_p_grad_j =
                                fe_interface_values[pressure].jump_in_gradients(j, q) * normal;

                            jumps_v_s_grad_j =
                                fe_interface_values[velocity_structure].jump_in_gradients(j, q) * normal;

                            jumps_u_grad_j =
                                fe_interface_values[displacement].jump_in_gradients(j, q) * normal;

                            // jumps in ansatz functions to be used in get_gp
                            std::vector<Tensor<1,dim>> jumps_v_f_j {jumps_v_f_grad_j, jumps_v_f_hess_j};
                            std::vector<double>        jumps_p_j   {jumps_p_grad_j};
                            std::vector<Tensor<1,dim>> jumps_v_s_j {jumps_v_s_grad_j};
                            std::vector<Tensor<1,dim>> jumps_u_j   {jumps_u_grad_j};

                            if (fe_degree_solid > 1)
                              {
                                jumps_v_s_hess_j =
                                    (fe_interface_values[velocity_structure].jump_in_hessians(j, q) * normal) * normal;
                                jumps_u_hess_j   =
                                    (fe_interface_values[displacement].jump_in_hessians(j, q) * normal) * normal;

                                jumps_v_s_j.push_back(jumps_v_s_hess_j);
                                jumps_u_j.push_back(jumps_u_hess_j);
                              }

                            double g_v_f = GhostPenalty::get_gp (GhostPenalty::v_f, ghost_prm_v_f, weight_fluid, h,
                                                                 jumps_v_f_i,
                                                                 jumps_v_f_j);

                            double g_p   = GhostPenalty::get_gp (GhostPenalty::p, ghost_prm_p, weight_fluid, h,
                                                                 jumps_p_i,
                                                                 jumps_p_j);

                            double g_v_s = GhostPenalty::get_gp (GhostPenalty::v_s, ghost_prm_v_s, weight_structure, h,
                                                                 jumps_v_s_i,
                                                                 jumps_v_s_j);

                            double g_u_v = GhostPenalty::get_gp (GhostPenalty::u, ghost_prm_u_v, weight_structure, h,
                                                                 jumps_v_s_i,
                                                                 jumps_u_j);

                            if (face_has_fluid_gp)
                              {
                                local_stabilization(i, j) += k *
                                    (2. * rho_f * nu_f * g_v_f + g_p) *
                                    fe_interface_values.JxW(q);
                              }
                            if (face_has_solid_gp)
                              {
                                local_stabilization(i, j) +=
                                    (rho_s * g_v_s + k * 2. * mu * g_u_v) *
                                    fe_interface_values.JxW(q);

#if defined(DEAL_II_LESS_9_8_0)
                                if (!cell_contains_structure)
                                  {
                                    Tensor<1, dim> jumps_u_grad_i =
                                        fe_interface_values[displacement].jump_in_gradients(i, q) * normal;
                                    std::vector<Tensor<1,dim>> jumps_u_i {jumps_u_grad_i};
                                    if (fe_degree_solid > 1)
                                      {
                                        Tensor<1, dim> jumps_u_hess_i   =
                                            (fe_interface_values[displacement].jump_in_hessians(i, q) * normal) * normal;

                                        jumps_u_i.push_back(jumps_u_hess_i);
                                      }

                                    const double g_u_u = GhostPenalty::get_gp (GhostPenalty::u, 1.0e-15, weight_structure, h,
                                                                               jumps_u_i,
                                                                               jumps_u_j);
                                    local_stabilization(i, j) +=  k * 2. * mu * g_u_u *
                                        fe_interface_values.JxW(q);
                                  }
#endif

                              }
                          }

                        if (face_has_solid_gp)
                          {
                            std::vector<Tensor<1,dim>>
                                jumps_v_s_solution_old {jumps_v_s_old_timestep_solution_grads[q] * normal};

                            if (fe_degree_solid > 1)
                              {
                                jumps_v_s_solution_old.push_back((jumps_v_s_old_timestep_solution_hess[q] * normal) * normal);
                              }

                            double old_g_v_s = GhostPenalty::get_gp (GhostPenalty::v_s, ghost_prm_v_s, weight_structure, h,
                                                                     jumps_v_s_i,
                                                                     jumps_v_s_solution_old);

                            local_rhs_stabilization(i) += rho_s * old_g_v_s * fe_interface_values.JxW(q);
                          }
                      }
                  }

                const std::vector<types::global_dof_index>
                    local_interface_dof_indices =
                    fe_interface_values.get_interface_dof_indices();

                constraints.distribute_local_to_global(local_stabilization,
                                                       local_rhs_stabilization,
                                                       local_interface_dof_indices,
                                                       system_matrix, rhs);
              }
          }
      }
  system_matrix.compress(VectorOperation::add);
  rhs.compress(VectorOperation::add);
}

/** Setting the boundary conditions as usual. Depending on the refinement level,
      the computational solid domain may also intersect the outer domain boundary
      in case of the spherical interface. Hence, we also apply zero boundary conditions
      to the solid components.*/
template <int dim>
void Stokes::StokesFSI<dim>::set_bc()
{
  const IndexSet locally_owned_dofs =
      dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  ComponentMask component_mask(3*dim+1, true);

  component_mask.set(pressure_index, false);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(3*dim+1),
                                           constraints,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           InflowBoundary<dim>(time, v_f_in),
                                           constraints,
                                           component_mask);

  constraints.close();
}

/** Solve the linear system of equations via the sparse direct solver MUMPS. */
template <int dim>
void Stokes::StokesFSI<dim>::solve()
{
  pcout << "Solving system" << std::endl;

  SolverControl cn;
  PETScWrappers::SparseDirectMUMPS solver(cn);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  PETScWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                             mpi_communicator);

  solver.solve(system_matrix,
               completely_distributed_solution,
               rhs);

  constraints.distribute(completely_distributed_solution);
  solution = completely_distributed_solution;
}

/** This class handles all solution values and derived quantities for graphical output.
  * In addition to the solution values, the gradients of the fluid velocity, the deformation
  * and the pressure will be handled as they appear in the numerical analysis of the error.*/
template <int dim>
class Stokes::StokesFSI<dim>::Postprocessor : public DataPostprocessor<dim>
{
public:
  Postprocessor () {}

  virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<std::string> get_names() const override;

  virtual std::vector<
  DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual UpdateFlags get_needed_update_flags() const override;
};

template <int dim>
std::vector<std::string>
Stokes::StokesFSI<dim>::Postprocessor::get_names() const
{
  std::vector<std::string> solution_names(dim, "velocity_fluid");
  solution_names.push_back("pressure");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("velocity_structure");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("displacement");
  for (unsigned int d = 0; d < dim * dim; ++d)
    solution_names.push_back("grad_v_f");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("grad_p");
  for (unsigned int d = 0; d < dim*dim; ++d)
    solution_names.push_back("grad_u");

  return solution_names;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
Stokes::StokesFSI<dim>::Postprocessor::get_data_component_interpretation()
const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector); // v_f
  interpretation.push_back(DataComponentInterpretation::component_is_scalar); // p
  for (unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector); // v_s
  for (unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector); // u
  for (unsigned int d = 0; d < dim * dim; ++d)
    interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor); // grad_v_f
  for (unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector); // grad_p
  for (unsigned int d = 0; d < dim * dim; ++d)
    interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor); // grad_u

  return interpretation;
}

template <int dim>
UpdateFlags
Stokes::StokesFSI<dim>::Postprocessor::get_needed_update_flags() const
{
  return update_values | update_gradients;
}

template <int dim>
void Stokes::StokesFSI<dim>::Postprocessor::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>>                &computed_quantities) const
{
  AssertDimension (input_data.solution_values.size(),
                   computed_quantities.size());
  AssertDimension (input_data.solution_gradients.size(),
                   computed_quantities.size());

  for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          computed_quantities[p][d] =
              input_data.solution_values[p][d]; // v_f

          computed_quantities[p][dim+1+d] =
              input_data.solution_values[p][dim+1+d]; // v_s

          computed_quantities[p][2*dim+1+d] =
              input_data.solution_values[p][2*dim+1+d]; // u
        }
      computed_quantities[p][dim] =
          input_data.solution_values[p][dim]; // p

      const unsigned int grad_v_f_index = 3*dim+1;
      const unsigned int grad_p_index = grad_v_f_index + dim*dim;
      const unsigned int grad_u_index = grad_p_index + dim;

      for (unsigned int d=0; d<dim; ++d)
        {
          for (unsigned int e=0; e<dim; ++e)
            {
              const unsigned int unrolled_index =
                  Tensor<2,dim>::component_to_unrolled_index(TableIndices<2>(d,e));

              computed_quantities[p][grad_v_f_index + unrolled_index] =
                  input_data.solution_gradients[p][d][e]; // grad_v_f

              computed_quantities[p][grad_u_index + unrolled_index] =
                  input_data.solution_gradients[p][2*dim+1+d][e]; // grad_u
            }
          computed_quantities[p][grad_p_index+d] =
              input_data.solution_gradients[p][dim][d]; // grad_p
        }
    }
}

/** Write solution quantities to vtu. The solution values and the derived gradients given by
   * \ref Postprocessor as well as the level set function and the partition in case of parallel
   * computations will be written to vtu. This method can output these quantities both for
   * the current solution, as well as for the error, as indicated by the solution_type.*/
template <int dim>
void Stokes::StokesFSI<dim>::output_results(const unsigned int cycle,
                                            const unsigned int timestep_no,
                                            SolutionType solution_type)
{
  DoFHandler<dim>*                           dof_handler_ptr;
  DoFHandler<dim>*                           level_set_dof_handler_ptr;
  PETScWrappers::MPI::Vector*                level_set_fluid_ptr;
  parallel::distributed::Triangulation<dim>* triangulation_ptr;
  PETScWrappers::MPI::Vector*                solution_ptr;

  std::string filename;

  if (solution_type == SolutionType::error)
    {
      dof_handler_ptr               = &ref_dof_handler;
      level_set_dof_handler_ptr     = &ref_level_set_dof_handler;
      level_set_fluid_ptr           = &ref_level_set_fluid;
      triangulation_ptr             = &ref_triangulation;
      solution_ptr                  = &err;

      filename = "error" + std::to_string(cycle) +
          "-" + std::to_string(timestep_no) + ".vtu";
    }
  else if (solution_type == SolutionType::current_solution)
    {
      dof_handler_ptr               = &dof_handler;
      level_set_dof_handler_ptr     = &level_set_dof_handler;
      level_set_fluid_ptr           = &level_set_fluid;
      triangulation_ptr             = &triangulation;
      solution_ptr                  = &solution;

      filename = "solution" + std::to_string(cycle) +
          "-" + std::to_string(timestep_no) + ".vtu";
    }
  else
    {
      std::string error_msg = std::string("Given SolutionType is unknown!")
          + std::string("SolutionType error or current_solution is required!");
      throw std::runtime_error(error_msg);
    }

  pcout << "Writing file " << filename << std::endl;

  Postprocessor postprocessor;

  DataOut<dim> data_out;
  data_out.attach_dof_handler (*dof_handler_ptr);

  data_out.add_data_vector(*solution_ptr,
                           postprocessor);

  data_out.add_data_vector(*level_set_dof_handler_ptr,
                           *level_set_fluid_ptr,
                           "level_set_fluid");

  Vector<float> subdomain(triangulation_ptr->n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation_ptr->locally_owned_subdomain();
  data_out.add_data_vector(subdomain,
                           "subdomain");
  Vector<float> fe_indices(triangulation_ptr->n_active_cells());
  for(const auto &cell : dof_handler_ptr->active_cell_iterators())
    if (cell->is_locally_owned())
      fe_indices(cell->active_cell_index()) = cell->active_fe_index();
  data_out.add_data_vector(fe_indices,
                           "fe_indices");

  data_out.build_patches();

  data_out.write_vtu_in_parallel(filename,
                                 mpi_communicator);
}

/** Computes the spatial L2-norms of the error analysis on the physical domains.
   * @param v_f \f$ ||v_f(T)||_{\Omega_f}^2 \f$.
   * @param v_s \f$ ||v_s(T)||_{\Omega_s}^2 \f$.
   * @param grad_u \f$ ||\nabla u(T)||_{\Omega_s}^2 \f$.
   * @param solution_type Compute these normt for the current solution or the error.
  */
template<int dim>
void Stokes::StokesFSI<dim>::L2_space_norm(double &v_f,
                                           double &v_s,
                                           double &grad_u,
                                           SolutionType solution_type)
{
  double local_v_f    = 0.0;
  double local_v_s    = 0.0;
  double local_grad_u = 0.0;

  DoFHandler<dim>*                  dof_handler_ptr;
  NonMatching::MeshClassifier<dim>* mesh_classifier_fluid_ptr;
  DoFHandler<dim>*                  level_set_dof_handler_ptr;
  PETScWrappers::MPI::Vector*       level_set_fluid_ptr;
  PETScWrappers::MPI::Vector*       solution_ptr;

  if (solution_type == SolutionType::error)
    {
      dof_handler_ptr               = &ref_dof_handler;
      mesh_classifier_fluid_ptr     = &ref_mesh_classifier_fluid;
      level_set_dof_handler_ptr     = &ref_level_set_dof_handler;
      level_set_fluid_ptr           = &ref_level_set_fluid;
      solution_ptr                  = &err;
    }
  else if (solution_type == SolutionType::current_solution)
    {
      dof_handler_ptr               = &dof_handler;
      mesh_classifier_fluid_ptr     = &mesh_classifier_fluid;
      level_set_dof_handler_ptr     = &level_set_dof_handler;
      level_set_fluid_ptr           = &level_set_fluid;
      solution_ptr                  = &solution;
    }
  else
    {
      std::string error_msg = std::string("Given SolutionType is unknown!")
          + std::string("SolutionType error or current_solution is required!");
      throw std::runtime_error(error_msg);
    }

  Assert(solution_ptr->has_ghost_elements(), ExcInternalError());

  const QGauss<1> quadrature_1D(quadrature_degree);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
      update_JxW_values | update_quadrature_points;
  region_update_flags.outside = update_values | update_gradients |
      update_JxW_values | update_quadrature_points;

  NonMatching::FEValues<dim> non_matching_fe_values_fluid(fe_collection,
                                                          quadrature_1D,
                                                          region_update_flags,
                                                          *mesh_classifier_fluid_ptr,
                                                          *level_set_dof_handler_ptr,
                                                          *level_set_fluid_ptr);
  solution_ptr->update_ghost_values();

  for (const auto &cell :
       dof_handler_ptr->active_cell_iterators())
    if (cell->is_locally_owned())
      {
        non_matching_fe_values_fluid.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values_fluid =
            non_matching_fe_values_fluid.get_inside_fe_values();
        const std::optional<FEValues<dim>> &inside_fe_values_structure =
            non_matching_fe_values_fluid.get_outside_fe_values();

        if (inside_fe_values_fluid)
          {
            std::vector<Vector<double>>
                local_solution_values (inside_fe_values_fluid->n_quadrature_points,
                                       Vector<double> (3*dim+1));

            inside_fe_values_fluid->get_function_values(*solution_ptr,
                                                        local_solution_values);

            for (const unsigned int q :
                 inside_fe_values_fluid->quadrature_point_indices())
              {
                Tensor<1,dim> v_f_solution;

                for (unsigned int k = 0; k < dim; k++)
                  {
                    v_f_solution[k] =
                        local_solution_values[q](k+velocity_fluid_index);
                  }

                local_v_f += v_f_solution * v_f_solution *
                    inside_fe_values_fluid->JxW(q);
              }
          }
        if (inside_fe_values_structure)
          {
            std::vector<Vector<double>>
                local_solution_values (inside_fe_values_structure->n_quadrature_points,
                                       Vector<double> (3*dim+1));
            std::vector<std::vector<Tensor<1,dim>>>
                local_solution_grads (inside_fe_values_structure->n_quadrature_points,
                                      std::vector<Tensor<1,dim>> (3*dim+1));

            inside_fe_values_structure->get_function_values(*solution_ptr,
                                                            local_solution_values);
            inside_fe_values_structure->get_function_gradients(*solution_ptr,
                                                               local_solution_grads);

            for (const unsigned int q :
                 inside_fe_values_structure->quadrature_point_indices())
              {
                Tensor<1,dim> v_s_solution;
                Tensor<2,dim> grad_v_s_solution;

                for (unsigned int k = 0; k < dim; k++)
                  {
                    v_s_solution[k] =
                        local_solution_values[q](k+velocity_structure_index);
                    grad_v_s_solution[k] =
                        local_solution_grads[q][k+displacement_index];
                  }

                local_v_s += scalar_product(v_s_solution,
                                            v_s_solution) *
                    inside_fe_values_structure->JxW(q);

                local_grad_u += scalar_product(grad_v_s_solution,
                                               grad_v_s_solution) *
                    inside_fe_values_structure->JxW(q);
              }
          }
      }

  v_f    = Utilities::MPI::sum(local_v_f, mpi_communicator);
  v_s    = Utilities::MPI::sum(local_v_s, mpi_communicator);
  grad_u = Utilities::MPI::sum(local_grad_u, mpi_communicator);

  v_f    = std::sqrt(v_f);
  v_s    = std::sqrt(v_s);
  grad_u = std::sqrt(grad_u);
}

/** Computes the square of the temporal L2-norms of the error analysis on the physical domains.
   * @param sum_grad_v_f \f$ \sum_{n=1}^N k   ||\nabla v_f(t_n)||_{\Omega_f}^2 \f$.
   * @param sum_grad_p \f$ \sum_{n=1}^N k h^2 ||\nabla p(t_n)||_{\Omega_f}^2 \f$.
   * @param solution_type Compute these normt for the current solution or the error.
  */
template <int dim>
void Stokes::StokesFSI<dim>::L2_space_time_norm(double & sum_grad_v_f,
                                                double & sum_grad_p,
                                                SolutionType solution_type)
{
  double local_grad_v_f    = 0.0;
  double local_grad_p      = 0.0;

  double h_current = 0.0;

  DoFHandler<dim>*                  dof_handler_ptr;
  NonMatching::MeshClassifier<dim>* mesh_classifier_fluid_ptr;
  DoFHandler<dim>*                  level_set_dof_handler_ptr;
  PETScWrappers::MPI::Vector*       level_set_fluid_ptr;
  PETScWrappers::MPI::Vector*       solution_ptr;

  if (solution_type == SolutionType::error)
    {
      dof_handler_ptr               = &ref_dof_handler;
      mesh_classifier_fluid_ptr     = &ref_mesh_classifier_fluid;
      level_set_dof_handler_ptr     = &ref_level_set_dof_handler;
      level_set_fluid_ptr           = &ref_level_set_fluid;
      solution_ptr                  = &err;
    }
  else if (solution_type == SolutionType::current_solution)
    {
      dof_handler_ptr               = &dof_handler;
      mesh_classifier_fluid_ptr     = &mesh_classifier_fluid;
      level_set_dof_handler_ptr     = &level_set_dof_handler;
      level_set_fluid_ptr           = &level_set_fluid;
      solution_ptr                  = &solution;
    }
  else
    {
      std::string error_msg = std::string("Given SolutionType is unknown!")
          + std::string("SolutionType error or current_solution is required!");
      throw std::runtime_error(error_msg);
    }

  Assert(solution_ptr->has_ghost_elements(), ExcInternalError());

  const QGauss<1> quadrature_1D(quadrature_degree);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
      update_JxW_values | update_quadrature_points;

  NonMatching::FEValues<dim> non_matching_fe_values_fluid(fe_collection,
                                                          quadrature_1D,
                                                          region_update_flags,
                                                          *mesh_classifier_fluid_ptr,
                                                          *level_set_dof_handler_ptr,
                                                          *level_set_fluid_ptr);

  solution_ptr->update_ghost_values();

  // The pressure norm is scaled with respect to the mesh size h_current
  // of the current solution
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          h_current = cell->diameter();
          break;
        }
    }
  h_current = Utilities::MPI::max(h_current, mpi_communicator);

  for (const auto &cell :
       dof_handler_ptr->active_cell_iterators())
    if (cell->is_locally_owned())
      {
        non_matching_fe_values_fluid.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values_fluid =
            non_matching_fe_values_fluid.get_inside_fe_values();

        if (inside_fe_values_fluid)
          {
            std::vector<std::vector<Tensor<1,dim>>>
                local_solution_grads (inside_fe_values_fluid->n_quadrature_points,
                                      std::vector<Tensor<1,dim>> (3*dim+1));

            inside_fe_values_fluid->get_function_gradients(*solution_ptr, local_solution_grads);

            for (const unsigned int q :
                 inside_fe_values_fluid->quadrature_point_indices())
              {
                Tensor<1,dim> grad_p_solution = local_solution_grads[q][pressure_index];
                Tensor<2,dim> grad_v_f_solution;

                for (unsigned int k = 0; k < dim; k++)
                  {
                    grad_v_f_solution[k] =
                        local_solution_grads[q][k+velocity_fluid_index];
                  }

                local_grad_v_f += k *
                    scalar_product(grad_v_f_solution, grad_v_f_solution) *
                    inside_fe_values_fluid->JxW(q);

                local_grad_p += k * Utilities::fixed_power<2>(h_current) *
                    scalar_product(grad_p_solution, grad_p_solution) *
                    inside_fe_values_fluid->JxW(q);
              }
          }
      }


  sum_grad_v_f += Utilities::MPI::sum(local_grad_v_f, mpi_communicator);
  sum_grad_p   += Utilities::MPI::sum(local_grad_p, mpi_communicator);
}

/** Reads in the reference solution in filename.*/
template<int dim>
void Stokes::StokesFSI<dim>::read_in_solution(std::string filename,
                                              PETScWrappers::MPI::Vector & ref_solution)
{
  std::ifstream solution_file(filename);
  std::string solution_line;
  std::getline(solution_file, solution_line);
  int pos1;
  int pos2;
  double entry;
  std::vector<double> values;
  std::vector<unsigned int> indices;
  unsigned int start_index;

  // Each process saves its part of the reference solution in a
  // seperate file, which can be distinguished by filename and also its header.
  // The header contains information about the writing process as well as the range
  // of indices of the reference solution vector that are stored in the given file.
  // Thus, to read in the vector values, one has to also find the index range.

  // Find the end of the header
  pos1 = solution_line.find(" ");
  pos2 = solution_line.find("-");
  // Save the start index of the index range
  start_index = atof((solution_line.substr(pos1, pos2)).c_str());

  pos1 = solution_line.find("]");
  solution_line = solution_line.substr(pos1 + 1);

  // Read in the actual values of the solution
  while ((pos1 = solution_line.find(" ")) >= 0)
    {
      solution_line = solution_line.substr(pos1 + 1);
      pos2 = solution_line.find(" ");
      if (pos2 >= 0)
        {
          entry = atof((solution_line.substr(0, pos2)).c_str());
          values.push_back(entry);
        }
    }
  // Save the index range
  for (unsigned int i = 0; i < values.size(); i++)
    indices.push_back(start_index + i);

  const IndexSet locally_owned_ref_dofs = ref_dof_handler.locally_owned_dofs();
  PETScWrappers::MPI::Vector completely_distributed_ref_solution (locally_owned_ref_dofs,
                                                                  mpi_communicator);

  // Write the read in values into the given index range
  completely_distributed_ref_solution.add(indices, values);
  completely_distributed_ref_solution.compress(VectorOperation::add);

  ref_solution = completely_distributed_ref_solution;
}

/** Compute the reference solution on the finest mesh.
   *
   * The refinement level of the finest mesh corresponds to
   * n_refinements + n_refinement_cycles. Save the reference solution for each
   * timestep, compute the reference norms and initialize the reference DoFHandlers,
   * level set function, triangulation and mesh classifier.
   */
template <int dim>
void Stokes::StokesFSI<dim>::compute_reference_solution()
{
  std::ios_base::fmtflags f(pcout.get_stream().flags());

  pcout << "\n=============================="
        << "====================================="
        << "\nCompute reference solution"
        << "\n=============================="
        << "====================================="
        << std::endl;

  // setup the mesh, mesh size and timestep size for the reference refinement level
  if (do_spatial_analysis)
    triangulation.refine_global(n_refinement_cycles);
  if (do_temporal_analysis)
    k /= std::pow(2, n_refinement_cycles);

  setup_discrete_level_sets(level_set_dof_handler,
                            level_set_fluid);
  pcout << "Classifying cells" << std::endl;
  mesh_classifier_fluid.reclassify();
  distribute_dofs(dof_handler, mesh_classifier_fluid);
  initialize_matrices();

  pcout << "\n=============================="
        << "====================================="
        << "\nTimestep " << timestep_no
        << ": " << time
        << " (" << k << ")"
        << "\n=============================="
        << "====================================="
        << std::endl;

  pcout << std::endl;
  pcout << "Initial value solution" << std::endl;
  output_results(n_refinement_cycles,
                 timestep_no,
                 SolutionType::current_solution);

  sum_grad_v_f_ref = 0.0;
  sum_grad_p_ref = 0.0;

  while (time < end_time)
    {
      time += k;
      timestep_no++;
      old_timestep_solution = solution;

      pcout << "\n=============================="
            << "====================================="
            << "\nTimestep " << timestep_no
            << ": " << time
            << " (" << k << ")"
            << "\n=============================="
            << "====================================="
            << std::endl;

      pcout << std::endl;

      set_bc();
      assemble_system();
      solve();

      L2_space_time_norm(sum_grad_v_f_ref, sum_grad_p_ref,
                         SolutionType::current_solution);

      pcout << "Save reference solution." << std::endl;
      const std::string filename =
          "ref-solution-"
          + std::to_string(this_mpi_process)
          + "-"
          + std::to_string(n_refinement_cycles)
          + "-"
          + std::to_string(timestep_no)
          + ".txt";

      std::ofstream solution_file;
      solution_file.open(filename);
      solution.print(solution_file,12);
      solution_file.close();

      if (timestep_no % output_skip == 0)
        output_results(n_refinement_cycles, timestep_no,
                       SolutionType::current_solution);
    }

  L2_space_norm(v_f_T_ref, v_s_T_ref, grad_u_T_ref,
                SolutionType::current_solution);

  sum_grad_v_f_ref = std::sqrt(sum_grad_v_f_ref);
  sum_grad_p_ref = std::sqrt(sum_grad_p_ref);

  // In case of a convergence analysis, prepare all necessary runtime variables
  if (n_refinement_cycles > 0)
    {
      // To compute the errors later on the refinement level of the reference solution,
      // corresponding DofHandlers need to be initialized.
      pcout << "Setting up reference dof handler." << std::endl;

      ref_triangulation.copy_triangulation(triangulation);

      setup_discrete_level_sets(ref_level_set_dof_handler,
                                ref_level_set_fluid);

      ref_mesh_classifier_fluid.reclassify();

      distribute_dofs(ref_dof_handler,
                      ref_mesh_classifier_fluid);

      const IndexSet locally_owned_ref_dofs =
          ref_dof_handler.locally_owned_dofs();
      const IndexSet locally_relevant_ref_dofs =
          DoFTools::extract_locally_relevant_dofs(ref_dof_handler);
      ref_solution.reinit(locally_owned_ref_dofs,
                          locally_relevant_ref_dofs,
                          mpi_communicator);

      pcout << "Reinit time variables and triangulation." << std::endl;

      time = 0.0;
      timestep_no = 0;

      if (do_spatial_analysis)
        triangulation.coarsen_global(n_refinement_cycles);
      if (do_temporal_analysis)
        k *= std::pow(2,n_refinement_cycles);
    }

  pcout << "\n=============================="
        << "====================================="
        << "\nReference solution done."
        << "\n=============================="
        << "====================================="
        << "\n\n"
        << std::endl;

  pcout.get_stream().flags(f);
}

/** Start the computation of the FSI problem.
   *
   * If do_spatial_analysis or temporal_analysis are set to true, includes a convergence
   * study in space and/or time. In case of a spatial analysis, the mesh refinement level
   * for the reference solution corresponds to n_refinements + n_refinement_cycles.
   * If n_refinement_cycles is set to 0, no convergence analysis will be conducted
   * but the solution for a single run will still be computed.
   * For each cycle of the convergence analysis, both the space time and space norms for
   * the current solution as well as the error will be printed in a table.
   */
template <int dim>
void Stokes::StokesFSI<dim>::run()
{
  std::ios_base::fmtflags f(pcout.get_stream().flags());
  set_runtime_parameters();

  ConvergenceTable   convergence_table; // norms of errors
  TableHandler       table; // norms of current solution

  // Variables to be stored the space-time-l2 norms:
  // error:
  double sum_grad_v_f_err;
  double sum_grad_p_err;
  // current solution:
  double sum_grad_v_f_sol;
  double sum_grad_p_sol;

  // Variables to be stored the space-l2 norms at the end time:
  // error:
  double v_f_T_err;
  double v_s_T_err;
  double grad_u_T_err;
  // current solution:
  double v_f_T_sol;
  double v_s_T_sol;
  double grad_u_T_sol;

  double h; // mesh size

  // We compute the error on the current coarse temporal discretization.
  // Hence, we need to identify the timestep number of the reference solution
  // that corresponds to the same point in time as the current timestep number
  // of the coarse solution. To this end, we define an offset value that is
  // set to zero for a pure spatial analysis and for the temporal analyis
  // initially to 2^n_refinement_cylces and half of that for the next cycle.
  double timestep_no_offset = 1.0;
  if (do_temporal_analysis)
    timestep_no_offset = std::pow(2,n_refinement_cycles);

  make_grid();

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          h = cell->minimum_vertex_distance();
          break;
        }
    }
  h = Utilities::MPI::max(h, mpi_communicator);

  pcout << "\n=============================="
        << "====================================="  << std::endl;
  pcout << " Interface type: " << interface_type
        << "\n=============================="
        << "====================================="  << std::endl;
  pcout << "m_f: " << fe_degree_fluid << "\t m_s: " << fe_degree_solid
        << "\nNonmatching Quadrature degree: "      << quadrature_degree
        << "\n=============================="
        << "====================================="  << std::endl;
  pcout << "Parameters\n"
        << "==========\n"
        << "Density fluid:      "   <<  rho_f << "\n"
        << "Viscosity fluid:    "   <<  nu_f << "\n"
        << "Inflow velocity:    "   <<  v_f_in << "\n"
        << "Density structure:  "   <<  rho_s << "\n"
        << "Lame coeff. mu:     "   <<  mu << "\n"
        << "Lame coeff. lambda: "   <<  lambda << "\n"
        << "gamma v_f:          "   <<  ghost_prm_v_f << "\n"
        << "gamma v_s:          "   <<  ghost_prm_v_s << "\n"
        << "gamma p  :          "   <<  ghost_prm_p << "\n"
        << "gamma u v:          "   <<  ghost_prm_u_v << "\n"
        << "max ghost weight:   "   <<  max_ghost_weight << "\n"
        << "gamma N:            "   <<  nitsche_parameter << "\n"
        << std::endl;

  const unsigned int output_skip = parameters.output_skip;

  compute_reference_solution();

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; cycle++)
    {
      pcout << "Refinement cycle " << cycle << std::endl;
      setup_discrete_level_sets(level_set_dof_handler,
                                level_set_fluid);
      pcout << "Classifying cells" << std::endl;
      mesh_classifier_fluid.reclassify();
      distribute_dofs(dof_handler, mesh_classifier_fluid);
      initialize_matrices();

      sum_grad_v_f_err = 0.0;
      sum_grad_p_err = 0.0;
      sum_grad_v_f_sol = 0.0;
      sum_grad_p_sol = 0.0;

      pcout << "\n=============================="
            << "====================================="
            << "\nTimestep " << timestep_no
            << ": " << time
            << " (" << k << ")"
            << "\n=============================="
            << "====================================="
            << std::endl;

      pcout << std::endl;

      pcout << "Initial value solution" << std::endl;
      output_results(cycle,timestep_no,
                     SolutionType::current_solution);

      // The initial value of each solution is the zero vector, hence the initial
      // error is always zero as well.
      if (do_spatial_analysis || do_temporal_analysis)
        {
          const IndexSet locally_owned_ref_dofs =
              ref_solution.locally_owned_elements();
          const IndexSet locally_relevant_ref_dofs =
              DoFTools::extract_locally_relevant_dofs(ref_dof_handler);

          err.reinit(locally_owned_ref_dofs,
                     locally_relevant_ref_dofs,
                     mpi_communicator);

          output_results(cycle, timestep_no,
                         SolutionType::error);
        }

      while (time < end_time)
        {
          time += k;
          timestep_no++;
          old_timestep_solution = solution;

          pcout << "\n=============================="
                << "====================================="
                << "\nTimestep " << timestep_no
                << ": " << time
                << " (" << k << ")"
                << "\n=============================="
                << "====================================="
                << std::endl;

          pcout << std::endl;

          set_bc();
          assemble_system();
          solve();

          // Compute the error between current and reference solution
          if (do_spatial_analysis || do_temporal_analysis)
            {
              unsigned int reference_timestep_no = timestep_no * timestep_no_offset;
              const std::string filename =
                  "ref-solution-"
                  + std::to_string(this_mpi_process)
                  + "-"
                  + std::to_string(n_refinement_cycles)
                  + "-"
                  + std::to_string(reference_timestep_no)
                  + ".txt";

              pcout << "Reading in reference solution "
                    << reference_timestep_no << std::endl;

              read_in_solution(filename, ref_solution);

              pcout << "Interpolating current solution to reference mesh" << std::endl;

              const IndexSet locally_owned_ref_dofs =
                  ref_solution.locally_owned_elements();
              const IndexSet locally_relevant_ref_dofs =
                  DoFTools::extract_locally_relevant_dofs(ref_dof_handler);

              coarse_solution_on_fine_grid.reinit(locally_owned_ref_dofs,
                                                  mpi_communicator);

              err.reinit(locally_owned_ref_dofs,
                         locally_relevant_ref_dofs,
                         mpi_communicator);

              if (do_spatial_analysis)
                {
                  VectorTools::interpolate_to_different_mesh(dof_handler,
                                                             solution,
                                                             ref_dof_handler,
                                                             coarse_solution_on_fine_grid);
                }
              else
                {
                  coarse_solution_on_fine_grid = solution;
                }

              coarse_solution_on_fine_grid -= ref_solution;
              err = coarse_solution_on_fine_grid;
            }

          L2_space_time_norm(sum_grad_v_f_err, sum_grad_p_err,
                             SolutionType::error);
          L2_space_time_norm(sum_grad_v_f_sol, sum_grad_p_sol,
                             SolutionType::current_solution);

          if (timestep_no % output_skip == 0)
            {
              output_results(cycle, timestep_no,
                             SolutionType::current_solution);
              output_results(cycle, timestep_no,
                             SolutionType::error);
            }
        }

      L2_space_norm(v_f_T_err, v_s_T_err, grad_u_T_err,
                    SolutionType::error);
      L2_space_norm(v_f_T_sol, v_s_T_sol, grad_u_T_sol,
                    SolutionType::current_solution);

      sum_grad_v_f_err = std::sqrt(sum_grad_v_f_err);
      sum_grad_p_err   = std::sqrt(sum_grad_p_err);
      sum_grad_v_f_sol = std::sqrt(sum_grad_v_f_sol);
      sum_grad_p_sol   = std::sqrt(sum_grad_p_sol);

      pcout << std::endl;
      pcout << "--------------------------------------------------------------" << std::endl;
      pcout << "Space norms: " <<std::endl;
      pcout << std::endl;
      pcout << std::setprecision(8) << std::scientific;
      pcout << std::setw(25) << std::left << "||err_v_f(T)||:"
            << std::setw(20) << v_f_T_err    << "\n"
            << std::setw(25) << std::left << "||err_v_s(T)||:"
            << std::setw(20) << v_s_T_err    << "\n"
            << std::setw(25) << std::left << "||grad err_u(T)||:"
            << std::setw(20) << grad_u_T_err << "\n";
      pcout << std::endl;
      pcout << "Space time norms: " << std::endl;
      pcout << std::endl;
      pcout << std::setw(25) << std::left << "||grad err_v_f||_I,O:"
            << std::setw(20) << sum_grad_v_f_err << "\n"
            << std::setw(25) << std::left << "||grad err_p||_I,O:"
            << std::setw(20) << sum_grad_p_err   << "\n";
      pcout << "--------------------------------------------------------------" << std::endl;

      // Add the computed values to the (convergence-) tables
      {
        {
          convergence_table.add_value("Cycle", cycle);
          convergence_table.add_value("h", h);
          convergence_table.add_value("k", k);
          convergence_table.add_value("||err_v_f(T)||", v_f_T_err);
          convergence_table.add_value("||err_v_s(T)||", v_s_T_err);
          convergence_table.add_value("||grad err_u(T)||", grad_u_T_err);
          convergence_table.add_value("||grad err_v_f||_I,O", sum_grad_v_f_err);
          convergence_table.add_value("||grad err_p||_I,O", sum_grad_p_err);

          convergence_table.set_precision("||err_v_f(T)||", 8);
          convergence_table.set_precision("||err_v_s(T)||", 8);
          convergence_table.set_precision("||grad err_u(T)||", 8);
          convergence_table.set_precision("||grad err_v_f||_I,O", 8);
          convergence_table.set_precision("||grad err_p||_I,O", 8);

          convergence_table.set_scientific("||err_v_f(T)||", true);
          convergence_table.set_scientific("||err_v_s(T)||", true);
          convergence_table.set_scientific("||grad err_u(T)||", true);
          convergence_table.set_scientific("||grad err_v_f||_I,O", true);
          convergence_table.set_scientific("||grad err_p||_I,O", true);
        }

        {
          table.add_value("Cycle", cycle);
          table.add_value("h", h);
          table.add_value("k", k);
          table.add_value("||v_f(T)||", v_f_T_sol);
          table.add_value("||v_s(T)||", v_s_T_sol);
          table.add_value("||grad u(T)||", grad_u_T_sol);
          table.add_value("||grad v_f||_I,O", sum_grad_v_f_sol);
          table.add_value("||grad p||_I,O", sum_grad_p_sol);

          table.set_precision("||v_f(T)||", 8);
          table.set_precision("||v_s(T)||", 8);
          table.set_precision("||grad u(T)||", 8);
          table.set_precision("||grad v_f||_I,O", 8);
          table.set_precision("||grad p||_I,O", 8);

          table.set_scientific("||v_f(T)||", true);
          table.set_scientific("||v_s(T)||", true);
          table.set_scientific("||grad u(T)||", true);
          table.set_scientific("||grad v_f||_I,O", true);
          table.set_scientific("||grad p||_I,O", true);
        }

        // Add the norms of the reference solution to the table in the last cycle
        if (cycle == n_refinement_cycles - 1)
          {
            if (do_spatial_analysis)
              h *= 0.5;
            if (do_temporal_analysis)
              k *= 0.5;

            table.add_value("Cycle", n_refinement_cycles);
            table.add_value("h", h);
            table.add_value("k", k);
            table.add_value("||v_f(T)||", v_f_T_ref);
            table.add_value("||v_s(T)||", v_s_T_ref);
            table.add_value("||grad u(T)||", grad_u_T_ref);
            table.add_value("||grad v_f||_I,O", sum_grad_v_f_ref);
            table.add_value("||grad p||_I,O", sum_grad_p_ref);

            table.set_precision("||v_f(T)||", 8);
            table.set_precision("||v_s(T)||", 8);
            table.set_precision("||grad u(T)||", 8);
            table.set_precision("||grad v_f||_I,O", 8);
            table.set_precision("||grad p||_I,O", 8);

            table.set_scientific("||v_f(T)||", true);
            table.set_scientific("||v_s(T)||", true);
            table.set_scientific("||grad u(T)||", true);
            table.set_scientific("||grad v_f||_I,O", true);
            table.set_scientific("||grad p||_I,O", true);
          }

        pcout << std::endl;
        pcout << "Solution norms: " << std::endl;
        pcout << std::endl;
        if (this_mpi_process == 0)
          table.write_text(std::cout);
        pcout << std::endl;

        if (n_refinement_cycles > 0)
          {
            convergence_table.evaluate_convergence_rates("||err_v_f(T)||",
                                                         ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates("||err_v_s(T)||",
                                                         ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates("||grad err_u(T)||",
                                                         ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates("||grad err_v_f||_I,O",
                                                         ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates("||grad err_p||_I,O",
                                                         ConvergenceTable::reduction_rate_log2);
            pcout << std::endl;
            pcout << "Convergence rates using reference solution: " << std::endl;
            pcout << std::endl;
            if (this_mpi_process == 0)
              convergence_table.write_text(std::cout);
            pcout << std::endl;
          }
      }

      // Prepare for the next cycle
      time = 0.0;
      timestep_no = 0;

      if (do_spatial_analysis)
        {
          triangulation.refine_global(1);
          h *= 0.5;
        }
      if (do_temporal_analysis)
        {
          k *= 0.5;
          timestep_no_offset *= 0.5;
        }
      pcout.get_stream().flags(f);
    }

  // Print solution norms if no convergence analysis is done
  if (n_refinement_cycles == 0)
    {
      table.add_value("h", h);
      table.add_value("k", k);
      table.add_value("||v_f(T)||", v_f_T_ref);
      table.add_value("||v_s(T)||", v_s_T_ref);
      table.add_value("||grad u(T)||", grad_u_T_ref);
      table.add_value("||grad v_f||_I,O", sum_grad_v_f_ref);
      table.add_value("||grad p||_I,O", sum_grad_p_ref);

      table.set_precision("||v_f(T)||", 8);
      table.set_precision("||v_s(T)||", 8);
      table.set_precision("||grad u(T)||", 8);
      table.set_precision("||grad v_f||_I,O", 8);
      table.set_precision("||grad p||_I,O", 8);

      table.set_scientific("||v_f(T)||", true);
      table.set_scientific("||v_s(T)||", true);
      table.set_scientific("||grad u(T)||", true);
      table.set_scientific("||grad v_f||_I,O", true);
      table.set_scientific("||grad p||_I,O", true);

      pcout << "Solution norms: " << std::endl;
      pcout << std::endl;
      if (this_mpi_process == 0)
        table.write_text(std::cout);
      pcout << std::endl;
    }
}

template class Stokes::StokesFSI<2>;
