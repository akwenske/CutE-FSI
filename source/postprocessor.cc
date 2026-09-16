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

#include "../include/postprocessor.h"

template <int dim>
std::vector<std::string>
Stokes::StokesFSI<dim>::Postprocessor::get_names() const
{
  std::vector<std::string> solution_names(dim, "velocity_fluid");
  solution_names.push_back("pressure");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("velocity_solid");
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

/** Computes the fluid velocity, solid velocity, displacement and pressure of the
 * solution as well as the gradients of the fluid velocity, displacement and pressure
 * for graphical output.
*/
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

      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int e = 0; e < dim; ++e)
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

template class Stokes::StokesFSI<2>::Postprocessor;
