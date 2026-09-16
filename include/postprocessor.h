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
#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "cute-fsi.h"

namespace Stokes
{
  using namespace dealii;

  /** This class handles all solution values and derived quantities for graphical output.
     * In addition to the solution values, the gradients of the fluid velocity, the deformation
     * and the pressure will be handled as they appear in the numerical analysis of the error.*/
  template <int dim>
  class StokesFSI<dim>::Postprocessor : public DataPostprocessor<dim>
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
}

#endif // POSTPROCESSOR_H
