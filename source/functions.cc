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

#include "../include/functions.h"

using namespace dealii;
//---------------------------------
// Interface
//---------------------------------
template <int dim>
double SignedDistanceInterface<dim>::value(const Point<dim> &p,
                                           const unsigned int /*component*/) const
{
  return 0.75 - p.norm();
}

//---------------------------------
// Inflow boundary
//---------------------------------
template <int dim>
double
InflowBoundary<dim>::value (const Point<dim>  &p,
                             const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  const long double pi = 3.141592653589793238462643;
  const double time_scaling = (time < 2.0 ? (1.0 - std::cos(pi/2.0 * time))/2.0 : 1.0);

  if (component == 0)
    {
    if (p(0) <= -0.7)
      return time_scaling * inflow_velocity * std::pow(std::sin(pi * (p(0) + 1.0) / 0.6),2);
    else if (p(0) >= 0.7)
      return time_scaling * inflow_velocity * std::pow(std::sin(pi * (p(0) - 1.0) / 0.6),2);
    else
      return time_scaling * inflow_velocity;
    }

  return 0.0;
}

template <int dim>
void
InflowBoundary<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values (c) = InflowBoundary<dim>::value (p, c);
}

//---------------------------------
// Right hand side
//---------------------------------
template <int dim>
Tensor<1, dim> RightHandSideFluid<dim>::vector_value (const Point<dim> & /*p*/) const
{
  Tensor<1, dim> values;
  for (unsigned int c=0; c<dim; ++c)
    values[c] = 0.0;
  return values;
}

template <int dim>
Tensor<1, dim> RightHandSideStructure<dim>::vector_value (const Point<dim> & /*p*/) const
{
  Tensor<1, dim> values;
  for (unsigned int c=0; c<dim; ++c)
    values[c] = 0.0;
  return values;
}

template class SignedDistanceInterface<2>;
template class InflowBoundary<2>;
template class RightHandSideFluid<2>;
template class RightHandSideStructure<2>;
