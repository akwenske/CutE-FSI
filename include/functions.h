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
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/lac/vector.h>

using namespace dealii;

template <int dim>
class SignedDistanceInterface : public Function<dim>
{
public:
  SignedDistanceInterface (const std::string interface_type)
    : Function<dim>()
  {
    this->interface_type = interface_type;
  }
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
private:
  std::string interface_type;
};

template <int dim>
class InflowBoundary : public Function<dim>
{
public:
  InflowBoundary (const double time, const double inflow_velocity)
    : Function<dim>(3*dim+1)
  {
    this->time = time;
    this->inflow_velocity= inflow_velocity;
  }

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const override;

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const override;

private:
  double time;
  double inflow_velocity;
};

template <int dim>
class RightHandSideFluid : public Function<dim>
{
public:
  RightHandSideFluid () : Function<dim>(dim) {}
  virtual Tensor<1, dim> vector_value (const Point<dim> &p = Point(0,0)) const;
};

template <int dim>
class RightHandSideStructure : public Function<dim>
{
public:
  RightHandSideStructure () : Function<dim>(dim) {}
  virtual Tensor<1, dim> vector_value (const Point<dim> &p = Point(0,0)) const;
};

#endif // FUNCTIONS_H
