#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/coupledRO.hpp"

/** \file coupledRO.hpp
 *  \brief Definition of the model of the fully-coupled recharge oscillator.
 */

/*
 * Method definitions:
 */

/*
 * Vector fields definitions:
 */

/** 
 * Evaluate the vector field of the coupled recharge oscillator
 * at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
coupledRO::evalField(const gsl_vector *state, gsl_vector *field,
		     const double t)
{
  const double x = gsl_vector_get(state, 0);
  const double y = gsl_vector_get(state, 1);
  const double z = gsl_vector_get(state, 2);
  
  const double tau = p["tauExt"] + p["mu"] * x;
  const double w = -p["deltas"] * tau + p["w0"];
  const double mv = p["deltas"] * tau;
  const double zp = p["eta1"] * z + p["eta2"];
  const double xs = p["xs0"] * (1. - nl(zp));
  
  gsl_vector_set(field, 0, -p["alpha"] * x - H(w) * w * (x - xs)
		 - H(mv) * mv * x);
  gsl_vector_set(field, 1, -p["r"] * (y + p["gamma"] / 2 * tau));
  gsl_vector_set(field, 2, p["epsh"] * (y + p["gamma"] * tau - z));
  
  return;
}


/** 
 * Evaluate the vector field of the coupled recharge oscillator
 * at a given state for continuation.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
coupledROCont::evalField(const gsl_vector *state, gsl_vector *field,
			 const double t)
{
  const double x = gsl_vector_get(state, 0);
  const double y = gsl_vector_get(state, 1);
  const double z = gsl_vector_get(state, 2);
  const double mu = gsl_vector_get(state, 3);
  
  const double tau = p["tauExt"] + mu * x;
  const double w = -p["deltas"] * tau + p["w0"];
  const double mv = p["deltas"] * tau;
  const double zp = p["eta1"] * z + p["eta2"];
  const double xs = p["xs0"] * (1. - nl(zp));
  
  gsl_vector_set(field, 0, -p["alpha"] * x - H(w) * w * (x - xs)
		 - H(mv) * mv * x);
  gsl_vector_set(field, 1, -p["r"] * (y + p["gamma"] / 2 * tau));
  gsl_vector_set(field, 2, p["epsh"] * (y + p["gamma"] * tau - z));
  
  // Last element is 0
  gsl_vector_set(field, 3, 0.);
 
  return;
}


/**
 * Update the matrix of the Jacobian of the quasi-geostrophic 4 modes model
 * conditionned on the state x.
 * \param[in] x State vector.
*/
void
JacobianCoupledRO::setMatrix(const gsl_vector *state)
{
  const double x = gsl_vector_get(state, 0);
  // const double y = gsl_vector_get(state, 1);
  const double z = gsl_vector_get(state, 2);
  
  const double tau = p["tauExt"] + p["mu"] * x;
  const double w = -p["deltas"] * tau + p["w0"];
  const double mv = p["deltas"] * tau;
  const double zp = p["eta1"] * z + p["eta2"];
  const double xs = p["xs0"] * (1. - nl(zp));
  
  // Derivatives of the wind-stress
  const double dtaudx = p["mu"];
  // Derivatives of the upwelling
  const double dwdx = -p["deltas"] * dtaudx;
  // Derivatives of the meridional velocity
  const double dmvdx = p["deltas"] * dtaudx;
  // Derivatives of the argument inside f
  const double dzpdz = p["eta1"];
  const double dxsdz = -p["xs0"] * dzpdz * dnl(zp);
  
  gsl_matrix_set(A, 0, 0, -p["alpha"] - H(w) * (dwdx * (x - xs) + w)  
		 - H(mv) * (dmvdx * x + mv));
  gsl_matrix_set(A, 0, 1, 0.);
  gsl_matrix_set(A, 0, 2, H(w) * w * dxsdz);
  gsl_matrix_set(A, 1, 0, -p["r"] * p["gamma"] / 2 * dtaudx);
  gsl_matrix_set(A, 1, 1, -p["r"]);
  gsl_matrix_set(A, 1, 2, 0.);
  gsl_matrix_set(A, 2, 0, p["epsh"] * p["gamma"] * dtaudx);
  gsl_matrix_set(A, 2, 1, p["epsh"]);
  gsl_matrix_set(A, 2, 2, -p["epsh"]);
  
  return;
}

/**
 * Update the matrix of the Jacobian of the quasi-geostrophic 4 modes model
 * conditionned on the state x for continuation with respect to \f$\sigma\f$.
 * \param[in] x State vector.
*/
void
JacobianCoupledROCont::setMatrix(const gsl_vector *state)
{
  const double x = gsl_vector_get(state, 0);
  // const double y = gsl_vector_get(state, 1);
  const double z = gsl_vector_get(state, 2);
  const double mu = gsl_vector_get(state, 3);
  
  const double tau = p["tauExt"] + mu * x;
  const double w = -p["deltas"] * tau + p["w0"];
  const double mv = p["deltas"] * tau;
  const double zp = p["eta1"] * z + p["eta2"];
  const double xs = p["xs0"] * (1. - nl(zp));
  
  // Derivatives of the wind-stress
  const double dtaudx = mu;
  const double dtaudmu = x;
  // Derivatives of the upwelling
  const double dwdx = -p["deltas"] * dtaudx;
  const double dwdmu = -p["deltas"] * dtaudmu;
  // Derivatives of the meridional velocity
  const double dmvdx = p["deltas"] * dtaudx;
  const double dmvdmu = p["deltas"] * dtaudmu;
  // Derivatives of the argument inside f
  const double dzpdz = p["eta1"];
  const double dxsdz = -p["xs0"] * dzpdz * dnl(zp);
  
  // Set last row to 0
  gsl_vector_view view = gsl_matrix_row(A, DIM);
  gsl_vector_set_zero(&view.vector);

  gsl_matrix_set(A, 0, 0, -p["alpha"] - H(w) * (dwdx * (x - xs) + w)
		 - H(mv) * (dmvdx * x + mv));
  gsl_matrix_set(A, 0, 1, 0.);
  gsl_matrix_set(A, 0, 2, H(w) * w * dxsdz);
  gsl_matrix_set(A, 1, 0, -p["r"] * p["gamma"] / 2 * dtaudx);
  gsl_matrix_set(A, 1, 1, -p["r"]);
  gsl_matrix_set(A, 1, 2, 0.);
  gsl_matrix_set(A, 2, 0, p["epsh"] * p["gamma"] * dtaudx);
  gsl_matrix_set(A, 2, 1, p["epsh"]);
  gsl_matrix_set(A, 2, 2, -p["epsh"]);
  gsl_matrix_set(A, 0, 3, -H(w) * dwdmu * (x - xs)
		 - H(mv) * dmvdmu * x);
  gsl_matrix_set(A, 1, 3, -p["r"] * p["gamma"] / 2 * dtaudmu);
  gsl_matrix_set(A, 2, 3, p["epsh"] * p["gamma"] * dtaudmu);
  
    return;
}


/*
 * Heaviside distribution.
 * \param[in] x Value.
 * \return      Result of application of Heaviside to x.
 */
double
H(const double x)
{
  return (x > 0 ? 1. : 0. );
}

/*
 * Define nonlinear function in subsurface temperature.
 */
double
nl(double zp)
{
  return tanh(zp);
}

/*
 * Define derivative of nonlinear function in subsurface temperature.
 */
double
dnl(double zp)
{
  return (1. - gsl_pow_2(tanh(zp)));
}

