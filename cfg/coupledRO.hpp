#ifndef COUPLEDRO_HPP
#define COUPLEDRO_HPP

#include <map>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ODESolvers.hpp>
#include <ergoParam.hpp>
#define DIM 3

/** \file coupledRO.hpp
 *  \brief Definition of the model of the fully-coupled recharge oscillator.
 */


/*
 * Class declarations:
 */


/** \brief Vector field for the coupled recharge oscillator.
 *
 *  Vector field for the coupled recharge oscillator.
 */
class coupledRO : public vectorField {
  param p; 
  
public:
  /** \brief Constructor defining the model parameters. */
  coupledRO(const param *p_) : vectorField(), p(*p_) { }

  /** \brief Destructor. */
  ~coupledRO() { }

  /** \brief Return the parameters of the model. */
  void getParameters(param *p_) { *p_ = p; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const param *p_) { p = *p_; return; }

  /** \brief Evaluate the vector field of the coupled recharge oscillator
   *  for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Vector field for the coupled recharge oscillator for continuation.
 *
 *  Vector field for the coupled recharge oscillator for continuation
 *  with respect to \f$\sigma\f$.
 */
class coupledROCont : public vectorField {
  param p;
  
public:
  /** \brief Constructor defining the model parameters. */
  coupledROCont(const param *p_)
    : vectorField(), p(*p_) { }

  /** \brief Destructor. */
  ~coupledROCont() { }

  /** \brief Return the parameters of the model. */
  void getParameters(param *p_) { *p_ = p; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const param *p_)
  { p = *p_; return; }

  /** \brief Evaluate the vector field of the coupled recharge oscillator
   *  for a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
};


/** \brief Jacobian of the coupled recharge oscillator.
 *
 *  Jacobian the coupled recharge oscillator.
 */
class JacobianCoupledRO : public linearField {
  
  param p;

public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianCoupledRO(const param *p_) : linearField(DIM), p(*p_) { }

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianCoupledRO(const param *p_, const gsl_vector *state_)
    : linearField(DIM), p(*p_) { setMatrix(state_); }

  /** \brief Destructor. */
  ~JacobianCoupledRO() { }

  /** \brief Return the parameters of the model. */
  void getParameters(param *p_) { *p_ = p; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const param *p_) { p = *p_; return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Jacobian of the coupled recharge oscillator for continuation.
 *
 *  Jacobian of the coupled recharge oscillator for continuation
 *  with respect to \f$\mu\f$.
 */
class JacobianCoupledROCont : public linearField {
  
  param p;

public:
  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianCoupledROCont(const param *p_)
    : linearField(DIM + 1), p(*p_) { }

  /** \brief Construction by allocating matrix of the linear operator. */
  JacobianCoupledROCont(const param *p_, const gsl_vector *state_)
    : linearField(DIM + 1), p(*p_) { setMatrix(state_); }

  /** \brief Destructor. */
  ~JacobianCoupledROCont() { }

  /** \brief Return the parameters of the model. */
  void getParameters(param *p_) { *p_ = p; return; }

  /** \brief Set parameters of the model. */
  void setParameters(const param *p_) { p = *p_; return; }

  /** \brief Update the matrix of the linear operator after the state. */
  void setMatrix(const gsl_vector *x);
};


/** \brief Heaviside distribution. */
double H(const double x);

/** \brief Define nonlinear function in subsurface temperature. */
double nl(double zp);

/** \brief Define derivative nonlinear function in sub-temperature. */
double dnl(double zp);

#endif
