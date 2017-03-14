#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"

using namespace libconfig;


/** \file FPContLoop.cpp 
 *  \brief Fixed point continuation in the coupled recharge oscillator.
 *
 *  Fixed point continuation in the coupled recharge oscillator.
 */


/** \brief Fixed point continuation in the coupled recharge oscillator.
 *
 *  Fixed point continuation in the coupled recharge oscillator.
 */
int main(int argc, char * argv[])
{
  // Read configuration file
  if (argc < 2)
    {
      std::cout << "Enter path to configuration file:" << std::endl;
      std::cin >> configFileName;
    }
  else
    {
      strcpy(configFileName, argv[1]);
    }
  try
   {
     Config cfg;
     std::cout << "Sparsing config file " << configFileName << std::endl;
     cfg.readFile(configFileName);
     readGeneral(&cfg);
     readModel(&cfg);
     readContinuation(&cfg);
     std::cout << "Sparsing success.\n" << std::endl;
    }
  catch(const SettingTypeException &ex) {
    std::cerr << "Setting " << ex.getPath() << " type exception."
	      << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "Setting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "Setting " << ex.getPath() << " name exception."
	      << std::endl;
    throw ex;
  }
  catch(const ParseException &ex) {
    std::cerr << "Parse error at " << ex.getFile() << ":" << ex.getLine()
              << " - " << ex.getError() << std::endl;
    throw ex;
  }
  catch(const FileIOException &ex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw ex;
  }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  gsl_matrix *solJac = gsl_matrix_alloc(dim + 1, dim + 1);
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *eigVec = gsl_matrix_complex_alloc(dim, dim);
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(dim);
  gsl_matrix_view jac;
  char dstFileName[256], srcPostfix[256], dstPostfix[256], contPostfix[256];
  FILE *dstStream, *dstStreamEigVec, *dstStreamEigVal;


  // Define names and open destination file
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(contPostfix, "_cont%04d_contStep%de%d",
	  (int) (gsl_vector_get(initCont, dim) * 1000 + 0.1),
	  (int) (mantis*1.01), (int) (exp*1.01));
  

  // Loop over various parameters
  double eta2Step, rStep, gammaStep;
  double eta2Min = 0.5; double eta2Max = 0.7; size_t neta2 = 21;
  double rMin = 0.1; double rMax = 0.3; size_t nr = 21;
  double gammaMin = 0.3; double gammaMax = 0.5; size_t ngamma = 21;
  if (neta2 == 1)
    eta2Step = 0.;
  else
    eta2Step = (eta2Max - eta2Min) / (neta2 - 1);
  if (nr == 1)
    rStep = 0.;
  else
    rStep = (rMax - rMin) / (nr - 1);
  if (ngamma == 1)
    gammaStep = 0.;
  else
    gammaStep = (gammaMax - gammaMin) / (ngamma - 1);
  for (size_t ieta2 = 0; ieta2 < neta2; ieta2++) {
    p["eta2"] = eta2Min + eta2Step * ieta2;
    for (size_t ir = 0; ir < nr; ir++) {
      p["r"] = rMin + rStep * ir;
      for (size_t igamma = 0; igamma < ngamma; igamma++) {
        p["gamma"] = gammaMin + gammaStep * igamma;
	std::cout << "Continuing for eta2 = " << p["eta2"] << ", r = "
		  << p["r"] << ", gamma = " << p["gamma"] << std::endl;

   	sprintf(dstPostfix, "%s_eta2%04d_r%04d_gamma%04d%s", srcPostfix,
		(int) (p["eta2"] * 1000 + 0.1), (int) (p["r"] * 1000 + 0.1),
		(int) (p["gamma"] * 1000 + 0.1), contPostfix);
	sprintf(dstFileName, "%s/continuation/fpCont/fpCont%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(dstStream = fopen(dstFileName, "w"))) {
	  fprintf(stderr, "Can't open %s for writing states: ",
		  dstFileName);
	  perror("");
	  return EXIT_FAILURE;
	}
	sprintf(dstFileName, "%s/continuation/fpEigVec/fpEigVecCont%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(dstStreamEigVec = fopen(dstFileName, "w"))) {
	  fprintf(stderr, "Can't open %s for writing eigenvectors: ",
		  dstFileName);
	  perror("");
	  return EXIT_FAILURE;
	}
  
	sprintf(dstFileName, "%s/continuation/fpEigVal/fpEigValCont%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(dstStreamEigVal = fopen(dstFileName, "w"))) {
	  fprintf(stderr, "Can't open %s for writing eigenvalues: ",
		  dstFileName);
	  perror("");
	  return EXIT_FAILURE;
	}

	// Start from fixed point for mu = 0 (x = 0)
	gsl_vector_set(initCont, 0, 0.);
	gsl_vector_set(initCont, 1, -p["gamma"] * p["tauExt"] / 2);
	gsl_vector_set(initCont, 2, p["gamma"] * p["tauExt"] / 2);
	gsl_vector_set(initCont, 3, 0.);
  
	// Define field
	vectorField *field = new coupledROCont(&p);
  
	// Define linearized field
	linearField *Jacobian = new JacobianCoupledROCont(&p, initCont);

	// Define fixed point problem
	fixedPointCont *track = new fixedPointCont(field, Jacobian, epsDist,
						   epsStepCorrSize, maxIter,
						   verbose);

	// First correct
	track->correct(initCont);

	if (!track->hasConverged()) {
	  std::cerr << "First correction could not converge." << std::endl;
	  return -1;
	}


	while ((gsl_vector_get(initCont, dim) >= contMin)
	       && (gsl_vector_get(initCont, dim) <= contMax))
	  {
	    // Find fixed point
	    track->continueStep(contStep);

	    if (!track->hasConverged()) {
	      std::cerr << "Continuation could not converge." << std::endl;
	      break;
	    }

	    // Get solution and the Jacobian
	    track->getCurrentState(initCont);
	    track->getStabilityMatrix(solJac);
	    jac = gsl_matrix_submatrix(solJac, 0, 0, dim, dim);

	    // Find eigenvalues
	    gsl_eigen_nonsymmv(&jac.matrix, eigVal, eigVec, w);

	    // Write results
	    if (strcmp(fileFormat, "bin") == 0)
	      {
		gsl_vector_fwrite(dstStream, initCont);
		gsl_vector_complex_fwrite(dstStreamEigVal, eigVal);
		gsl_matrix_complex_fwrite(dstStreamEigVec, eigVec);
	      }
	    else
	      {
		gsl_vector_fprintf(dstStream, initCont, "%lf");
		gsl_vector_complex_fprintf(dstStreamEigVal, eigVal, "%lf");
		gsl_matrix_complex_fprintf(dstStreamEigVec, eigVec, "%lf");
	      }
      
	    // Flush in case premature exit
	    fflush(dstStream);
	    fflush(dstStreamEigVal);
	    fflush(dstStreamEigVec);
	  }

	// Free
	delete track;
	delete Jacobian;
	delete field;
	fclose(dstStreamEigVal);
	fclose(dstStreamEigVec);
	fclose(dstStream);
      }
    }
  }
  // Free
  gsl_matrix_free(solJac);
  gsl_eigen_nonsymmv_free(w);
  gsl_vector_complex_free(eigVal);
  gsl_matrix_complex_free(eigVec);
  freeConfig();

  return 0;
}
