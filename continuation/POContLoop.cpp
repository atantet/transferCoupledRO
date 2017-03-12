#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <cstring>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl_extension.hpp>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"

using namespace libconfig;


void handler(const char *reason, const char *file, int line, int gsl_errno);
  
/** \file POContLoop.cpp 
 *  \brief Periodic orbit continuation in the coupled recharge oscillator.
 *
 * Periodic orbit continuation in the coupled recharge oscillator.
 */


/** \brief Periodic orbit continuation in the coupled recharge oscillator.
 *
 * Periodic orbit continuation in the coupled recharge oscillator.
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
     readSimulation(&cfg);
     readContinuation(&cfg);
     std::cout << "Sparsing success.\n" << std::endl;
    }
  catch(const SettingTypeException &ex) {
    std::cerr << "\nSetting " << ex.getPath() << " type exception."
	      << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "\nSetting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "\nSetting " << ex.getPath() << " name exception."
	      << std::endl;
    throw ex;
  }
  catch(const ParseException &ex) {
    std::cerr << "\nParse error at " << ex.getFile() << ":" << ex.getLine()
              << " - " << ex.getError() << std::endl;
    throw ex;
  }
  catch(const FileIOException &ex) {
    std::cerr << "\nI/O error while reading configuration file." << std::endl;
    throw ex;
  }
  catch (...)
    {
      std::cerr << "\nError reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  gsl_vector *initContFP = gsl_vector_alloc(dim + 1);
  bool foundUnstable;
  gsl_vector_complex *eigValFP = gsl_vector_complex_alloc(dim);
  gsl_complex ev;
  char fileName[256], srcPostfix[256], dstPostfix[256], contPostfix[256];
  FILE *dstStream, *dstStreamExp, *dstStreamVecLeft, *dstStreamVecRight,
    *srcStreamFP, *srcStreamEigValFP;
  gsl_vector_complex *FloquetExp = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *FloquetVecLeft = gsl_matrix_complex_alloc(dim, dim);
  gsl_matrix_complex *FloquetVecRight = gsl_matrix_complex_alloc(dim, dim);


  // Define postfix for fixed point files
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(contPostfix, "_cont%04d_contStep%de%d", 0,
	  (int) (mantis*1.01), (int) (exp*1.01));

  // Desactivate GSL's default error handler
  gsl_error_handler_t *old_handler = gsl_set_error_handler(&handler);

  // Loop over various parameters
  // double eta2Min = 0.5; double eta2Max = 0.7; size_t neta2 = 11;
  double eta2Min = 0.62; double eta2Max = 0.7; size_t neta2 = 5;
  double rMin = 0.1; double rMax = 0.3; size_t nr = 11;
  double gammaMin = 0.3; double gammaMax = 0.5; size_t ngamma = 11;
  double eta2Step = (eta2Max - eta2Min) / (neta2 - 1);
  double rStep = (rMax - rMin) / (nr - 1);
  double gammaStep = (gammaMax - gammaMin) / (ngamma - 1);
  for (size_t ieta2 = 0; ieta2 < neta2; ieta2++) {
    p["eta2"] = eta2Min + eta2Step * ieta2;
    for (size_t ir = 0; ir < nr; ir++) {
      p["r"] = rMin + rStep * ir;
      for (size_t igamma = 0; igamma < ngamma; igamma++) {
        p["gamma"] = gammaMin + gammaStep * igamma;
	std::cout << "Continuing for eta2 = " << p["eta2"] << ", r = "
		  << p["r"] << ", gamma = " << p["gamma"] << std::endl;

	// Define initial state from fixed point continuation
   	sprintf(dstPostfix, "%s_eta2%04d_r%04d_gamma%04d%s", srcPostfix,
		(int) (p["eta2"] * 1000 + 0.1), (int) (p["r"] * 1000 + 0.1),
		(int) (p["gamma"] * 1000 + 0.1), contPostfix);
	sprintf(fileName, "%s/continuation/fpCont/fpCont%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStreamFP = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading eigenvectors: ",
		  fileName);
	  perror("");
	  return EXIT_FAILURE;
	}
	sprintf(fileName, "%s/continuation/fpEigVal/fpEigValCont%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStreamEigValFP = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading eigenvalues: ",
		  fileName);
	  perror("");
	  return EXIT_FAILURE;
	}
	
	// Scan for the first unstable complex eigenvalue
	foundUnstable = false;
	if (strcmp(fileFormat, "bin") == 0) {
	  while (!gsl_vector_fread(srcStreamFP, initContFP)) {
	    // Read eigenvalues
	    if (gsl_vector_complex_fread(srcStreamEigValFP, eigValFP))
	      std::cerr << "Error reading fixed-point eigenvalues."
			<< " Continuing." << std::endl;
	    else {
	      // Test for unstable complex eigenvalues
	      for (size_t iev = 0; iev < (size_t) dim; iev++) {
		ev = gsl_vector_complex_get(eigValFP, iev);
		if ((GSL_REAL(ev) > 0) && (GSL_IMAG(ev) > 1.e-6)) {
		  foundUnstable = true;
		  break;
		}
	      }
	    }
	    if (foundUnstable)
	      break;
	  }
	}
	else {
	  while (!gsl_vector_fscanf(srcStreamFP, initContFP)) {
	    // Read eigenvalues
	    gsl_vector_complex_fscanf(srcStreamEigValFP, eigValFP);
	    // Test for unstable complex eigenvalues
	    for (size_t iev = 0; iev < (size_t) dim; iev++) {
	      ev = gsl_vector_complex_get(eigValFP, iev);
	      if ((GSL_REAL(ev) > 0) && (GSL_IMAG(ev) > 1.e-6)) {
		foundUnstable = true;
		break;
	      }
	    }
	    if (foundUnstable)
	      break;
	  }
	}

	// If an unstable complex eigenvalue has been found,
	// start the periodic orbit continuation from there.
	if (foundUnstable) {
	  sprintf(fileName, "%s/continuation/poCont/poState%s.%s",
		  resDir, dstPostfix, fileFormat);
	  if (!(dstStream = fopen(fileName, "w")))
	    {
	      fprintf(stderr, "Can't open %s for writing solution: ",
		      fileName);
	      perror("");
	      return EXIT_FAILURE;
	    }

	  sprintf(fileName, "%s/continuation/poVecLeft/poVecLeft%s.%s",
		  resDir, dstPostfix, fileFormat);
	  if (!(dstStreamVecLeft = fopen(fileName, "w")))
	    {
	      fprintf(stderr,
		      "Can't open %s for writing left Floquet vectors: ",
		      fileName);
	      perror("");
	      return EXIT_FAILURE;
	    }
  
	  sprintf(fileName, "%s/continuation/poVecRight/poVecRight%s.%s",
		  resDir, dstPostfix, fileFormat);
	  if (!(dstStreamVecRight = fopen(fileName, "w")))
	    {
	      fprintf(stderr,
		      "Can't open %s for writing right Floquet vectors: ",
		      fileName);
	      perror("");
	      return EXIT_FAILURE;
	    }
  
	  sprintf(fileName, "%s/continuation/poExp/poExp%s.%s",
		  resDir, dstPostfix, fileFormat);
	  if (!(dstStreamExp = fopen(fileName, "w")))
	    {
	      fprintf(stderr,
		      "Can't open %s for writing Floquet exponents: ",
		      fileName);
	      perror("");
	      return EXIT_FAILURE;
	    }

	  // Set initial state and control parameter
	  for (size_t d = 0; d < (size_t) (dim + 1); d++)
	    gsl_vector_set(initCont, d, gsl_vector_get(initContFP, d));
	  // Perturb parameter
	  gsl_vector_set(initCont, 0, gsl_vector_get(initCont, 0)
			 * (1 + sqrt(GSL_REAL(ev))));
	  // Set period from imaginary part of eigenvalue
	  gsl_vector_set(initCont, dim + 1, 2 * M_PI / GSL_IMAG(ev));
  
	  // Define field
	  vectorField *field = new coupledROCont(&p);
  
	  // Define linearized field
	  linearField *Jacobian = new JacobianCoupledROCont(&p, initCont);

	  // Define numerical scheme
	  numericalScheme *scheme = new RungeKutta4(dim + 1);

	  // Define model (the initial state will be assigned later)
	  model *mod = new model(field, scheme);

	  // Define linearized model 
	  fundamentalMatrixModel *linMod
	    = new fundamentalMatrixModel(mod, Jacobian);

	  // Define periodic orbit problem
	  periodicOrbitCont *track
	    = new periodicOrbitCont(linMod, epsDist, epsStepCorrSize,
				    maxIter, dt, numShoot, verbose);

	  try {
	    // First correct
	    track->correct(initCont);

	    if (!track->hasConverged()) {
	      std::cerr << "\nFirst correction could not converge."
			<< std::endl;
	      throw std::exception();
	    }

	    // Get Floquet elements
	    getFloquet(track, initCont, FloquetExp,
		       FloquetVecLeft, FloquetVecRight, true);
  
	    // Find and write Floquet elements
	    writeFloquet(track, initCont, FloquetExp, FloquetVecLeft,
			 FloquetVecRight, dstStream, dstStreamExp,
			 dstStreamVecLeft, dstStreamVecRight,
			 fileFormat, verbose);

	    int predCount = 0;
	    while ((gsl_vector_get(initCont, dim) >= contMin)
		   && (gsl_vector_get(initCont, dim) <= contMax))
	      {
		if (predCount >= maxPred) {
		  std::cerr << "\nPrediction count exceeding maxPred = "
			    << maxPred << std::endl;
		  throw std::exception();
		}

		// Find periodic orbit
		std::cout << "Applying continuation step "
			  << predCount << " for mu = "
			  << gsl_vector_get(initCont, dim) << std::endl;
		track->continueStep(contStep);

		if (!track->hasConverged()) {
		  std::cerr << "\nContinuation could not converge."
			    << std::endl;
		  throw std::exception();
		}
		
		// Get Floquet elements
		getFloquet(track, initCont, FloquetExp,
			   FloquetVecLeft, FloquetVecRight, true);
      
		// Find and write Floquet elements
		writeFloquet(track, initCont, FloquetExp,
			     FloquetVecLeft, FloquetVecRight,
			     dstStream, dstStreamExp, dstStreamVecLeft,
			     dstStreamVecRight, fileFormat, verbose);
      
		predCount++;

		// Flush in case premature exit
		fflush(dstStream);
		fflush(dstStreamExp);
		fflush(dstStreamVecLeft);
		fflush(dstStreamVecRight);
	      }
	  }
	  catch(const std::exception &ex) {
	    std::cout << "Exception.\n" << std::endl;
	  }

	  delete track;
	  delete linMod;
	  delete mod;
	  delete scheme;
	  delete Jacobian;
	  delete field;
	  fclose(dstStreamExp);
	  fclose(dstStreamVecLeft);
	  fclose(dstStreamVecRight);
	  fclose(dstStream);
	}
	else 
	  std::cout << "No unstable complex eigenvalue found." << std::endl;
	fclose(srcStreamFP);
	fclose(srcStreamEigValFP);
      }
    }
  }
  gsl_vector_complex_free(eigValFP);
  gsl_vector_free(initContFP);
  gsl_vector_complex_free(FloquetExp);
  gsl_matrix_complex_free(FloquetVecLeft);
  gsl_matrix_complex_free(FloquetVecRight);
  freeConfig();

  return 0;
}


void
handler(const char *reason, const char *file, int line, int gsl_errno)
{
  std::cerr << "\nGSL Error " << gsl_errno << ": " << reason << " in file "
	    << file << " at line " << line << ". Continuing." << std::endl;
  
  return;
}
