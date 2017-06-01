#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl_extension.hpp>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"


/** \file POCont.cpp 
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

  char fileName[256], srcPostfix[256], dstPostfix[256], contPostfix[256];
  FILE *dstStream, *dstStreamExp, *dstStreamVecLeft, *dstStreamVecRight;
  gsl_vector_complex *FloquetExp = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *FloquetVecLeft = gsl_matrix_complex_alloc(dim, dim);
  gsl_matrix_complex *FloquetVecRight = gsl_matrix_complex_alloc(dim, dim);

  // Define names and open destination file
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(contPostfix, "_cont%04d_contStep%de%d", 0,
	  (int) (mantis*1.01), (int) (exp*1.01));
  sprintf(dstPostfix, "%s_eta2%04d_r%04d_gamma%04d%s_dt%d", srcPostfix,
	  (int) (p["eta2"] * 1000 + 0.1), (int) (p["r"] * 1000 + 0.1),
	  (int) (p["gamma"] * 1000 + 0.1), contPostfix,
	  (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1));
  
  sprintf(fileName, "%s/continuation/poState/poState%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(fileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing solution: ", fileName);
      perror("");
      return EXIT_FAILURE;
    }

  sprintf(fileName, "%s/continuation/poVecLeft/poVecLeft%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamVecLeft = fopen(fileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing left Floquet vectors: ",
	      fileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(fileName, "%s/continuation/poVecRight/poVecRight%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamVecRight = fopen(fileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing right Floquet vectors: ",
	      fileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(fileName, "%s/continuation/poExp/poExp%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamExp = fopen(fileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing Floquet exponents: ",
	      fileName);
      perror("");
      return EXIT_FAILURE;
    }

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new coupledROCont(&p);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianCoupledROCont(&p, initCont);

  // Define numerical scheme
  //std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim + 1);
  //numericalScheme *scheme = new Euler(dim + 1);

  // Define model (the initial state will be assigned later)
  //std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Define linearized model 
  //std::cout << "Defining linearized model..." << std::endl;
  fundamentalMatrixModel *linMod = new fundamentalMatrixModel(mod, Jacobian);

  // Define periodic orbit problem
  periodicOrbitCont *track = new periodicOrbitCont(linMod, epsDist,
						   epsStepCorrSize, maxIter,
						   dt, numShoot, verbose);

  try {
    // First correct
    if (verbose)
      std::cout << "\nApplying initial correction:" << std::endl;
    track->correct(initCont);

    if (!track->hasConverged()) {
      std::cerr << "\nFirst correction could not converge." << std::endl;
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
  }
  catch(const std::exception &ex) {
    std::cout << "Exception during initial correction.\n"
	      << std::endl;
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
    throw std::exception();
  }
  
  int predCount = 0;
  double contStepAdapt = contStep;
  while (predCount < maxPred)
    {
      // Find periodic orbit
      std::cout << "Applying continuation step "
		<< predCount << " for mu = "
		<< gsl_vector_get(initCont, dim)
		<< " with step " << contStepAdapt << std::endl;
      try {
	track->continueStep(contStepAdapt);

	// Useful?
	if (!track->hasConverged()) {
	  std::cerr << "\nContinuation could not converge."
		    << std::endl;
	  throw std::exception();
	}
	else // Update initial state to current state
	  track->setInitialStateToCurrent();
		
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
      catch(const std::exception &ex) {
	std::cout << "Exception: continuation." << std::endl;
	if (contStepAdapt < 1.e-6) {
	  std::cout << "Time-step already too small.\n"
		    << std::endl;
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
	  throw std::exception();
	}
	else {
	  std::cout << "Dividing time-step.\n" << std::endl;
	  // Divide time step and reinitialize
	  contStepAdapt /= 2;
	  track->initialize();
	}
      }
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
  gsl_vector_complex_free(FloquetExp);
  gsl_matrix_complex_free(FloquetVecLeft);
  gsl_matrix_complex_free(FloquetVecRight);
  freeConfig();

  return 0;
}


