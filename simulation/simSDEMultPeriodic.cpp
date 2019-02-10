#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h++>
#include <SDESolvers.hpp>
#include <ODESolvers.hpp>
#include <gsl_extension.hpp>
#include <omp.h>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"

using namespace libconfig;


/** \file simMult.cpp 
 *  \ingroup examples
 *  \brief Simulate multiple trajectories.
 *
 *  Simulate multiple trajectories.
 */


/** \brief Simulation of multiple trajectories.
 *
 *  Simulation of multiple trajectories.
 */
int main(int argc, char * argv[])
{
  char dstPostfix[256], srcPostfix[256];

  // Read configuration file
  if (argc < 2) {
    std::cout << "Enter path to configuration file:" << std::endl;
    std::cin >> configFileName;
  }
  else
    strcpy(configFileName, argv[1]);
  try {
    Config cfg;
    std::cout << "Sparsing config file " << configFileName << std::endl;
    cfg.readFile(configFileName);
    readGeneral(&cfg);
    readModel(&cfg);
    readSimulation(&cfg);
    readSprinkle(&cfg);
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
  catch (...) {
    std::cerr << "Error reading configuration file" << std::endl;
    return(EXIT_FAILURE);
  }

  sprintf(srcPostfix, "_%s", caseName);
  sprintf(dstPostfix, "%s_mu%03d_eta2%03d_gamma%03d_r%03d\
_sigmahInf2%03d_L%d_spinup%d_dt%d_samp%d", srcPostfix,
	  (int) (p["mu"] * 100 + 0.1), (int) (p["eta2"] * 1000 + 0.1),
	  (int) (p["gamma"] * 1000 + 0.1), (int) (p["r"] * 1000 + 0.1),
	  (int) (p["sigmahInf2"] * 1000 + 0.1), (int) L, (int) spinup,
	  (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  
  // Iterate several trajectories
#pragma omp parallel
  {
    gsl_matrix *X;
    char dstFileName[256];
    FILE *dstStream;
    size_t seed;
    gsl_rng * r;
    gsl_matrix *S;
    
    // Define diffusion matrix
    S = gsl_matrix_calloc(dim, dim);
    gsl_matrix_set(S, dim - 1, dim - 1, p["sigmah"]);

    // Set random number generator
    r = gsl_rng_alloc(gsl_rng_ranlxs1);

    // Get seed and set random number generator
    seed = (size_t) (1 + omp_get_thread_num());
#pragma omp critical
    {
      std::cout << "Setting random number generator with seed: " << seed
		<< std::endl;
      std::cout.flush();
    }
    gsl_rng_set(r, seed);

    // Define field
    vectorField *field = new coupledROPeriodic(&p);
  
    // Define stochastic vector field
    vectorFieldStochastic *stocField = new additiveWiener(S, r);

    // Define numerical scheme
    numericalSchemeStochastic *scheme = new EulerMaruyama(dim);

    // Define model
    modelStochastic *mod
      = new modelStochastic(field, stocField, scheme);
    
#pragma omp for
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      mod->setCurrentState(initState);

      // Numerical integration of spinup
#pragma omp critical
      {
	std::cout << "Integrating spinup " << traj << std::endl;
      }
      X = gsl_matrix_alloc(1, 1); // Fake allocation
      mod->integrate(initState, spinup, dt, 0., printStepNum, &X);

      // Numerical integration
#pragma omp critical
      {
	std::cout << "Integrating trajectory " << traj << std::endl;
      }
      mod->integrate(LCut, dt, 0., printStepNum, &X);

      // Write results
#pragma omp critical
      {
	sprintf(dstFileName, "%s/simulation/sim%s_traj%d.%s",
		resDir, dstPostfix, (int) traj, fileFormat);
	if (!(dstStream = fopen(dstFileName, "w"))) {
	  std::cerr << "Can't open " << dstFileName
		    << " for writing simulation: " << std::endl;;
	  perror("");
	}

	std::cout << "Writing " << traj << std::endl;
	if (strcmp(fileFormat, "bin") == 0)
	  gsl_matrix_fwrite(dstStream, X);
	else
	  gsl_matrix_fprintf(dstStream, X, "%f");
	fclose(dstStream);  
      }

      // Free
      gsl_matrix_free(X);
    }
    delete mod;
    delete scheme;
    delete field;
    delete stocField;
    gsl_rng_free(r);
    gsl_matrix_free(S);
  }
  
  freeConfig();

  return 0;
}
