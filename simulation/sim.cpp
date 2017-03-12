#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"

using namespace libconfig;


/** \file sim.cpp 
 *  \ingroup examples
 *  \brief Simulate coupled recharge oscillator deterministic flow.
 *
 *  Simulate coupled recharge oscillator deterministic flow.
 */


/** \brief Simulation of coupled recharge oscillator deterministic flow.
 *
 *  Simulation of coupled recharge oscillator deterministic flow.
 *  After parsing the configuration file,
 *  the vector field of the coupled recharge oscillator flow
 *  and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  FILE *dstStream;
  gsl_matrix *X;
  char srcPostfix[256], dstFileName[256], dstPostfix[256];

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

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new coupledRO(&p);
  
  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);

  // Define model (the initial state will be assigned later)
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Define names and open destination file
  sprintf(srcPostfix, "_%s_mu%03d", caseName, (int) (p["mu"] * 100 + 0.1));
  sprintf(dstPostfix, "%s_L%d_spinup%d_dt%d_samp%d", srcPostfix,
	  (int) L, (int) spinup,
	  (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  sprintf(dstFileName, "%s/simulation/sim%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      std::cerr << "Can't open " << dstFileName
		<< " for writing simulation: " << std::endl;;
      perror("");
      return EXIT_FAILURE;
    }

  // Numerical integration
  std::cout << "Integrating simulation..." << std::endl;
  X = gsl_matrix_alloc(1, 1); // False allocation will be corrected
  mod->integrateForward(initState, L, dt, spinup, printStepNum, &X);

  // Write results
  std::cout << "Writing..." << std::endl;
  if (strcmp(fileFormat, "bin") == 0)
    gsl_matrix_fwrite(dstStream, X);
  else
    gsl_matrix_fprintf(dstStream, X, "%f");
  fclose(dstStream);  

  // Free
  gsl_matrix_free(X);
  delete mod;
  delete scheme;
  delete field;
  freeConfig();

  return 0;
}
