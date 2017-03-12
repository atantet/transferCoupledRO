#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <cstring>
#include <vector>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl_extension.hpp>
#include <gsl/gsl_blas.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"
#include "../cfg/coupledRO.hpp"

using namespace libconfig;


void handler(const char *reason, const char *file, int line, int gsl_errno);
  
void getArgContSel(FILE *srcStream, gsl_vector *initCont, size_t nCont,
		   gsl_vector_uint *argContSel);

/** \file getPhaseDiffuion.cpp 
 *  \brief Get phase diffusion in coupled recharge oscillator.
 *
 * Get phase diffusion from periodic orbit continuation 
 * in the coupled recharge oscillator.
 */


/** \brief Get phase diffusion in coupled recharge oscillator.
 *
 * Get phase diffusion from periodic orbit continuation 
 * in the coupled recharge oscillator.
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

  double T;
  double phi;
  gsl_vector_view init;
  char fileName[256], srcPostfix[256], dstPostfix[256], contPostfix[256];
  FILE *srcStream, *srcStreamVecLeft, *srcStreamVecRight, *srcStreamExp,
    *dstStream;
  gsl_vector_complex *FloquetExp = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *FloquetVecLeft = gsl_matrix_complex_alloc(dim, dim);
  gsl_matrix_complex *FloquetVecRight = gsl_matrix_complex_alloc(dim, dim);
  gsl_vector_complex_view vLeft, vRight;
  gsl_vector_view vLeftReal, vRightReal;
  gsl_matrix *Q = gsl_matrix_alloc(dim, dim);


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

  // Define diffusion matrix
  gsl_matrix_set_zero(Q);
  gsl_matrix_set(Q, dim - 1, dim - 1, p["sigmah2"]);
  if (verbose) {
    std::cout << "Diffusion matrix Q = " << std::endl;
    gsl_matrix_fprintf(stdout, Q, "%lf");
    std::cout << std::endl;
  }

  // Loop over various parameters
  size_t nCont = 11;
  gsl_vector_uint *argContSel = gsl_vector_uint_alloc(nCont);
  // double eta2Min = 0.5; double eta2Max = 0.7; size_t neta2 = 11;
  // double rMin = 0.1; double rMax = 0.3; size_t nr = 11;
  // double gammaMin = 0.3; double gammaMax = 0.5; size_t ngamma = 11;
  double eta2Min = 0.54; double eta2Max = 0.7; size_t neta2 = 9;
  double rMin = 0.24; double rMax = 0.3; size_t nr = 4;
  double gammaMin = 0.44; double gammaMax = 0.5; size_t ngamma = 4;
  double eta2Step = (eta2Max - eta2Min) / (neta2 - 1);
  double rStep = (rMax - rMin) / (nr - 1);
  double gammaStep = (gammaMax - gammaMin) / (ngamma - 1);
  for (size_t ieta2 = 0; ieta2 < neta2; ieta2++) {
    p["eta2"] = eta2Min + eta2Step * ieta2;
    for (size_t ir = 0; ir < nr; ir++) {
      p["r"] = rMin + rStep * ir;
      for (size_t igamma = 0; igamma < ngamma; igamma++) {
        p["gamma"] = gammaMin + gammaStep * igamma;
	std::cout << "\nGetting phase diffusion for eta2 = " << p["eta2"]
		  << ", r = " << p["r"] << ", gamma = " << p["gamma"]
		  << std::endl;

	// Open source files
   	sprintf(dstPostfix, "%s_eta2%04d_r%04d_gamma%04d%s", srcPostfix,
		(int) (p["eta2"] * 1000 + 0.1), (int) (p["r"] * 1000 + 0.1),
		(int) (p["gamma"] * 1000 + 0.1), contPostfix);
	sprintf(fileName, "%s/continuation/poCont/poState%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStream = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading solution: ",
		  fileName);
	  perror("");
	  continue;
	}

	sprintf(fileName, "%s/continuation/poVecLeft/poVecLeft%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStreamVecLeft = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading left Floquet vectors: ",
		  fileName);
	  perror("");
	  continue;
	}
  
	sprintf(fileName, "%s/continuation/poVecRight/poVecRight%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStreamVecRight = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading right Floquet vectors: ",
		  fileName);
	  perror("");
	  continue;
	}
  
	sprintf(fileName, "%s/continuation/poExp/poExp%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(srcStreamExp = fopen(fileName, "rb"))) {
	  fprintf(stderr, "Can't open %s for reading Floquet exponents: ",
		  fileName);
	  perror("");
	  continue;
	}

	sprintf(fileName, "%s/continuation/phase/phaseDiffusion%s.%s",
		resDir, dstPostfix, fileFormat);
	if (!(dstStream = fopen(fileName, "w"))) {
	  fprintf(stderr, "Can't open %s for writing Floquet exponents: ",
		  fileName);
	  perror("");
	  continue;
	}

	// Get minimum and maximum parameter
	getArgContSel(srcStream, initCont, nCont, argContSel);

	// Read Floquet analysis
	size_t readCount = 0;
	size_t readCountSel = 0;
	if (strcmp(fileFormat, "bin") == 0) {
	  // First count the number of 
	  while ((!gsl_vector_fread(srcStream, initCont))
		 && (readCountSel < nCont)) {
	    if (readCount == gsl_vector_uint_get(argContSel, readCountSel)) {
	      // Read Floquet elements
	      gsl_vector_complex_fread(srcStreamExp, FloquetExp);
	      gsl_matrix_complex_fread(srcStreamVecLeft, FloquetVecLeft);
	      gsl_matrix_complex_fread(srcStreamVecRight, FloquetVecRight);
	    
	      // Set control parameter and period
	      p["mu"] = gsl_vector_get(initCont, dim);
	      T = gsl_vector_get(initCont, dim + 1);
	      init = gsl_vector_subvector(initCont, 0, dim);
	      std::cout << "Computing phase diffusion coefficient for mu = "
			<< p["mu"] << " with period " << T << std::endl;
	      // Get total number of time steps in period
	      nt = (size_t) (ceil(T / dt) + 0.1);
	      // Get time step adapted to period
	      dt = T / nt;
	      // Get left and right Floquet vectors in the direction of
	      // the flow (must have been sorted before)
	      vLeft = gsl_matrix_complex_column(FloquetVecLeft, 0);
	      vRight = gsl_matrix_complex_column(FloquetVecRight, 0);
	      vLeftReal = gsl_vector_complex_real(&vLeft.vector);
	      vRightReal = gsl_vector_complex_real(&vRight.vector);
	      
	      // Define field
	      vectorField *field = new coupledRO(&p);
	    
	      // Define linearized field
	      linearField *Jacobian = new JacobianCoupledRO(&p);

	      // Define numerical scheme
	      numericalScheme *scheme = new RungeKutta4(dim);

	      // Define model (the initial state will be assigned later)
	      model *mod = new model(field, scheme);

	      // Define linearized model 
	      fundamentalMatrixModel *linMod
		= new fundamentalMatrixModel(mod, Jacobian);

	      gsl_matrix *xt = gsl_matrix_alloc(1, 1);
	      std::vector<gsl_matrix *> Mts;
	      std::vector<gsl_matrix *> Qs;

	      // Integrate fundamental matrix from 0 to T
	      if (verbose)
		std::cout << "Integrating from 0 to " << T
			  << "..." << std::endl;
	      linMod->integrateRange(&init.vector, nt, dt, &xt, &Mts, 0);

	      if (verbose) {
		std::cout << "M(T, 0) = " << std::endl;
		gsl_matrix_fprintf(stdout, Mts.at(0), "%lf");
		std::cout << "FloquetExp = " << std::endl;
		gsl_vector_complex_fprintf(stdout, FloquetExp, "%lf");
		std::cout << "FloquetVecLeft = " << std::endl;
		gsl_matrix_complex_fprintf(stdout, FloquetVecLeft, "%lf");
		std::cout << "FloquetVecRight = " << std::endl;
		gsl_matrix_complex_fprintf(stdout, FloquetVecRight, "%lf");
	      }

	      // Get time-dependent diffusion matrix
	      if (verbose)
		std::cout << "Getting diffusion matrices..." << std::endl;
	      Qs.resize(Mts.size());
	      for (size_t r = 0; r < Qs.size(); r++) {
		Qs.at(r) = gsl_matrix_alloc(dim, dim);
		gsl_matrix_memcpy(Qs.at(r), Q);
	      }

	      // Get phase diffusion
	      if (verbose)
		std::cout << "Compute phase diffusion..." << std::endl;
	      phi = getPhaseDiffusion(&Qs, &Mts, &vLeftReal.vector,
				      &vRightReal.vector, dt);
	      // if (verbose)
	      std::cout << "phi = " << phi << std::endl;

	      // Write phase diffusion
	      double buf = p["mu"];
	      fwrite(&buf, sizeof(double), 1, dstStream);
	      fwrite(&phi, sizeof(double), 1, dstStream);
	      fflush(dstStream);

	      readCountSel++;

	      // Free
	      gsl_matrix_free(xt);
	      for (size_t r = 0; r < Mts.size(); r++)
		gsl_matrix_free(Mts.at(r));
	      for (size_t r = 0; r < Qs.size(); r++)
		gsl_matrix_free(Qs.at(r));
	      delete linMod;
	      delete mod;
	      delete scheme;
	      delete Jacobian;
	      delete field;
	    }
	    readCount++;
	  }
	}
	// else
	//   while (!(gsl_vector_fscanf(srcStream, initCont)
	// 	   || gsl_vector_complex_fscanf(srcStreamExp, FloquetExp)
	// 	   || gsl_matrix_complex_fscanf(srcStreamVec, FloquetVec))) {
	//     getPhaseDiffusion();
	//   }

	// Close src files
	fclose(srcStreamExp);
	fclose(srcStreamVecLeft);
	fclose(srcStreamVecRight);
	fclose(srcStream);
	fclose(dstStream);
      }
    }
  }
  gsl_vector_uint_free(argContSel);
  gsl_vector_complex_free(FloquetExp);
  gsl_matrix_complex_free(FloquetVecLeft);
  gsl_matrix_complex_free(FloquetVecRight);
  gsl_matrix_free(Q);
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


void
getArgContSel(FILE *srcStream, gsl_vector *initCont, size_t nCont,
	      gsl_vector_uint *argContSel)
{
  gsl_vector *contRng = gsl_vector_alloc(maxPred);
  size_t readCount = 0;
  double minCont = 1.e27;
  double maxCont = -1.e27;
  size_t arg;
  double conti, distCont, distContj, deltaCont;

  // Read all continuation parameters and get min and max
  while (!gsl_vector_fread(srcStream, initCont)
	 && (readCount < maxPred)) {
    conti = gsl_vector_get(initCont, dim);
    gsl_vector_set(contRng, readCount, conti);
    if (conti > maxCont)
      maxCont = conti;
    if (conti < minCont)
      minCont = conti;
    readCount++;
  }

  // Get delta
  deltaCont = (maxCont - minCont) / (nCont - 1);

  // Get parameter indices closest to regular grid
  for (size_t i = 0; i < nCont; i++) {
    // Find the closest to minCont + i * deltaCont
    arg = 0;
    distCont = fabs(minCont + i * deltaCont - gsl_vector_get(contRng, 0));
    for (size_t j = 1; j < readCount; j++) {
      distContj = fabs(minCont + i*deltaCont
		       - gsl_vector_get(contRng, j));
      if (distContj < distCont) {
	arg = j;
	distCont = distContj;
      }
    }
    // Save
    gsl_vector_uint_set(argContSel, i, arg);
  }
  // Free
  gsl_vector_free(contRng);
  // Rewind for future read of stream
  rewind(srcStream);

  return;
}
