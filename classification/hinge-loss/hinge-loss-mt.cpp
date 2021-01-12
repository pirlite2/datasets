// Multi-objective steady-state GP for minimising hinge loss using automatic differentiation -- pir -- 6.1.2021
// Build with USE_DUAL_NUMBERS defined + link with nlopt library * pthread

// REVISION HISTORY:
// From tuning-ad-mt regression program -- pir -- 6.1.2021
// Replaced increment of shared variable g_uNoSubgradients with __atomic_fetch_add(&g_uNoSubgradients, 1, __ATOMIC_ACQ_REL) -- pir -- 8.1.2021
// Used SetConstantNodeIndex_mt() & ExtractConstantTreeNodes_mt() in ObjectiveFunction_mt() to prevent race hazard -- pir -- 8.1.2021

//*****************************************************************************

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include <nlopt.h>

#include <CMatrix/CVector.h>
#include <Evolutionary/CPopulation/CPopulation.h>
#include <Evolutionary/GP/GPNodes.h>
#include <Generic/BinaryFileUtilities/BinaryFileUtilities.h>
#include <Evolutionary/GP/DualNumbers.h>

using namespace std;

//-----------------------------------------------------------------------------
// Global constants

uint32_t g_uNoInitialisationRepeats = 1;
const uint32_t g_uPopulationSize = 100;

const enum enInitialisationMethod_t
	{
	enRampedHalfHalf, enRandomSizes
	}
g_enInitialisationMethod = enRandomSizes;

// Random number base seed values
const int64_t g_lnGP_BaseSeed = 10000L;
const int64_t g_lnLocalSeed = 1867L;

const uint32_t g_uRandomTreeNodeMax = 63;
const uint32_t g_uMaxInitialDepth = 6;
const double g_dTerminalProbability =  0.1;
const double g_dConstantProbability = 0.1;
const double g_dUnaryProbability = 0.1;
const double g_dBinaryProbability = 1.0 - g_dTerminalProbability - g_dConstantProbability - g_dUnaryProbability;
const double g_dTernaryProbability = 0.0;

const double g_dConstantRange = 1.0;
const double g_dConstantPrecision = 0.1;

const enum enOffspringSelect_t
	{
	enOneChild, enBothChildren
	}
g_enOffspringSelect = enBothChildren;

// Crossover parameters
pfnCrossOverOperator_t g_pfnCrossOverOperator = PointCrossOver;
const double g_dCrossOverFunctionNodeBias = 0.9;

// Mutation parameters
const double g_dMutationProbability = 1.0;
pfnMutateOperator_t g_pfnMutateOperator = PointMutate;
const enReplacementTree_t g_enReplacementTree = enGP_FULL_DEPTH;
const uint32_t g_uMutationDepth = 4;

const uint32_t g_uMaxNoTreeEvaluations = 10000;

//-----------------------------------------------------------------------------

// Datasets
CDataset<int32_t> g_TrainingSet;
CDataset<int32_t> g_ValidationSet;
CDataset<int32_t> g_TestSet;

uint32_t g_uNoTreeEvaluations;	// No of tree evaluations

//-----------------------------------------------------------------------------

// NLOpt optimiser settings
const double g_dNLOptRelTol = 1e-4;
const uint32_t g_uNLOptMaxEvaluations = 50;
double g_dLambda = 0.0;

uint32_t g_uNoSubgradients = 0; // No. of sub-gradient uses; used by both threads so incremented using __atomic_fetch_add
uint32_t g_uNoMaxEvaluations = 0;

#define USE_MULTIPLE_THREADS 1

//-----------------------------------------------------------------------------

class CFitnessVector : public CColumnVector
// Definition of fitness vector class
	{
	public:
		CFitnessVector() : CColumnVector(2) {
			/* EMPTY*/
			};
		~CFitnessVector() {
			/* EMPTY*/
			};
	};

//-----------------------------------------------

bool operator < (CFitnessVector& A, CFitnessVector& B)
// Pareto comparison: A < B
	{
	assert(A.NoElements() == B.NoElements());

	// Check A[i] <= B[i] for all i
	for(uint32_t j = 1; j <= A.NoElements(); j++)
		{
		if(A[j] > B[j])
			{
			return false;
			}
		}

	// Check that A[i] < B[i] for at least one i
	for(uint32_t j = 1; j <= A.NoElements(); j++)
		{
		if(A[j] < B[j])
			{
			return true;
			}
		}

	return false;
	} // operator < ()

//-----------------------------------------------

bool operator > (CFitnessVector& A, CFitnessVector& B)
// Pareto comparison: A > B
	{
	assert(A.NoElements() == B.NoElements());

	// Check A[i] >= B[i] for all i
	for(uint32_t j = 1; j <= A.NoElements(); j++)
		{
		if(A[j] < B[j])
			{
			return false;
			}
		}

	// Check that A[i] > B[i] for at least one i
	for(uint32_t j = 1; j <= A.NoElements(); j++)
		{
		if(A[j] > B[j])
			{
			return true;
			}
		}

	return false;
	} // operator > ()

//--------------------------------------------------------------------------------------

void StandardiseDatasets()
// Standardise all attributes excluding the predictor to be zero-mean, unit-variance over training set; adjust validations and tests sets accordingly
	{
	//#define SHOW_STANDARDISATION_CHECK

	const uint32_t uNoTrainingData = g_TrainingSet.NoStoredPatterns();
	const uint32_t uVectorLength = g_TrainingSet.VectorLength();

	// Calculate mean
	double* pMeanVector = new double[uVectorLength];
	for(uint32_t i = 0; i < uVectorLength; i++)
		{
		pMeanVector[i] = 0.0;
		}

	for(uint32_t i = 1; i <= uNoTrainingData; i++)
		{
		CColumnVector patternVector = g_TrainingSet[i];
		for(uint32_t j = 0; j < uVectorLength; j++)
			{
			pMeanVector[j] += patternVector[j + 1];
			}
		}

	for(uint32_t i = 0; i < uVectorLength; i++)
		{
		pMeanVector[i] /= static_cast<double>(uNoTrainingData);
		}

	// Calculate variance
	double* pVarianceVector = new double[uVectorLength];
	for(uint32_t i = 0; i < uVectorLength; i++)
		{
		pVarianceVector[i] = 0.0;
		}

	for(uint32_t i = 1; i <= uNoTrainingData; i++)
		{
		CColumnVector patternVector = g_TrainingSet[i];
		for(uint32_t j = 0; j < uVectorLength; j++)
			{
			const double element = patternVector[j + 1];
			const double mean = pMeanVector[j];
			pVarianceVector[j] += (element - mean) * (element - mean);
			}
		}

	#ifdef SHOW_STANDARDISATION_CHECK
	cout << "Calculated variances" << endl;
	for(uint32_t i = 0; i < uVectorLength; i++)
		{
		pVarianceVector[i] /= static_cast<double>(uNoTrainingData - 1);
		cout << i << " -> " << pVarianceVector[i] << endl;
		}
	cout << endl;
	#endif // SHOW_STANDARDISATION_CHECK

	// Perform standardisation of training set
	for(uint32_t i = 1; i <= uNoTrainingData; i++)
		{
		CColumnVector x = g_TrainingSet[i];
		for(uint32_t j = 1; j <= uVectorLength; j++)
			{
			x[j] = (x[j] - pMeanVector[j - 1]) / sqrt(pVarianceVector[j - 1]);
			}
		g_TrainingSet[i] = x;
		}

	// Perform standardisation of validation set
	for(uint32_t i = 1; i <= g_ValidationSet.NoStoredPatterns(); i++)
		{
		CColumnVector x = g_ValidationSet[i];
		for(uint32_t j = 1; j <= uVectorLength; j++)
			{
			x[j] = (x[j] - pMeanVector[j - 1]) / sqrt(pVarianceVector[j - 1]);
			}
		g_ValidationSet[i] = x;
		}

	// Perform standardisation of test set
	for(uint32_t i = 1; i <= g_TestSet.NoStoredPatterns(); i++)
		{
		CColumnVector x = g_TestSet[i];
		for(uint32_t j = 1; j <= uVectorLength; j++)
			{
			x[j] = (x[j] - pMeanVector[j - 1]) / sqrt(pVarianceVector[j - 1]);
			}
		g_TestSet[i] = x;
		}

	//StandardisePredictor();

	#ifdef SHOW_STANDARDISATION_CHECK
	// Check of training set standardisation!
	CColumnVector meanVector2(uVectorLength);
	Set2NullVector(meanVector2);
	for(uint32_t i = 1; i <= uNoTrainingData; i++)
		{
		CColumnVector patternVector = g_TrainingSet[i];
		meanVector2 = meanVector2 + patternVector;
		}
	meanVector2 = (1.0 / static_cast<double>(uNoTrainingData - 1)) * meanVector2;

	cout << "Checked means" << endl;
	for(uint32_t i = 1; i <= uVectorLength; i++)
		{
		cout << i << " -> " << meanVector2[i] << endl;
		}
	cout << endl;

	CColumnVector sumSquaredVector2(uVectorLength);
	Set2NullVector(sumSquaredVector2);
	for(uint32_t i = 1; i <= uNoTrainingData; i++)
		{
		CColumnVector x = g_TrainingSet[i];
		for(uint32_t j = 1; j <= uVectorLength; j++)
			{
			const double element = x[j];
			const double mean = meanVector2[j];
			sumSquaredVector2[j] += (element - mean) * (element - mean);
			}
		}
	CColumnVector varianceVector2 = (1.0 / static_cast<double>(uNoTrainingData - 1)) * sumSquaredVector2;

	cout << "Checked variances" << endl;
	for(uint32_t j = 1; j <= uVectorLength; j++)
		{
		cout << j << " -> " << varianceVector2[j] << endl;
		}

	exit(1);
	#endif // SHOW_STANDARDISATION_CHECK

	delete[] pMeanVector;
	delete[] pVarianceVector;

	return;
	} // StandardiseDatasets()

//--------------------------------------------------------------------------------------

CFitnessVector(*pfnObjectiveFn)(const CHROMOSOME pstRootNode);	// Prototype for objective function

// For multi-threaded version
struct stThreadArgs_t
	{
	stGPNode_t* pTree;
	double dLoss;
	double dNoNodes;
	};

void* (*pfnObjectiveFn_mt)(void* pvThreadArgs);	// Prototype for multi-threaded objective function

//--------------------------------------------------------------------------------------

CFitnessVector ObjectiveFunction(const CHROMOSOME pTree)
// Returns vector fitness: used only for fitness of the initial population
	{
	CFitnessVector FitnessVector;
	FitnessVector[1] = NoTreeNodes(pTree, true);

	const uint32_t uVectorLength = g_TrainingSet.VectorLength();
	CColumnVector PatternVector(uVectorLength);

	// Calculate the loss over the training set
	double hingeLoss = 0.0;
	double regTerm = 0.0;
	for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
		{
		PatternVector = g_TrainingSet[i];
		const double y = TreeEvaluate(PatternVector, pTree);
		const int32_t label = g_TrainingSet.Tag(i);

		// Calculate hinge loss
		if(label == 0)
			{
			hingeLoss += max(1.0 + y, 0.0);
			}
		else    // label == 1
			{
			hingeLoss += max(1.0 - y, 0.0);
			}

		regTerm += (y * y);   // Label-independent regularisation term: spread of mappings around zero
		}

	FitnessVector[2] = (hingeLoss + (g_dLambda * regTerm)) / static_cast<double>(g_TrainingSet.NoStoredPatterns());

	return FitnessVector;
	} // ObjectiveFunction

//-----------------------------------------------------------------------------

void* ObjectiveFunction_mt(void* pvThreadArgs)
// Calculates vector fitness & returns values in pvThreadArgs: multithreaded version
	{
	struct stThreadArgs_t* threadArgs = static_cast<stThreadArgs_t*>(pvThreadArgs);
	stGPNode_t* pTree = threadArgs->pTree;

	threadArgs->dNoNodes = static_cast<double>(NoTreeNodes(pTree, true));

	const uint32_t uVectorLength = g_TrainingSet.VectorLength();
	CColumnVector PatternVector(uVectorLength);

	// Calculate the loss over the training set
	double hingeLoss = 0.0;
	double regTerm = 0.0;
	for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
		{
		PatternVector = g_TrainingSet[i];
		const double y = TreeEvaluate(PatternVector, pTree);
		const int32_t label = g_TrainingSet.Tag(i);

		// Calculate hinge loss
		if(label == 0)
			{
			hingeLoss += max(1.0 + y, 0.0);
			}
		else    // label == 1
			{
			hingeLoss += max(1.0 - y, 0.0);
			}

		regTerm += (y * y);   // Label-independent regularisation term: spread of mappings around zero
		}

	threadArgs->dLoss = (hingeLoss + (g_dLambda * regTerm))  / static_cast<double>(g_TrainingSet.NoStoredPatterns());

	return NULL;
	} // ObjectiveFunction_mt()

//-----------------------------------------------------------------------------

double nloptObjectiveFn(unsigned n, const double* padConstants, double* grad, void* pvTree)
// NLOpt-compliant function to calculate objective and its gradients; returns the (un-normalised) sum of loss terms
	{
	assert(n > 0);  // Trap trees with no constants

	stGPNode_t* pTree = static_cast<stGPNode_t*>(pvTree);

	double hingeLoss = 0.0;
	double regTerm = 0.0;

	if(grad != NULL)
		{
		for(uint32_t j = 0; j < n; j++)
			{
			grad[j] = 0.0;
			}
		}

	for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
		{
		// Calculate tree response
		CColumnVector x = g_TrainingSet[i];
		stDualNumber_t ym(n);
		ym = TreeEvaluateAugmented(x, padConstants, n, pTree);
		const double y = ym.m_dValue;

		// Calculate hinge loss
		const int32_t label = g_TrainingSet.Tag(i);
		if(label == 0)
			{
			hingeLoss += max(1.0 + y, 0.0);
			}
		else    // label == 1
			{

			hingeLoss += max(1.0 - y, 0.0);
			}

		regTerm += (y * y);   // Label-independent regularisation term: spread of mappings around zero

		if(grad != NULL)
			{
			// Calculate gradient of hinge loss
			for(uint32_t j = 0; j < n; j++)
				{
				if(label == 0)
					{
					if(y >= -1.0)
						{
						if(y == 1.0)
							{
							grad[j] += 0.5 * ym.m_pdGradient[j + 1];
							__atomic_fetch_add(&g_uNoSubgradients, 1, __ATOMIC_ACQ_REL);    // g_uNoSubgradients++;
							}
						else
							{
							grad[j] += ym.m_pdGradient[j + 1];
							}

						grad[j] += g_dLambda * (2.0 * y * ym.m_pdGradient[j + 1]);
						}
					else
						{
						// do nothing! zero gradient
						}
					}
				else    // label == 1
					{

					if(y <= +1.0)
						{
						if(y == 1.0)
							{
							grad[j] -= 0.5 * ym.m_pdGradient[j + 1];
							__atomic_fetch_add(&g_uNoSubgradients, 1, __ATOMIC_ACQ_REL);    // g_uNoSubgradients++;
							}
						else
							{
							grad[j] -= ym.m_pdGradient[j + 1];
							}

						grad[j] += g_dLambda * (2.0 * y * ym.m_pdGradient[j + 1]);
						}
					else
						{
						// do nothing! zero gradient
						}
					}
				}
			}
		}

	//cout << "in nloptObjectiveFn: mean loss = " << (hingeLoss + (g_dLambda * regTerm)) / static_cast<double>(g_TrainingSet.NoStoredPatterns()) << endl;    // DEBUG

	return hingeLoss + (g_dLambda * regTerm);
	} // nloptObjectiveFn()

//-----------------------------------------------------------------------------

double nloptObjectiveFn_mt(unsigned n, const double* padConstants, double* grad, void* pvTree)
// NLOpt-compliant function to calculate objective and its gradients; returns the (un-normalised) sum of loss terms: multithreaded version
	{
	assert(n > 0);  // Trap trees with no constants

	stGPNode_t* pTree = static_cast<stGPNode_t*>(pvTree);

	double hingeLoss = 0.0;
	double regTerm = 0.0;

	if(grad != NULL)
		{
		for(uint32_t j = 0; j < n; j++)
			{
			grad[j] = 0.0;
			}
		}

	for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
		{
		// Calculate tree response
		CColumnVector x = g_TrainingSet[i];
		stDualNumber_t ym(n);
		ym = TreeEvaluateAugmented(x, padConstants, n, pTree);
		const double y = ym.m_dValue;

		// Calculate hinge loss
		const int32_t label = g_TrainingSet.Tag(i);
		if(label == 0)
			{
			hingeLoss += max(1.0 + y, 0.0);
			}
		else    // label == 1
			{

			hingeLoss += max(1.0 - y, 0.0);
			}

		regTerm += (y * y);   // Label-independent regularisation term: spread of mappings around zero

		if(grad != NULL)
			{
			// Calculate gradient of hinge loss
			for(uint32_t j = 0; j < n; j++)
				{
				if(label == 0)
					{
					if(y >= -1.0)
						{
						if(y == 1.0)
							{
							grad[j] += 0.5 * ym.m_pdGradient[j + 1];
							__atomic_fetch_add(&g_uNoSubgradients, 1, __ATOMIC_ACQ_REL);    // g_uNoSubgradients++;
							}
						else
							{
							grad[j] += ym.m_pdGradient[j + 1];
							}

						grad[j] += g_dLambda * (2.0 * y * ym.m_pdGradient[j + 1]);
						}
					else
						{
						// do nothing! zero gradient
						}
					}
				else    // label == 1
					{
					if(y <= +1.0)
						{
						if(y == 1.0)
							{
							grad[j] -= 0.5 * ym.m_pdGradient[j + 1];
							__atomic_fetch_add(&g_uNoSubgradients, 1, __ATOMIC_ACQ_REL);    // g_uNoSubgradients++;
							}
						else
							{
							grad[j] -= ym.m_pdGradient[j + 1];
							}

						grad[j] += g_dLambda * (2.0 * y * ym.m_pdGradient[j + 1]);
						}
					else
						{
						// do nothing! zero gradient
						}
					}
				}
			}
		}

	//cout << "in nloptObjectiveFn: mean loss = " << (hingeLoss + (g_dLambda * regTerm)) / static_cast<double>(g_TrainingSet.NoStoredPatterns()) << endl;    // DEBUG

	return hingeLoss + (g_dLambda * regTerm);
	} // nloptObjectiveFn_mt()

//-----------------------------------------------------------------------------

CFitnessVector ObjectiveFn_ConstOptimise(stGPNode_t* pTree)
// Interfacing function for ordinary GP programs; returns vector fitness after optimising constants
	{
	CFitnessVector FitnessVector;
	FitnessVector[1] = NoTreeNodes(pTree, true);

	const uint32_t noConstants = NoConstantNodes(pTree, false); // Return number of mutable constants; ignore immutable constants
	if(noConstants == 0)
		{
		// Trap pathological case of trees with zero constants - calculate the error over the training set & return
		const uint32_t uVectorLength = g_TrainingSet.VectorLength();
		CColumnVector PatternVector(uVectorLength);

		double dLoss = 0.0;
		for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
			{
			PatternVector = g_TrainingSet[i];
			const double y = TreeEvaluate(PatternVector, pTree);
			const int32_t label = g_TrainingSet.Tag(i);

			if(label == 0)
				{
				dLoss += max(1.0 + y, 0.0);
				}
			else
				{
				// label == 1
				dLoss += max(1.0 - y, 0.0);
				}

			dLoss += g_dLambda * (y * y);
			}

		FitnessVector[2] = dLoss / static_cast<double>(g_TrainingSet.NoStoredPatterns());

		return FitnessVector;
		}

	SetConstantNodeIndex(0);
	double adConstants[noConstants];
	ExtractConstantTreeNodes(pTree, adConstants);

	nlopt_opt Optimiser = nlopt_create(NLOPT_LD_SLSQP, noConstants);
	nlopt_set_min_objective(Optimiser, nloptObjectiveFn, static_cast<void*>(pTree));
	nlopt_set_ftol_rel(Optimiser, g_dNLOptRelTol);
	nlopt_set_maxeval(Optimiser, g_uNLOptMaxEvaluations);

	double minLoss;
	nlopt_result OptimiserResult = nlopt_optimize(Optimiser, adConstants, &minLoss);
	switch(OptimiserResult)
		{
		case NLOPT_SUCCESS:
		case NLOPT_STOPVAL_REACHED:
		case NLOPT_FTOL_REACHED:
		case NLOPT_XTOL_REACHED:
			// Normal termination!
			break;

		case NLOPT_MAXEVAL_REACHED:
		case NLOPT_ROUNDOFF_LIMITED:
			// Acceptable termination
			//cout << "NLOpt terminated because " << nlopt_result_to_string(OptimiserResult) << endl;
			break;

		case NLOPT_MAXTIME_REACHED:
		case NLOPT_FAILURE:
		case NLOPT_INVALID_ARGS:
		case NLOPT_OUT_OF_MEMORY:
		case NLOPT_FORCED_STOP:
			// Abnormal termination
			cout << "NLOpt terminated because " << nlopt_result_to_string(OptimiserResult) << endl;
			ErrorHandler("Optimiser terminated");
			break;

		default:
			cout << OptimiserResult << endl;	// DEBUG
			ErrorHandler("Unrecognised return code from NLOpt");
		}

	//const uint32_t uNoEvaluations = nlopt_get_numevals(Optimiser);
	//cout << "Optimiser needed " << uNoEvaluations << " evaluations for " << noConstants << " dimensional problem" << endl;

	SetIndexedConstants(pTree, adConstants);	// Embed optimise constant values in tree


	//	// DEBUG - Examine values of constants obtained from optimisation
	//	for(uint32_t i = 0; i < noConstants; i++)
	//		{
	//		cout << adConstants[i] << "  ";
	//		}
	//	cout << endl;
	//	// DEBUG


	// Tidy up
	nlopt_destroy(Optimiser);

	FitnessVector[2] = minLoss / static_cast<double>(g_TrainingSet.NoStoredPatterns());

	return FitnessVector;
	} // ObjectiveFn_ConstOptimise()

//-----------------------------------------------------------------------------

void* ObjectiveFn_ConstOptimise_mt(void* pvThreadArgs)
// Interfacing function for multithreaded GP programs; returns vector fitness after optimising constants via pvThreadArgs
	{
	struct stThreadArgs_t* threadArgs = static_cast<stThreadArgs_t*>(pvThreadArgs);
	stGPNode_t* pTree = threadArgs->pTree;

	threadArgs->dNoNodes = static_cast<double>(NoTreeNodes(pTree, true));

	const uint32_t noConstants = NoConstantNodes(pTree, false); // Return number of mutable constants; ignore immutable constants (false)
	if(noConstants == 0)
		{
		// Trap pathological case of trees with zero constants - calculate the error over the training set & return
		const uint32_t uVectorLength = g_TrainingSet.VectorLength();
		CColumnVector PatternVector(uVectorLength);

		double dLoss = 0.0;
		for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
			{
			PatternVector = g_TrainingSet[i];
			const double y = TreeEvaluate(PatternVector, pTree);
			const int32_t label = g_TrainingSet.Tag(i);

			if(label == 0)
				{
				dLoss += max(1.0 + y, 0.0);
				}
			else
				{
				// label == 1
				dLoss += max(1.0 - y, 0.0);
				}

			dLoss += g_dLambda * (y * y);
			}

		threadArgs->dLoss = dLoss / static_cast<double>(g_TrainingSet.NoStoredPatterns());

		return NULL;
		}

	SetConstantNodeIndex_mt(0); // Note: _mt version!
	double adConstants[noConstants];
	ExtractConstantTreeNodes_mt(pTree, adConstants);    // Note: _mt version!

	nlopt_opt Optimiser = nlopt_create(NLOPT_LD_SLSQP, noConstants);
	nlopt_set_min_objective(Optimiser, nloptObjectiveFn_mt, static_cast<void*>(pTree));
	nlopt_set_ftol_rel(Optimiser, g_dNLOptRelTol);
	nlopt_set_maxeval(Optimiser, g_uNLOptMaxEvaluations);

	double minLoss;
	nlopt_result OptimiserResult = nlopt_optimize(Optimiser, adConstants, &minLoss);
	switch(OptimiserResult)
		{
		case NLOPT_SUCCESS:
		case NLOPT_STOPVAL_REACHED:
		case NLOPT_FTOL_REACHED:
		case NLOPT_XTOL_REACHED:
			// Normal termination!
			break;

		case NLOPT_MAXEVAL_REACHED:
		case NLOPT_ROUNDOFF_LIMITED:
			// Acceptable termination
			//cout << "NLOpt terminated because " << nlopt_result_to_string(OptimiserResult) << endl;
			break;

		case NLOPT_MAXTIME_REACHED:
		case NLOPT_FAILURE:
		case NLOPT_INVALID_ARGS:
		case NLOPT_OUT_OF_MEMORY:
		case NLOPT_FORCED_STOP:
			// Abnormal termination
			cout << "NLOpt terminated because " << nlopt_result_to_string(OptimiserResult) << endl;
			ErrorHandler("Optimiser terminated");
			break;

		default:
			cout << OptimiserResult << endl;	// DEBUG
			ErrorHandler("Unrecognised return code from NLOpt");
		}

	//const uint32_t uNoEvaluations = nlopt_get_numevals(Optimiser);
	//cout << "Optimiser needed " << uNoEvaluations << " evaluations for " << noConstants << " dimensional problem" << endl;

	SetIndexedConstants(pTree, adConstants);	// Embed optimise constant values in tree

	//	// DEBUG - Examine values of constants obtained from optimisation
	//	for(uint32_t i = 0; i < noConstants; i++)
	//		{
	//		cout << adConstants[i] << "  ";
	//		}
	//	cout << endl;
	//	// DEBUG

	// Tidy up
	nlopt_destroy(Optimiser);

	threadArgs->dLoss = minLoss / static_cast<double>(g_TrainingSet.NoStoredPatterns());

	return NULL;
	} // ObjectiveFn_ConstOptimise_mt()

//-----------------------------------------------------------------------------

double ValidationSetEvaluation(CHROMOSOME pTree)
// Returns zero-one error over the validation set
	{
	const uint32_t  uVectorLength = g_ValidationSet.VectorLength();
	CColumnVector PatternVector(uVectorLength);

	// Calculate the zero-one loss over the training set
	uint32_t uNoErrors = 0;
	for(uint32_t i = 1; i <= g_ValidationSet.NoStoredPatterns(); i++)
		{
		PatternVector = g_ValidationSet[i];
		const double y = TreeEvaluate(PatternVector, pTree);
		int32_t tag = g_ValidationSet.Tag(i);

		if(tag == 0)
			{
			if(y >= 0)
				{
				uNoErrors++;
				}
			}
		else
			{
			assert(tag == 1);
			if(y < 0)
				{
				uNoErrors++;
				}
			}
		}

	return static_cast<double>(uNoErrors) / static_cast<double>(g_ValidationSet.NoStoredPatterns());
	} // ValidationSetEvaluation()

//-----------------------------------------------------------------------------

double TestSetEvaluation(CHROMOSOME pTree)
// Returns mean squared error over the test set
	{
	const uint32_t  uVectorLength = g_TrainingSet.VectorLength();
	CColumnVector PatternVector(uVectorLength);

	// Calculate the zero-one loss over the training set
	uint32_t uNoErrors = 0;
	for(uint32_t i = 1; i <= g_TestSet.NoStoredPatterns(); i++)
		{
		PatternVector = g_TestSet[i];
		const double y = TreeEvaluate(PatternVector, pTree);
		int32_t tag = g_TestSet.Tag(i);

		if(tag == 0)
			{
			if(y >= 0)
				{
				uNoErrors++;
				}
			}
		else
			{
			assert(tag == 1);
			if(y < 0)
				{
				uNoErrors++;
				}
			}
		}

	return static_cast<double>(uNoErrors) / static_cast<double>(g_TestSet.NoStoredPatterns());
	} // TestEvaluation()

//-----------------------------------------------------------------------------

stGP_Parameters_t* g_pstGP_Parameters = NULL;

//*****************************************************************************

int main(int argc, char* argv[])
	{
	// Load all datasets & repeat parameter
	// argv[1] = filename (e.g. australian)
	// argv[2] = fold index [0..9]
    // argv[3] = attempt \in [1..30]
	if(argc != 4)
		{
		ErrorHandler("Expecting 3 command line parameters");
		}

	char szFilename[128];
	strcpy(szFilename, argv[1]);

	int32_t nFoldIndex = strtol(argv[2], NULL, 10);
	assert((nFoldIndex >= 0) and (nFoldIndex <= 9));

//	g_dLambda = strtod(argv[3], NULL);
//	assert(g_dLambda >= 0.0);
//	cout << "lambda = " << g_dLambda << endl;   // TEST

	int32_t nAttempt = strtol(argv[3], NULL, 10);
	assert((nAttempt >= 1) and (nAttempt <= 30));

	char szTrainingsSetName[256];
	sprintf(szTrainingsSetName, "../%s/%s_training-%d.dat", szFilename, szFilename, nFoldIndex);
	g_TrainingSet.Load(szTrainingsSetName);

	char szValidationSetName[256];
	sprintf(szValidationSetName, "../%s/%s_validation-%d.dat", szFilename, szFilename, nFoldIndex);
	g_ValidationSet.Load(szValidationSetName);

	char szTestSetName[256];
	sprintf(szTestSetName, "../%s/%s_test-%d.dat", szFilename, szFilename, nFoldIndex);
	g_TestSet.Load(szTestSetName);

	cout << "opening: " << szTrainingsSetName << endl;	// TEST
	cout << "opening: " << szValidationSetName << endl;	// TEST
	cout << "opening: " << szTestSetName << endl;	// TEST

	StandardiseDatasets();

	g_uNoInitialisationRepeats = (static_cast<uint32_t>(nFoldIndex) + 1) + (10 * (nAttempt - 1));
	g_pstGP_Parameters = new stGP_Parameters_t(g_lnGP_BaseSeed, g_uNoInitialisationRepeats);

	//-----------------------------------------------

	// Set MOGP parameters
	const uint32_t uVectorLength  = g_TrainingSet.VectorLength();
	g_pstGP_Parameters->SetVectorLength(uVectorLength);
	g_pstGP_Parameters->SetMutationDepth(g_uMutationDepth);

	// Set constant parameters
	g_pstGP_Parameters->SetConstantRange(g_dConstantRange);
	g_pstGP_Parameters->SetConstantPrecision(g_dConstantPrecision);

	// Set node selection probabilities
	g_pstGP_Parameters->SetNodeSelectionProbabilities(g_dTerminalProbability, g_dConstantProbability, g_dUnaryProbability, g_dBinaryProbability, g_dTernaryProbability);

	//-----------------------------------------------

	CPopulation<CHROMOSOME, CFitnessVector> Population(g_uPopulationSize);

	CUniform2* pMutationSelector = NULL;
	CUniform2* pOffspringSelector = NULL;
	CUniform2* pRandomInitialTreeSizeSelector = NULL;

	assert(g_uNoInitialisationRepeats >= 1);
	int64_t lnLocalSeed = g_lnLocalSeed;
	for(uint32_t i = 1; i <= g_uNoInitialisationRepeats; i++)
		{
		// Initialise mutation and offspring selectors
		if(pMutationSelector != NULL)
			{
			delete pMutationSelector;
			}
		lnLocalSeed++;
		pMutationSelector = new CUniform2(lnLocalSeed);

		if(pOffspringSelector != NULL)
			{
			delete pOffspringSelector;
			}
		lnLocalSeed++;
		pOffspringSelector = new CUniform2(lnLocalSeed);

		if(pRandomInitialTreeSizeSelector != NULL)
			{
			delete pRandomInitialTreeSizeSelector;
			}
		lnLocalSeed++;
		pRandomInitialTreeSizeSelector = new CUniform2(lnLocalSeed);

		// Delete previous population (apart from the first time through the loop)
		if(i > 1)
			{
			for(uint32_t j = 1; j <= g_uPopulationSize; j++)
				{
				DeleteChromosome(Population[j]);
				}
			}

		//Creation of initial population (half full-depth , half random depth)
		cout << "Creating initial population (" << i << ")..." << endl;
		for(uint32_t j = 1; j <= (g_uPopulationSize / 2); j++)
			{
			CHROMOSOME pTree;
			if(g_enInitialisationMethod == enRampedHalfHalf)
				{
				// Create full-depth trees
				const bool bCreateFullDepthTrees = true;
				pTree = CreateRandomTree1(g_uMaxInitialDepth, bCreateFullDepthTrees);
				}
			else
				{
				// Create random depth trees
				const double dTreeSizeSelector = pRandomInitialTreeSizeSelector->NextVariate();
				const double dTreeSize = (static_cast<double>(g_uRandomTreeNodeMax - 1) * dTreeSizeSelector) + 1.0;
				const uint32_t uTreeSize = static_cast<uint32_t>(round(dTreeSize));
				pTree = CreateRandomTree2(uTreeSize);
				}

			Population[j] = pTree;
			if(i == g_uNoInitialisationRepeats)
				{
				Population.Fitness(j) = ObjectiveFunction(Population[j]);
				}
			}

		for(uint32_t j = ((g_uPopulationSize / 2) + 1); j <= (g_uPopulationSize + 2); j++)
			{
			CHROMOSOME pTree;
			if(g_enInitialisationMethod == enRampedHalfHalf)
				{
				// Create mixed-depth trees
				const bool bCreateFullDepthTrees = false;
				pTree = CreateRandomTree1(g_uMaxInitialDepth, bCreateFullDepthTrees);
				}
			else
				{
				// Create random depth trees
				const double dTreeSizeSelector = pRandomInitialTreeSizeSelector->NextVariate();
				const double dTreeSize = (static_cast<double>(g_uRandomTreeNodeMax - 1) * dTreeSizeSelector) + 1.0;
				const uint32_t uTreeSize = static_cast<uint32_t>(round(dTreeSize));
				pTree = CreateRandomTree2(uTreeSize);
				}

			Population[j] = pTree;
			if(i == g_uNoInitialisationRepeats)
				{
				Population.Fitness(j) = ObjectiveFunction(Population[j]);
				}
			}
		}

	Population.MOSort(enASCENDING);

	// Print initial population
	cout << "Initial population..." << endl;
	for(uint32_t i = 1; i <= g_uPopulationSize; i++)
		{
		cout << i
			 << "   Node count = "
			 << Population.Fitness(i)
			 [1]
			 << ", Mean squared error = "
			 << Population.Fitness(i)[2]
			 << ",  Rank = "
			 << Population.Rank(i)
			 << endl;
		}
	cout << endl;

	//-----------------------------------------------

	// Genetic evolution loop
	uint32_t uMinTrainIndex;
	double dMinTrainError;
	uint32_t uNoIterations = 0;
	g_uNoTreeEvaluations = 0;

	#ifdef USE_DUAL_NUMBERS
	pfnObjectiveFn = ObjectiveFn_ConstOptimise;
	pfnObjectiveFn_mt = ObjectiveFn_ConstOptimise_mt;
	#else
	pfnObjectiveFn = ObjectiveFunction;
	pfnObjectiveFn_mt = ObjectiveFunction_mt;
	#endif // USE_DUAL_NUMBERS

	cout << "Entering evolutionary loop..." << endl;

	do
		{
		uNoIterations++;
		if((uNoIterations % 1000) == 0)
			{
			cout << "No of iterations = " << uNoIterations << endl;
			cout << "Min training error = " << dMinTrainError << endl;		// TEST
			}

		uint32_t uParent1Index;
		uint32_t uParent2Index;
		Population.SelectParents(uParent1Index, uParent2Index);

		// Perform crossover & mutation
		CHROMOSOME Parent1Chromosome = Population[uParent1Index];
		CHROMOSOME Parent2Chromosome = Population[uParent2Index];
		CHROMOSOME Child1Chromosome;
		CHROMOSOME Child2Chromosome;
		g_pfnCrossOverOperator(Parent1Chromosome, Parent2Chromosome, &Child1Chromosome, &Child2Chromosome, g_dCrossOverFunctionNodeBias);

		const double dMutateSelector = pMutationSelector->NextVariate();
		if(dMutateSelector <= g_dMutationProbability)
			{
			g_pfnMutateOperator(&Child1Chromosome, g_enReplacementTree);
			g_pfnMutateOperator(&Child2Chromosome, g_enReplacementTree);
			}

		// Evaluate child fitness & insert into population
		if(enOneChild == g_enOffspringSelect)
			{
			// Select which child to keep
			const double dOffspringSelector = pOffspringSelector->NextVariate();
			if(dOffspringSelector < 0.5)
				{
				// Evaluate child fitness & insert into child population
				CFitnessVector Child1Fitness = pfnObjectiveFn(Child1Chromosome);
				g_uNoTreeEvaluations++;
				Population.InsertChild(Child1Chromosome, Child1Fitness);

				DeleteChromosome(Child2Chromosome);
				}
			else
				{
				// Evaluate child fitness & insert into child population
				CFitnessVector Child2Fitness = pfnObjectiveFn(Child2Chromosome);
				g_uNoTreeEvaluations++;
				Population.InsertChild(Child2Chromosome, Child2Fitness);

				DeleteChromosome(Child1Chromosome);
				}
			}
		else
			{
			// Add both children to population

			#ifdef USE_MULTIPLE_THREADS
			// Twin thread implementation
			struct stThreadArgs_t threadArgs = {Child1Chromosome, NAN, NAN};
			void* functionArg = static_cast<void*>(&threadArgs);
			pthread_t thread;
			pthread_create(&thread, NULL, pfnObjectiveFn_mt, functionArg);
			CFitnessVector Child2Fitness = pfnObjectiveFn(Child2Chromosome);

			pthread_join(thread, NULL);

			CFitnessVector Child1Fitness;
			Child1Fitness[1] = threadArgs.dNoNodes;
			Child1Fitness[2] = threadArgs.dLoss;
			#else
			// Single thread implementation
			CFitnessVector Child1Fitness = pfnObjectiveFn(Child1Chromosome);
			CFitnessVector Child2Fitness = pfnObjectiveFn(Child2Chromosome);
			#endif

			g_uNoTreeEvaluations += 2;

			Population.AppendChildren(Child1Chromosome, Child1Fitness, Child2Chromosome, Child2Fitness);
			}

		// Sort the new population
		Population.MOSort(enASCENDING);

		// Find smallest training set error
		dMinTrainError = INFINITY;
		uMinTrainIndex = 1;
		for(uint32_t i = 1; i <= g_uPopulationSize; i++)
			{
			if(Population.Fitness(i)[2] < dMinTrainError)
				{
				dMinTrainError = Population.Fitness(i)[2];
				uMinTrainIndex = i;
				}
			}
		}
	while(g_uNoTreeEvaluations < g_uMaxNoTreeEvaluations);

	//-------------------------------------------------------

	// Print final population
	cout << "Final resorted population..." << endl;
	for(uint32_t i = 1; i <= g_uPopulationSize; i++)
		{
		cout << i
			 << " -> ("
			 << Population.Fitness(i)
			 [1]
			 << ", "
			 << Population.Fitness(i)[2]
			 << ")   rank = "
			 << Population.Rank(i)
			 << endl;
		}
	cout << endl;

	//--------------------------------------------------------

	cout << "No of tree evaluations = " << g_uNoTreeEvaluations << endl;

	//-----------------------------------------------

	// Log training data
	char szLogfileName[128];
	sprintf(szLogfileName, "%s-training.log", szFilename);
	FILE* pLogfile = fopen(szLogfileName, "a");
	if(pLogfile == NULL)
		{
		ERROR_HANDLER("Unable to open log file");
		}

	// Best training individual
	stGPNode_t* pBestTrainedIndividual = Population[uMinTrainIndex];
	cout << "Smallest training error = " << dMinTrainError << " with ";
	cout << NoTreeNodes(pBestTrainedIndividual, true) << " nodes & depth = ";
	cout << MaxTreeDepth(pBestTrainedIndividual);
	cout << " with test error = " << TestSetEvaluation(Population[uMinTrainIndex]);
	cout << endl;

	fprintf(pLogfile, "%lf,", dMinTrainError);
	fprintf(pLogfile, "%u,", NoTreeNodes(pBestTrainedIndividual, true));
	fprintf(pLogfile, "%u,", MaxTreeDepth(pBestTrainedIndividual));
	fprintf(pLogfile, "%lf,", TestSetEvaluation(Population[uMinTrainIndex]));

	// Find individual with the smallest validation error
	uint32_t uMinValidationIndex = UINT32_MAX;
	double dMinValidationError = INFINITY;
	for(uint32_t i = 1; i <= g_uPopulationSize; i++)
		{
		const double validationError = ValidationSetEvaluation(Population[i]);
		if(validationError < dMinValidationError)
			{
			dMinValidationError = validationError;
			uMinValidationIndex = i;
			}
		}

	cout << "Smallest validation error = " << dMinValidationError << " for individual " << uMinValidationIndex << endl;
	fprintf(pLogfile, "%lf,%u,", dMinValidationError, uMinValidationIndex);

	// Get test error of best validation error model
	const double dTestError = TestSetEvaluation(Population[uMinValidationIndex]);
	const uint32_t uNoBestValidationNodes = NoTreeNodes(Population[uMinValidationIndex], true);
	cout << "Test error = " << dTestError << " with " << uNoBestValidationNodes << " nodes";
	const uint32_t uBestValidationRank = Population.Rank(uMinValidationIndex);
	cout << " & rank = " << uBestValidationRank << endl;

	fprintf(pLogfile, "%lf,%u,", dTestError, uNoBestValidationNodes);
	fprintf(pLogfile, "%u\n", uBestValidationRank);

	fclose(pLogfile);

	// Write results to report file
	char szReportName[128];
	sprintf(szReportName, "%s_report", szFilename);
	FILE* pReportFile = fopen(szReportName, "a");
	if(pReportFile == NULL)
		{
		ERROR_HANDLER("Unable to open report file");
		}

	fprintf(pReportFile, "%d, %lf, %lf, %u, %u\n", nFoldIndex, dMinValidationError, dTestError, uNoBestValidationNodes, uBestValidationRank);
	fclose(pReportFile);

	// Sub-gradient use reporting
	if(g_uNoSubgradients > 0)
		{
		char szSugradientFilename[128];
		sprintf(szSugradientFilename, "%s-subgradients-%d", szFilename, nFoldIndex);
		FILE* pSubgradientFile = fopen(szSugradientFilename, "a");
		fprintf(pSubgradientFile, "%d\n", g_uNoSubgradients);
		fclose(pSubgradientFile);
		}

	#ifdef USE_DUAL_NUMBERS
	cout << "fraction of iterations exceeding iteration limit = " << static_cast<double>(g_uNoMaxEvaluations) / static_cast<double>(g_uMaxNoTreeEvaluations) << endl;
	#endif // USE_DUAL_NUMBERS

	// Tidy-up
	delete pMutationSelector;
	delete g_pstGP_Parameters;

	return EXIT_SUCCESS;
	} // main()

//*****************************************************************************













