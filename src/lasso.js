/*
 Least Attribute Shrinkage and Selection Operator (LASSO) Regression

 Javascript version Based on Lasso4j and (2008) Regularization Paths 
 for Generalized Linear Models via Coordinate Descent. 
 http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf
 */


// --------------------
// MathUtil
// --------------------

function MathUtil() {}

MathUtil.getAvg = function(arr) {
	var sum = 0;
	for (var item=0; item<arr.length; item++) {
		sum += Number(arr[item]);
	}
	return sum / arr.length;
};

MathUtil.getStg = function(arr) {
	return MathUtil.getStd(arr, MathUtil.getAvg(arr));
};

MathUtil.getStd = function(arr, avg) {
	var sum = 0;
	for (var item=0; item<arr.length; item++) {
		sum += Math.pow(arr[item] - avg, 2);
	}
	return Math.sqrt(sum / arr.length);
};

MathUtil.getDotProduct = function(vector1, vector2) {
	var product = 0;
	for (var i = 0; i < vector1.length; i++) {
		product += vector1[i] * vector2[i];
	}
	return product;
};

// Divides the second vector from the first one (vector1[i] /= val)
MathUtil.divideInPlace = function(vector, val) {
	var length = vector.length;
	for (var i = 0; i < length; i++) {
		vector[i] /= val;
	}
};

MathUtil.allocateDoubleMatrix = function(m, n) {
	var mat = [];
	for (var i=0; i<m; i++) {
		var row = [];
		for (var j=0; j<n; j++) {
			row.push(0);
		}
		mat.push(row);
	}
	return mat;
};

MathUtil.initArray = function(m) {
	return Array.apply(null, {length: m}).map(function() { return 0; });
};

MathUtil.getFormattedDouble = function(x, digs) {
	return Math.round(x*(Math.pow(10,digs)))/Math.pow(10,digs);
};


// --------------------
// LassoFit
// --------------------

function LassoFit(numberOfLambdas, maxAllowedFeaturesAlongPath, numFeatures) { 
	// Number of lambda values
	this.numberOfLambdas = numberOfLambdas;

	// Intercepts
	this.intercepts = [];

	// Pointers to compressed weights
	this.indices = [];

	// Number of weights for each solution
	this.numberOfWeights = [];

	// Number of non-zero weights for each solution
	this.nonZeroWeights = MathUtil.initArray(numberOfLambdas);

	// The value of lambdas for each solution
	this.lambdas = [];

	// R^2 value for each solution
	this.rsquared = [];

	// Total number of passes over data
	this.numberOfPasses = null;

	// Compressed weights for each solution	
	this.compressedWeights =
		MathUtil.allocateDoubleMatrix(this.numberOfLambdas,
										 maxAllowedFeaturesAlongPath);

	this.numFeatures = numFeatures;
}

LassoFit.prototype.getWeights = function(lambdaIdx) {
	var weights = [];
	for (var i = 0; i < this.numberOfWeights[lambdaIdx]; i++) {
		weights[this.indices[i]] = this.compressedWeights[lambdaIdx][i];
	}
	return weights;
};

LassoFit.prototype.toString = function() {
	var sb = "";
	var numberOfSolutions = this.numberOfLambdas;
	sb += "Compression R2 values:\n";
	sb += "i\tnonzero weights\trsquare\tlambda\n";
	for (var i = 0; i < numberOfSolutions; i++) {
		sb += ((i + 1) + "\t" + this.nonZeroWeights[i] + "\t" +
				MathUtil.getFormattedDouble(this.rsquared[i], 4) + "\t"
				+ MathUtil.getFormattedDouble(this.lambdas[i], 5) + "\n");
	}
	return sb;
};


// --------------------
// LassoFitGenerator
// --------------------

function LassoFitGenerator() {
	// This module shouldn't consume more than 8GB of memory
	this.MAX_AVAILABLE_MEMORY = 8 * 1024 * 1024 * 1024;

	// In order to speed up the compression, we limit the number of
	// observations,
	// but this limit is dependent on the number of features that we should
	// learn
	// their weights. In other words, for learning weights of more features, we
	// need more observations.
	this.MAX_OBSERVATIONS_TO_FEATURES_RATIO = 100;
	
	this.EPSILON = 1.0e-6;

	// The default number of lambda values to use
	this.DEFAULT_NUMBER_OF_LAMBDAS = 100;

	// Convergence threshold for coordinate descent
	// Each inner coordination loop continues until the relative change
	// in any coefficient is less than this threshold
	this.CONVERGENCE_THRESHOLD = 1.0e-4;

	this.SMALL = 1.0e-5;
	this.MIN_NUMBER_OF_LAMBDAS = 5;
	this.MAX_RSQUARED = 0.99999;

	this.targets = [];
	this.observations = [];
	this.numFeatures = 0;
	this.numObservations = 0;
}

LassoFitGenerator.prototype.getMaxAllowedObservations = function(maxNumFeatures) {
	/*
	var maxObservations = Math.floor(this.MAX_AVAILABLE_MEMORY / maxNumFeatures / (Number.MAX_SAFE_INTEGER / 8));
	return maxObservations;
	 */
	return maxNumFeatures * 999999999;
};

LassoFitGenerator.prototype.init = function(maxNumFeatures, numObservations) {
	this.numFeatures = maxNumFeatures;

	if (numObservations > this.getMaxAllowedObservations(maxNumFeatures)) {
		throw "Number of observations (" +
			numObservations + ") exceeds the maximum allowed number: " + 
			this.getMaxAllowedObservations(maxNumFeatures);
	}
	this.numObservations = numObservations;
	this.observations = [];
	for (var t = 0; t < maxNumFeatures; t++) {
		this.observations[t] = MathUtil.initArray(this.numObservations);
	}
	this.targets = MathUtil.initArray(this.numObservations);
};

LassoFitGenerator.prototype.setNumberOfFeatures = function(numFeatures) {
	this.numFeatures = numFeatures;
};

LassoFitGenerator.prototype.setFeatureValues = function(idx, values) {
	for (var i = 0; i < values.length; i++) {
		this.observations[idx][i] = values[i];
	}
};

LassoFitGenerator.prototype.getFeatureValues = function(idx) {
	return this.observations[idx];
};

LassoFitGenerator.prototype.setObservationValues = function(idx, values) {
	for (var f = 0; f < this.numFeatures; f++) {
		if (isNaN(values[f])) {
			debugger;
		}
		this.observations[f][idx] = Number(values[f]);
	}
};

LassoFitGenerator.prototype.getLassoFit = function(maxAllowedFeaturesPerModel) {
	var startTime = (new Date()).getTime();

	if (maxAllowedFeaturesPerModel < 0) {
		maxAllowedFeaturesPerModel = this.numFeatures;
	}
	var numberOfLambdas = this.DEFAULT_NUMBER_OF_LAMBDAS;
	var maxAllowedFeaturesAlongPath = Math.floor(Math.min(maxAllowedFeaturesPerModel * 1.2, this.numFeatures));

	// lambdaMin = flmin * lambdaMax
	var flmin = (this.numObservations < this.numFeatures ? 5e-2 : 1e-4);

	/********************************
	 * Standardize features and target: Center the target and features
	 * (mean 0) and normalize their vectors to have the same standard
	 * deviation
	 */
	var featureMeans = MathUtil.initArray(this.numFeatures);
	var featureStds = MathUtil.initArray(this.numFeatures);
	var feature2residualCorrelations = MathUtil.initArray(this.numFeatures);

	var factor = 1.0 / Math.sqrt(this.numObservations);
	for (var j = 0; j < this.numFeatures; j++) {
		var mean = MathUtil.getAvg(this.observations[j]);
		featureMeans[j] = mean;
		for (var i = 0; i < this.numObservations; i++) {
			this.observations[j][i] = (factor * (this.observations[j][i] - mean));
		}

		featureStds[j] = Math.sqrt(MathUtil.getDotProduct(this.observations[j], this.observations[j]));

		MathUtil.divideInPlace(this.observations[j], featureStds[j]);
	}

	var targetMean = MathUtil.getAvg(this.targets);
	for (var i = 0; i < this.numObservations; i++) {
		this.targets[i] = factor * (this.targets[i] - targetMean);
	}
	var targetStd = Math.sqrt(MathUtil.getDotProduct(this.targets, this.targets));
	MathUtil.divideInPlace(this.targets, targetStd);

	for (var j = 0; j < this.numFeatures; j++) {
		feature2residualCorrelations[j] = MathUtil.getDotProduct(this.targets, this.observations[j]);
	}

	var feature2featureCorrelations = MathUtil
			.allocateDoubleMatrix(this.numFeatures, maxAllowedFeaturesAlongPath);
	var activeWeights = MathUtil.initArray(this.numFeatures);
	var correlationCacheIndices = MathUtil.initArray(this.numFeatures);
	var denseActiveSet = MathUtil.initArray(this.numFeatures);

	var fit = new LassoFit(numberOfLambdas,
							  maxAllowedFeaturesAlongPath,
							  this.numFeatures);
	fit.numberOfLambdas = 0;

	var alf = Math.pow(Math.max(this.EPSILON, flmin), 1.0 / (numberOfLambdas - 1));
	var rsquared = 0.0;
	fit.numberOfPasses = 0;
	var numberOfInputs = 0;
	var minimumNumberOfLambdas = Math.min(this.MIN_NUMBER_OF_LAMBDAS, numberOfLambdas);

	var curLambda = 0;
	var maxDelta;
	for (var iteration = 1; iteration <= numberOfLambdas; iteration++) {
		console.log("Starting iteration " + iteration + " of Compression.");

		/**********
		 * Compute lambda for this round
		 */
		if (iteration == 1) {
			curLambda = Number.MAX_SAFE_INTEGER;
		} else if (iteration == 2) {
			curLambda = 0.0;
			for (var j = 0; j < this.numFeatures; j++) {
				curLambda = Math.max(curLambda, Math.abs(feature2residualCorrelations[j]));
			}
			curLambda = alf * curLambda;
		} else {
			curLambda = curLambda * alf;
		}

		var prevRsq = rsquared;
		var v;
		while (true) {
			fit.numberOfPasses++;
			maxDelta = 0.0;
			for (var k = 0; k < this.numFeatures; k++) {
				var prevWeight = activeWeights[k];
				var u = feature2residualCorrelations[k] + prevWeight;
				v = (u >= 0 ? u : -u) - curLambda;
				// Computes sign(u)(|u| - curLambda)+
				activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);

				// Is the weight of this variable changed?
				// If not, we go to the next one
				if (activeWeights[k] == prevWeight) {
					continue;
				}

				// If we have not computed the correlations of this
				// variable with other variables, we do this now and
				// cache the result
				if (correlationCacheIndices[k] == 0) {
					numberOfInputs++;
					if (numberOfInputs > maxAllowedFeaturesAlongPath) {
						// we have reached the maximum
						break;
					}
					for (var j = 0; j < this.numFeatures; j++) {
						// if we have already computed correlations for
						// the jth variable, we will reuse it here.
						if (correlationCacheIndices[j] != 0) {
							feature2featureCorrelations[j][numberOfInputs - 1] = feature2featureCorrelations[k][correlationCacheIndices[j] - 1];
						} else {
							// Correlation of variable with itself if one
							if (j == k) {
								feature2featureCorrelations[j][numberOfInputs - 1] = 1.0;
							} else {
								feature2featureCorrelations[j][numberOfInputs - 1] = MathUtil.getDotProduct(
									this.observations[j], this.observations[k]);
							}
						}
					}
					correlationCacheIndices[k] = numberOfInputs;
					fit.indices[numberOfInputs - 1] = k;
				}

				// How much is the weight changed?
				var delta = activeWeights[k] - prevWeight;
				rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
				maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);

				for (var j = 0; j < this.numFeatures; j++) {
					feature2residualCorrelations[j] -= feature2featureCorrelations[j][correlationCacheIndices[k] - 1]
						* delta;
				}
			}

			if (maxDelta < this.CONVERGENCE_THRESHOLD || numberOfInputs > maxAllowedFeaturesAlongPath) {
				break;
			}

			for (var ii = 0; ii < numberOfInputs; ii++) {
				denseActiveSet[ii] = activeWeights[fit.indices[ii]];
			}

			do {
				fit.numberOfPasses++;
				maxDelta = 0.0;
				for (var l = 0; l < numberOfInputs; l++) {
					var k = fit.indices[l];
					var prevWeight = activeWeights[k];
					var u = feature2residualCorrelations[k] + prevWeight;
					v = (u >= 0 ? u : -u) - curLambda;
					activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);
					if (activeWeights[k] == prevWeight) {
						continue;
					}
					var delta = activeWeights[k] - prevWeight;
					rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
					maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);
					for (var j = 0; j < numberOfInputs; j++) {
						feature2residualCorrelations[fit.indices[j]] -= feature2featureCorrelations[fit.indices[j]][correlationCacheIndices[k] - 1]
							* delta;
					}
				}
			} while (maxDelta >= this.CONVERGENCE_THRESHOLD);

			for (var ii = 0; ii < numberOfInputs; ii++) {
				denseActiveSet[ii] = activeWeights[fit.indices[ii]] - denseActiveSet[ii];
			}
			for (var j = 0; j < this.numFeatures; j++) {
				if (correlationCacheIndices[j] == 0) {
					feature2residualCorrelations[j] -= MathUtil.getDotProduct(denseActiveSet,
																			  feature2featureCorrelations[j], numberOfInputs);
				}
			}
		}

		if (numberOfInputs > maxAllowedFeaturesAlongPath) {
			break;
		}
		if (numberOfInputs > 0) {
			for (var ii = 0; ii < numberOfInputs; ii++) {
				fit.compressedWeights[iteration - 1][ii] = activeWeights[fit.indices[ii]];
			}
		}
		fit.numberOfWeights[iteration - 1] = numberOfInputs;
		fit.rsquared[iteration - 1] = rsquared;
		fit.lambdas[iteration - 1] = curLambda;
		fit.numberOfLambdas = iteration;

		if (iteration < minimumNumberOfLambdas) {
			continue;
		}

		var me = 0;
		for (var j = 0; j < numberOfInputs; j++) {
			if (fit.compressedWeights[iteration - 1][j] != 0.0) {
				me++;
			}
		}
		if (me > maxAllowedFeaturesPerModel || ((rsquared - prevRsq) < (this.SMALL * rsquared))
			|| rsquared > this.MAX_RSQUARED) {
			break;
		}
	}

	for (var k = 0; k < fit.numberOfLambdas; k++) {
		fit.lambdas[k] = targetStd * fit.lambdas[k];
		var nk = fit.numberOfWeights[k];
		for (var l = 0; l < nk; l++) {
			fit.compressedWeights[k][l] = targetStd * fit.compressedWeights[k][l] / featureStds[fit.indices[l]];
			if (fit.compressedWeights[k][l] != 0) {
				debugger;
				fit.nonZeroWeights[k]++;
			}
		}
		var product = 0;
		for (var i = 0; i < nk; i++) {
			product += fit.compressedWeights[k][i] * featureMeans[fit.indices[i]];
		}
		fit.intercepts[k] = targetMean - product;
	}

	// First lambda was infinity; fixing it
	fit.lambdas[0] = Math.exp(2 * Math.log(fit.lambdas[1]) - Math.log(fit.lambdas[2]));

	var duration = (new Date()).getTime() - startTime;
	console.log("Elapsed time for compression: " + duration);

	return fit;
};

LassoFitGenerator.prototype.setTargets = function(targets) {
	for (var i = 0; i < this.numObservations; i++) {
		this.targets[i] = Number(targets[i]);
	}
}

LassoFitGenerator.prototype.setTarget = function(idx, target) {
	this.targets[idx] = Number(target);
};

LassoFitGenerator.prototype.fit = function(maxAllowedFeaturesPerModel) {
	var fit = this.getLassoFit(maxAllowedFeaturesPerModel);
	return fit;
};


// --------------------
// TestLasso
// --------------------

function TestLasso() { }

TestLasso.main = function(filename) {
	
	var lines = fs.readFileSync(filename, 'utf8').split('\n');
	
	/*
	 * The first line of the input file is the header which should be ignored.
	 * So, we read the first line
	 */
	var line = lines[0];
	
	/*
	 * Number of features (predictors) is determined based on the 
	 * number of columns in the header line
	 */
	var parts = line.split("\t");
	var featuresCount = parts.length - 1;
	
	/*
	 * Observations and targets are read and loaded from the input file
	 */
	var observations = [];
	var targets = [];

	for (var i=1; i<lines.length; i++) {
		line = lines[i];
		parts = line.split("\t");
		if (line.match(/^\s*$/)) {
			console.warn("Skipping empty line: " + i);
		} else if (parts.length !== (featuresCount+1)) {
			debugger;
			console.warn("Incorrect number of features on line " + i);
		} else {
			var curObservation = [];
			for (var f = 0; f < featuresCount; f++) {
				curObservation[f] = Number(parts[f]);
			}
			observations.push(curObservation);
			targets.push(Number(parts[parts.length - 1]));
		}
	}

	/*
	 * LassoFitGenerator is initialized
	 */
	var fitGenerator = new LassoFitGenerator();
	var numObservations = observations.length;
	fitGenerator.init(featuresCount, numObservations);
	for (i = 0; i < numObservations; i++) {

		fitGenerator.setObservationValues(i, observations[i]);
		fitGenerator.setTarget(i, targets[i]);
	}
	debugger;
	
	/*
	 * Generate the Lasso fit. The -1 arguments means that
	 * there would be no limit on the maximum number of 
	 * features per model
	 */
	var fit = fitGenerator.fit(-1);
	
	/*
	 * Print the generated fit
	 */
	console.log(fit.toString());
};

var fs = require('fs');
var filename = '../data/diabetes.data';
TestLasso.main(filename);
