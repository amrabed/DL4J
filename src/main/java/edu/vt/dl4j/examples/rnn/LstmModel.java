package edu.vt.dl4j.examples.rnn;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.tools.CharacterIterator;

/**
 * Example LSTM Model
 * 
 * @author AmrAbed
 *
 */
public class LstmModel extends Model
{
    public LstmModel(Parameters parameters, Data data)
    {
	super(parameters, data);
    }

    @Override
    protected MultiLayerConfiguration getConfiguration()
    {
	final int[] hiddenLayerNodes = parameters.getHiddeLayerNodes();
	final int nLayers = hiddenLayerNodes.length + 1;

	final ListBuilder list = new NeuralNetConfiguration.Builder()
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.iterations(parameters.getIterations()).learningRate(parameters.getLearningRate()).rmsDecay(0.95)
		.seed(parameters.getSeed()).regularization(true).l2(0.001).list(nLayers).pretrain(false).backprop(true);

	for (int i = 0; i < nLayers; i++)
	{
	    int nIn;
	    if (i == 0)
	    {
		nIn = parameters.getInputSize();
	    }
	    else
	    {
		nIn = hiddenLayerNodes[i - 1];
	    }
	    if (i < nLayers - 1)
	    {
		final GravesLSTM layer = new GravesLSTM.Builder().nIn(nIn).nOut(hiddenLayerNodes[i])
			.updater(Updater.RMSPROP).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
			.dist(new UniformDistribution(-0.08, 0.08)).build();
		list.layer(i, layer);
	    }
	    else
	    {
		final RnnOutputLayer outputLayer = new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")
			.updater(Updater.RMSPROP).nIn(hiddenLayerNodes[1]).nOut(parameters.getOutputSize())
			.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.08, 0.08)).build();
		list.layer(i, outputLayer);
	    }
	}
	return list.build();
    }

    @Override
    public Model train()
    {
	try
	{
	    final int numEpochs = 30;
	    final Sampler sampler = new Sampler();
	    final CharacterIterator iterator = (CharacterIterator) data.getIterator();
	    for (int i = 0; i < numEpochs; i++)
	    {
		model.fit(iterator);

		System.out.println("--------------------");
		System.out.println("Completed epoch " + i);
		final String[] samples = sampler.sampleCharactersFromNetwork(iterator);
		for (int j = 0; j < samples.length; j++)
		{
		    System.out.println("----- Sample " + j + " -----");
		    System.out.println(samples[j]);
		    System.out.println();
		}
		// Reset iterator for next epoch
		iterator.reset(); 
	    }
	    System.out.println("\n----- Training complete -----");
	}
	catch (Exception e)
	{
	    e.printStackTrace();
	}
	return this;
    }

    private class Sampler
    {
	private final Random random = new Random(parameters.getSeed());
	// Optional character initialization; a random character is used if null
	private String initialization = null;
	// Number of samples to generate after each training epoch
	private final int numSamples = 4;
	// Length of each sample to generate
	private final int charactersToSample = 300;

	/**
	 * Generate a sample from the network, given an (optional, possibly
	 * null) initialization. Initialization can be used to 'prime' the RNN
	 * with a sequence you want to extend/continue.<br>
	 * Note that the initialization is used for all samples
	 * 
	 * @param initialization
	 *            String, may be null. If null, select a random character as
	 *            initialization for all samples
	 * @param charactersToSample
	 *            Number of characters to sample from network (excluding
	 *            initialization)
	 * @param net
	 *            MultiLayerNetwork with one or more GravesLSTM/RNN layers
	 *            and a softmax output layer
	 * @param iterator
	 *            CharacterIterator. Used for going from indexes back to
	 *            characters
	 */
	private String[] sampleCharactersFromNetwork(CharacterIterator iterator)
	{
	    // Set up initialization. If no initialization: use a random
	    // character
	    if (initialization == null)
	    {
		initialization = String.valueOf(iterator.getRandomCharacter());
	    }

	    // Create input for initialization
	    INDArray initializationInput = Nd4j.zeros(numSamples, iterator.inputColumns(), initialization.length());
	    char[] init = initialization.toCharArray();
	    for (int i = 0; i < init.length; i++)
	    {
		int idx = iterator.convertCharacterToIndex(init[i]);
		for (int j = 0; j < numSamples; j++)
		{
		    initializationInput.putScalar(new int[] { j, idx, i }, 1.0f);
		}
	    }

	    final StringBuilder[] string = new StringBuilder[numSamples];
	    for (int i = 0; i < numSamples; i++)
	    {
		string[i] = new StringBuilder(initialization);
	    }

	    // Sample from network (and feed samples back into input) one
	    // character
	    // at a time (for all samples)
	    // Sampling is done in parallel here
	    model.rnnClearPreviousState();
	    INDArray output = model.rnnTimeStep(initializationInput);
	    // Gets the last time step output
	    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);

	    for (int i = 0; i < charactersToSample; i++)
	    {
		// Set up next input (single time step) by sampling from
		// previous
		// output
		final INDArray nextInput = Nd4j.zeros(numSamples, iterator.inputColumns());
		// Output is a probability distribution. Sample from this for
		// each
		// example we want to generate, and add it to the new input
		for (int s = 0; s < numSamples; s++)
		{
		    final double[] outputProbDistribution = new double[iterator.totalOutcomes()];
		    for (int j = 0; j < outputProbDistribution.length; j++)
		    {
			outputProbDistribution[j] = output.getDouble(s, j);
		    }
		    int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);

		    // Prepare next time step input
		    nextInput.putScalar(new int[] { s, sampledCharacterIdx }, 1.0f);
		    // Add sampled character to StringBuilder
		    // (human readable output)
		    string[s].append(iterator.convertIndexToCharacter(sampledCharacterIdx));
		}

		// Do one time step of forward pass
		output = model.rnnTimeStep(nextInput);
	    }

	    String[] out = new String[numSamples];
	    for (int i = 0; i < numSamples; i++)
		out[i] = string[i].toString();
	    return out;
	}

	/**
	 * Given a probability distribution over discrete classes, sample from
	 * the distribution and return the generated class index.
	 * 
	 * @param distribution
	 *            Probability distribution over classes. Must sum to 1.0
	 */
	private int sampleFromDistribution(double[] distribution)
	{
	    final double d = random.nextDouble();
	    double sum = 0.0;
	    for (int i = 0; i < distribution.length; i++)
	    {
		sum += distribution[i];
		if (d <= sum)
		    return i;
	    }
	    // Should never happen if distribution is a valid probability
	    // distribution
	    throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
	}
    }
}
