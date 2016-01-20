package edu.vt.dl4j.examples.dbn;

import org.nd4j.linalg.factory.Nd4j;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.ModelParameters;

/**
 * Created by AmrAbed on Jan 11, 2016
 */
public class DeepBeliefNetworkRun
{
    private static final int nSamples = 150;
    private static final int batchSize = 150;
    private static final int trainingDataSize = (int) (batchSize * .8);
    private static final int seed = 123;
    private static final int inputSize = 4;
    private static final int outputSize = 3;
    private static final int iterations = 5;
    private static final double learningRate = 1e-6f;
    private static final int listenerFrequency = 1;

    public static void main(String[] args)
    {
	Nd4j.MAX_SLICES_TO_PRINT = -1;
	Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

	final Data data = new DeepBeliefNetworkIrisData(batchSize, nSamples, trainingDataSize, seed).load().split();

	final ModelParameters parameters = new ModelParameters().setSeed(seed).setInputSize(inputSize)
		.setOutputSize(outputSize).setIterations(iterations).setLearningRate(learningRate)
		.setListenerFrequency(listenerFrequency);

	final Model model = new DeepBeliefNetworkModel(data, parameters);
	model.configure();
	model.build();
	model.train();
	model.evaluate();
    }
}
