package edu.vt.dl4j.examples.anomaly;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.data.MnistData;

/**
 * @author AmrAbed
 */
public class AnomalyDetectionRun
{
    private static final int seed = 1234;

    private static final int nSamples = 5000;
    private static final int batchSize = 100;
    private static final int trainingDataSize = 80;
    
    private static final int inputSize = 784;
    private static final int outputSize = 784;
    private static final int iterations = 1;
    private static final double learningRate = 0.05;
    private static final int listenerFrequency = 1;

    public static void main(String[] args)
    {
	final Data data = new MnistData(seed, batchSize, nSamples, false).load().split(trainingDataSize);

	final Parameters parameters = new Parameters().setSeed(seed).setInputSize(inputSize)
		.setOutputSize(outputSize).setIterations(iterations).setLearningRate(learningRate)
		.setListenerFrequency(listenerFrequency);

	final Model model = new AnomalyDetectionModel(parameters, data);
	model.build();
	model.train();
	model.evaluate();
    }
}
