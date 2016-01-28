package edu.vt.dl4j.examples.cnn.mnist;

import edu.vt.dl4j.data.MnistData;
import edu.vt.dl4j.examples.cnn.ConvulationalNetParameters;

public class ConvolutionalNetRun
{
    private static final int seed = 123;

    private static final int batchSize = 150;
    private static final int nSamples = 150;
    private static final int trainingDataSize = 100;

    private static final int channels = 1;
    private static final int outputSize = 3;
    private static final int iterations = 10;
    private static final int listenerFrequency = 1;
    private static final int rows = 2;
    private static final int columns = 2;

    public static void main(String[] args)
    {
	final MnistData data = new MnistData(seed, batchSize, nSamples, true).load().split(trainingDataSize);
	
	final ConvulationalNetParameters parameters = new ConvulationalNetParameters()
		.setChannels(channels).setRows(rows).setColumns(columns).setSeed(seed).setInputSize(channels)
		.setOutputSize(outputSize).setIterations(iterations).setListenerFrequency(listenerFrequency);

	new ConvolutionalNetModel(parameters, data).build().train().print().evaluate();
    }

}
