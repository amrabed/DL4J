package edu.vt.dl4j.examples.cnn.iris;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.data.IrisData;
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
	final Data data = new IrisData(seed, batchSize, nSamples).load().shuffle().split(trainingDataSize);
	
	final ConvulationalNetParameters parameters = new ConvulationalNetParameters()
		.setChannels(channels).setRows(rows).setColumns(columns).setSeed(seed).setInputSize(channels)
		.setOutputSize(outputSize).setIterations(iterations).setListenerFrequency(listenerFrequency);

	new ConvolutionalNetModel(parameters, data).build().train().print().evaluate();
    }

}
