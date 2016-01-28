package edu.vt.dl4j.examples.autoencoder;

import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.data.MnistData;

public class StackedAutoEncoderRun
{
    private static final int seed = 123;

    private static final int nSamples = 60000;
    private static final int batchSize = 100;

    private static final int rows = 28;
    private static final int columns = 28;
    private static final int outputSize = 10;
    private static final int iterations = 10;
    private static final int listenerFrequency = batchSize / 5;

    public static void main(String[] args)
    {
	final MnistData data = new MnistData(seed, batchSize, nSamples, true).load();
	
	final Parameters parameters = new Parameters().setSeed(seed).setInputSize(rows * columns)
		.setOutputSize(outputSize).setIterations(iterations).setListenerFrequency(listenerFrequency);
	
	new StackedAutoEncoderModel(parameters, data).build().train().evaluate();
    }

}
