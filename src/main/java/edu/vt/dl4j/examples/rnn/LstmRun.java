package edu.vt.dl4j.examples.rnn;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.data.ShakespeareData;
import edu.vt.dl4j.tools.CharacterIterator;

/**
 * @author AmrAbed
 */
public class LstmRun
{
    private final static int seed = 12345;

    private final static int miniBatchSize = 32;
    private final static int examplesPerEpoch = 50 * miniBatchSize;
    private final static int exampleLength = 100;

    private final static int iterations = 1;
    private final static double learningRate = 0.1;
    private final static int[] hiddenLayerNodes = new int[] { 200, 200 };

    public static void main(String[] args)
    {
	final Data data = new ShakespeareData(seed, miniBatchSize, examplesPerEpoch, exampleLength).load();
	final CharacterIterator iterator = (CharacterIterator) data.getIterator();
	
	final Parameters parameters = new Parameters().setSeed(seed).setIterations(iterations)
		.setInputSize(iterator.inputColumns()).setOutputSize(iterator.totalOutcomes())
		.setLearningRate(learningRate).setHiddeLayerNodes(hiddenLayerNodes).setListenerFrequency(1);

	new LstmModel(parameters, data).build().train();
    }

}
