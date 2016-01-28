package edu.vt.dl4j.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import edu.vt.dl4j.base.Data;

/**
 * MINST data
 * 
 * @author AmrAbed
 */
public class MnistData extends Data
{
    protected final int batchSize, nSamples;
    protected final boolean binarize;

    private final List<DataSet> trainingData = new ArrayList<>();
    private final List<INDArray> trainingFeatures = new ArrayList<>();
    private final List<INDArray> testingFeatures = new ArrayList<>();
    private final List<INDArray> testingLabels = new ArrayList<>();

    public MnistData(int seed, int batchSize, int nSamples, boolean binarize)
    {
	super(seed);
	this.batchSize = batchSize;
	this.nSamples = nSamples;
	this.binarize = binarize;
    }

    @Override
    public MnistData load()
    {
	try
	{
	    iterator = new MnistDataSetIterator(batchSize, nSamples, binarize);
	}
	catch (IOException e)
	{
	    e.printStackTrace();
	}
	return this;
    }

    @Override
    public MnistData split(int trainingDataSize)
    {
	while (iterator.hasNext())
	{
	    final SplitTestAndTrain split = iterator.next().splitTestAndTrain(trainingDataSize, new Random(seed));
	    trainingData.add(split.getTrain());
	    trainingFeatures.add(split.getTrain().getFeatureMatrix());
	    testingFeatures.add(split.getTest().getFeatureMatrix());
	    testingLabels.add(Nd4j.argMax(split.getTest().getLabels(), 1));
	}
	return this;
    }

    public List<DataSet> getTrainingData()
    {
	return trainingData;
    }

    public List<INDArray> getTrainingFeatures()
    {
	return trainingFeatures;
    }

    public List<INDArray> getTestingFeatures()
    {
	return testingFeatures;
    }

    public List<INDArray> getTestingLabels()
    {
	return testingLabels;
    }
}
