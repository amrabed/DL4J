package edu.vt.dl4j.data;

import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import edu.vt.dl4j.base.Data;

public class IrisData extends Data
{
    private final int batchSize, nSamples;

    private DataSet dataset, trainingData, testingData;

    public IrisData(int seed, int batchSize, int nSamples)
    {
	super(seed);
	this.batchSize = batchSize;
	this.nSamples = nSamples;
    }

    @Override
    public IrisData load()
    {
	iterator = new IrisDataSetIterator(batchSize, nSamples);
	dataset = iterator.next();
	dataset.normalizeZeroMeanZeroUnitVariance();
	return this;
    }
    
    public IrisData shuffle()
    {
	Nd4j.shuffle(dataset.getFeatureMatrix(), new Random(seed), 1);
	Nd4j.shuffle(dataset.getLabels(), new Random(seed), 1);
	return this;
    }

    @Override
    public IrisData split(int trainingDataSize)
    {
	final SplitTestAndTrain splits = dataset.splitTestAndTrain(trainingDataSize, new Random(seed));
	trainingData = splits.getTrain();
	testingData = splits.getTest();

	return this;
    }
    
    public IrisData forceNumericalStability()
    {
	Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
	return this;
    }

    public DataSet getTrainingData()
    {
	return trainingData;
    }

    public DataSet getTestingData()
    {
	return testingData;
    }
}
