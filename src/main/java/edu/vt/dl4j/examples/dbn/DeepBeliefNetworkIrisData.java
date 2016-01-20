package edu.vt.dl4j.examples.dbn;

import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import edu.vt.dl4j.base.Data;

/**
 * Created by AmrAbed on Jan 20, 2016
 */
public class DeepBeliefNetworkIrisData extends Data
{
    protected DataSet dataSet, trainingData, testingData;

    public DeepBeliefNetworkIrisData(int batchSize, int nSamples, int trainingDataSize, int seed)
    {
	super(batchSize, nSamples, trainingDataSize, seed);
    }

    public DeepBeliefNetworkIrisData load()
    {
	iterator = new IrisDataSetIterator(batchSize, nSamples);
	dataSet = iterator.next();
	dataSet.normalizeZeroMeanZeroUnitVariance();
	return this;
    }

    public DeepBeliefNetworkIrisData split()
    {
	final SplitTestAndTrain splits = dataSet.splitTestAndTrain(trainingDataSize, new Random(seed));
	trainingData = splits.getTrain();
	testingData = splits.getTest();
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
