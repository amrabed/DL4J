package edu.vt.dl4j.examples.anomaly;

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
 * Created by AmrAbed on Jan 20, 2016
 */
public class AnomalyDetectionData extends Data
{
    private final List<INDArray> trainingFeatures = new ArrayList<>();
    private final List<INDArray> testingFeatures = new ArrayList<>();
    private final List<INDArray> testingLabels = new ArrayList<>();

    public AnomalyDetectionData(int batchSize, int nSamples, int trainingDataSize, int seed)
    {
	super(batchSize, nSamples, trainingDataSize, seed);
    }

    @Override
    public Data load()
    {
	try
	{
	    iterator = new MnistDataSetIterator(batchSize, nSamples, false);
	}
	catch (IOException e)
	{
	    e.printStackTrace();
	}
	return this;
    }

    @Override
    public Data split()
    {
	while (iterator.hasNext())
	{
	    final DataSet dataSet = iterator.next();
	    SplitTestAndTrain split = dataSet.splitTestAndTrain(trainingDataSize, new Random(seed));
	    trainingFeatures.add(split.getTrain().getFeatureMatrix());
	    
	    final DataSet testingData = split.getTest();
	    testingFeatures.add(testingData.getFeatureMatrix());
	    
	    final INDArray indexes = Nd4j.argMax(testingData.getLabels(), 1); 
	    testingLabels.add(indexes);
	}

	return this;
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
