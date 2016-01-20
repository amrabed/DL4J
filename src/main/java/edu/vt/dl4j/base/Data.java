package edu.vt.dl4j.base;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Created by AmrAbed on Jan 20, 2016
 */
public abstract class Data
{
    protected final int batchSize, nSamples, trainingDataSize, seed;

    protected DataSetIterator iterator;

    public Data(int batchSize, int nSamples, int trainingDataSize, int seed)
    {
	this.batchSize = batchSize;
	this.nSamples = nSamples;
	this.trainingDataSize = trainingDataSize;
	this.seed = seed;
    }

    public abstract Data load();

    public abstract Data split();

    public DataSetIterator getIterator()
    {
	return iterator;
    }

}
