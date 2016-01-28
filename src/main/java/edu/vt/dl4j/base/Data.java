package edu.vt.dl4j.base;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Abstract class for Data used by model
 * 
 * @author AmrAbed
 */
public abstract class Data
{
    protected final int seed;
    
    protected DataSetIterator iterator;

    public Data(int seed)
    {
	this.seed = seed;
    }
    
    public abstract Data load();

    public Data split(int trainingDataSize)
    {
	return this;
    }
    
    public DataSetIterator getIterator()
    {
	return iterator;
    }
}
