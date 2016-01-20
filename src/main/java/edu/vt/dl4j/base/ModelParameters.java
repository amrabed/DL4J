package edu.vt.dl4j.base;

public class ModelParameters
{
    private int seed;
    private int inputSize;
    private int outputSize;
    private int iterations;
    private double learningRate;
    private int listenerFrequency;
    private int hiddeLayerNodes [];

    public int getSeed()
    {
	return seed;
    }

    public ModelParameters setSeed(int seed)
    {
	this.seed = seed;
	return this;
    }

    public int getInputSize()
    {
	return inputSize;
    }

    public ModelParameters setInputSize(int inputSize)
    {
	this.inputSize = inputSize;
	return this;
   }

    public int getOutputSize()
    {
	return outputSize;
    }

    public ModelParameters setOutputSize(int outputSize)
    {
	this.outputSize = outputSize;
	return this;
   }

    public int getIterations()
    {
	return iterations;
    }

    public ModelParameters setIterations(int iterations)
    {
	this.iterations = iterations;
	return this;
   }

    public double getLearningRate()
    {
	return learningRate;
    }

    public ModelParameters setLearningRate(double learningRate)
    {
	this.learningRate = learningRate;
	return this;
   }

    public int getListenerFrequency()
    {
	return listenerFrequency;
    }

    public ModelParameters setListenerFrequency(int listenerFrequency)
    {
	this.listenerFrequency = listenerFrequency;
	return this;
    }

    public int[] getHiddeLayerNodes()
    {
	return hiddeLayerNodes;
    }

    public ModelParameters setHiddeLayerNodes(int hiddeLayerNodes[])
    {
	this.hiddeLayerNodes = hiddeLayerNodes;
	return this;
    }

}
