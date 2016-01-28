package edu.vt.dl4j.base;

/**
 * Hyper-parameters used for Model Configuration
 * 
 * @author AmrAbed
 *
 */
@SuppressWarnings("unchecked")
public class Parameters
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

    public <T extends Parameters> T setSeed(int seed)
    {
	this.seed = seed;
	return (T) this;
    }

    public int getInputSize()
    {
	return inputSize;
    }

    public <T extends Parameters> T setInputSize(int inputSize)
    {
	this.inputSize = inputSize;
	return (T) this;
   }

    public int getOutputSize()
    {
	return outputSize;
    }

    public <T extends Parameters> T setOutputSize(int outputSize)
    {
	this.outputSize = outputSize;
	return (T) this;
   }

    public int getIterations()
    {
	return iterations;
    }

    public <T extends Parameters> T setIterations(int iterations)
    {
	this.iterations = iterations;
	return (T) this;
   }

    public double getLearningRate()
    {
	return learningRate;
    }

    public <T extends Parameters> T setLearningRate(double learningRate)
    {
	this.learningRate = learningRate;
	return (T) this;
   }

    public int getListenerFrequency()
    {
	return listenerFrequency;
    }

    public <T extends Parameters> T setListenerFrequency(int listenerFrequency)
    {
	this.listenerFrequency = listenerFrequency;
	return (T) this;
    }

    public int[] getHiddeLayerNodes()
    {
	return hiddeLayerNodes;
    }

    public <T extends Parameters> T setHiddeLayerNodes(int hiddeLayerNodes[])
    {
	this.hiddeLayerNodes = hiddeLayerNodes;
	return (T) this;
    }
}
