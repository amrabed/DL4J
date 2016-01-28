package edu.vt.dl4j.base;

import java.util.Arrays;
import java.util.Map;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import com.google.common.annotations.Beta;

/**
 * Abstract class for Deep Learning Models
 * 
 * @author AmrAbed
 */
public abstract class Model
{
    protected final Parameters parameters;
    protected final Data data;

    protected MultiLayerNetwork model;

    public Model(Parameters parameters, Data data)
    {
	this.parameters = parameters;
	this.data = data;
    }

    protected abstract MultiLayerConfiguration getConfiguration();

    public Model build()
    {
	model = new MultiLayerNetwork(getConfiguration());
	model.init();
	model.setListeners(
		Arrays.asList((IterationListener) new ScoreIterationListener(parameters.getListenerFrequency())));
	return this;
    }

    public abstract Model train();

    /**
     * Default implementation does nothing. If needed, override in subclass to
     * evaluate model
     * 
     * @return this
     */
    public Model evaluate()
    {
	return this;
    }
    
    /**
     * Default implementation does nothing. If needed, override in subclass to
     * print model parameters
     * 
     * @return this
     */
    public Model print()
    {
	return this;
    }

    @Beta
    protected class LayerBuilder extends ListBuilder
    {
	protected LayerBuilder(Map<Integer, Builder> layerMap)
	{
	    super(layerMap);
	}

	@SuppressWarnings("rawtypes")
	protected ListBuilder addLayers(FeedForwardLayer.Builder hiddelLayerBuilder,
		FeedForwardLayer.Builder outputLayerBuilder)
	{
	    final int[] hiddenLayerNodes = parameters.getHiddeLayerNodes();
	    final int nLayers = hiddenLayerNodes.length + 1;
	    for (int i = 0; i < nLayers; i++)
	    {
		int nIn;
		if (i == 0)
		{
		    nIn = parameters.getInputSize();
		}
		else
		{
		    nIn = hiddenLayerNodes[i - 1];
		}
		if (i < nLayers - 1)
		{
		    layer(i, hiddelLayerBuilder.nIn(nIn).nOut(hiddenLayerNodes[i]).build());
		}
		else
		{
		    layer(i, outputLayerBuilder.nIn(nIn).nOut(parameters.getOutputSize()).build());
		}
	    }
	    return this;
	}
    }
}
