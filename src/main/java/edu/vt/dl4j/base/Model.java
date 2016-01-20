package edu.vt.dl4j.base;

import java.util.Arrays;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

/**
 * Created by AmrAbed on Jan 20, 2016
 */
public abstract class Model
{
    protected final Data data;
    protected final ModelParameters parameters;

    protected MultiLayerConfiguration configuration;
    protected MultiLayerNetwork model;

    public Model(Data data, ModelParameters parameters)
    {
	this.data = data;
	this.parameters = parameters;
    }

    public abstract Model configure();

    public Model build()
    {
	model = new MultiLayerNetwork(configuration);
	model.init();
	model.setListeners(
		Arrays.asList((IterationListener) new ScoreIterationListener(parameters.getListenerFrequency())));
	return this;
    }

    public abstract Model train();

    public abstract Model evaluate();
}
