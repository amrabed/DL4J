package edu.vt.dl4j.examples.cnn.iris;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.data.IrisData;
import edu.vt.dl4j.examples.cnn.ConvulationalNetParameters;

public class ConvolutionalNetModel extends Model
{

    public ConvolutionalNetModel(Parameters parameters, Data data)
    {
	super(parameters, data);
    }

    @Override
    protected MultiLayerConfiguration getConfiguration()
    {
	final ConvulationalNetParameters parameters = (ConvulationalNetParameters) this.parameters;
	final MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(parameters.getSeed())
		.iterations(parameters.getIterations())
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list(2)
		.layer(0,
			new ConvolutionLayer.Builder(new int[] { 1, 1 }).nIn(parameters.getInputSize()).nOut(1000)
				.activation("relu").weightInit(WeightInit.RELU).build())
		.layer(1,
			new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nOut(parameters.getOutputSize())
				.weightInit(WeightInit.XAVIER).activation("softmax").build())
		.backprop(true).pretrain(false);

	new ConvolutionLayerSetup(builder, parameters.getRows(), parameters.getColumns(), parameters.getChannels());

	return builder.build();
    }

    @Override
    public Model train()
    {
	model.fit(((IrisData) data).getTrainingData());
	return this;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public Model evaluate()
    {
	final DataSet testingData = ((IrisData) data).getTestingData();
	final Evaluation evaluation = new Evaluation(parameters.getOutputSize());
	evaluation.eval(testingData.getLabels(), model.output(testingData.getFeatureMatrix()));
	System.out.println(evaluation.stats());
	return this;
    }

    @Override
    public Model print()
    {
	for (org.deeplearning4j.nn.api.Layer layer : model.getLayers())
	{
	    System.out.println("Weights: " + layer.getParam(DefaultParamInitializer.WEIGHT_KEY));
	}
	return this;
    }

}
