package edu.vt.dl4j.examples.autoencoder;

import java.io.IOException;
import java.util.Collections;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;

/**
 * Example Stacked-AutoEncoder Model
 * 
 * @author AmrAbed
 *
 */
public class StackedAutoEncoderModel extends Model
{
    public StackedAutoEncoderModel(Parameters parameters, Data data)
    {
	super(parameters, data);
    }

    @Override
    protected MultiLayerConfiguration getConfiguration()
    {
	return new NeuralNetConfiguration.Builder().seed(parameters.getSeed())
		.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
		.gradientNormalizationThreshold(1.0).iterations(parameters.getIterations()).momentum(0.5)
		.momentumAfter(Collections.singletonMap(3, 0.9))
		.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list(4)
		.layer(0,
			new AutoEncoder.Builder().nIn(parameters.getInputSize()).nOut(500).weightInit(WeightInit.XAVIER)
				.lossFunction(LossFunction.RMSE_XENT).corruptionLevel(0.3).build())
		.layer(1, new AutoEncoder.Builder().nIn(500).nOut(250).weightInit(WeightInit.XAVIER)
			.lossFunction(LossFunction.RMSE_XENT).corruptionLevel(0.3)

			.build())
		.layer(2,
			new AutoEncoder.Builder().nIn(250).nOut(200).weightInit(WeightInit.XAVIER)
				.lossFunction(LossFunction.RMSE_XENT).corruptionLevel(0.3).build())
		.layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(200)
			.nOut(parameters.getOutputSize()).build())
		.pretrain(true).backprop(false).build();
    }

    @Override
    public Model train()
    {
	model.fit(data.getIterator());
	return this;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public Model evaluate()
    {
	final Evaluation evaluation = new Evaluation(parameters.getOutputSize());
	try
	{
	    final DataSetIterator iterator = new MnistDataSetIterator(100, 10000);
	    while (iterator.hasNext())
	    {
		final DataSet testingData = iterator.next();
		evaluation.eval(testingData.getLabels(), model.output(testingData.getFeatureMatrix()));
	    }

	    System.out.println(evaluation.stats());
	}
	catch (IOException e)
	{
	    e.printStackTrace();
	}
	return this;
    }

}
