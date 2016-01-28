package edu.vt.dl4j.examples.cnn.mnist;

import java.util.List;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;
import edu.vt.dl4j.data.MnistData;
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
		.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list(3)
		.layer(0,
			new ConvolutionLayer.Builder(10, 10).stride(2, 2).nIn(parameters.getChannels()).nOut(6)
				.weightInit(WeightInit.XAVIER).activation("relu").build())
		.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build())
		.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			.nOut(parameters.getOutputSize()).weightInit(WeightInit.XAVIER).activation("softmax").build())
		.backprop(true).pretrain(false);

	new ConvolutionLayerSetup(builder, parameters.getRows(), parameters.getColumns(), parameters.getChannels());

	return builder.build();
    }

    @Override
    public Model train()
    {
	for (DataSet trainingData : ((MnistData) data).getTrainingData())
	{
	    model.fit(trainingData);
	}
	return this;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public Model evaluate()
    {
	final List<INDArray> testingFeatures = ((MnistData) data).getTestingFeatures();
	final List<INDArray> testingLabels = ((MnistData) data).getTestingLabels();
	final Evaluation evaluation = new Evaluation(parameters.getOutputSize());
	for (int i = 0; i < testingFeatures.size(); i++)
	{
	    evaluation.eval(testingLabels.get(i), model.output(testingFeatures.get(i)));
	}
	// evaluation.eval(testingLabels.get(0),
	// model.output(testingFeatures.get(0)));
	System.out.println(evaluation.stats());
	return this;
    }
}
