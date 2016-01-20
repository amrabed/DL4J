package edu.vt.dl4j.examples.dbn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.ModelParameters;

/**
 * Created by AmrAbed on Jan 20, 2016
 */
public class DeepBeliefNetworkModel extends Model
{
    public DeepBeliefNetworkModel(Data data, ModelParameters parameters)
    {
	super(data, parameters);
    }

    public DeepBeliefNetworkModel configure()
    {
	final RBM hiddenLayer = new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
		.nIn(parameters.getInputSize()).nOut(3).weightInit(WeightInit.XAVIER).k(1).activation("relu")
		.lossFunction(LossFunctions.LossFunction.RMSE_XENT).updater(Updater.ADAGRAD).dropOut(0.5).build();

	final OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(3)
		.nOut(parameters.getOutputSize()).activation("softmax").build();

	configuration = new NeuralNetConfiguration.Builder().seed(parameters.getSeed())
		.iterations(parameters.getIterations()).learningRate(parameters.getLearningRate())
		.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).l2(2e-4).regularization(true).momentum(0.9)
		.useDropConnect(true).list(2).layer(0, hiddenLayer).layer(1, outputLayer).build();

	return this;
    }

    public DeepBeliefNetworkModel train()
    {
	model.fit(((DeepBeliefNetworkIrisData) data).getTrainingData());

	for (Layer layer : model.getLayers())
	{
	    final INDArray weights = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
	    System.out.println("Weights: " + weights);
	}
	return this;

    }

    @SuppressWarnings("rawtypes")
    public DeepBeliefNetworkModel evaluate()
    {
	final DataSet testingData = ((DeepBeliefNetworkIrisData) data).getTestingData();

	final Evaluation evaluation = new Evaluation(parameters.getOutputSize());
	for (int j = 0; j < 2; j++)
	{
	    final INDArray output = model.output(testingData.getFeatureMatrix(), Layer.TrainingMode.TEST);

	    for (int i = 0; i < output.rows(); i++)
	    {
		String actual = testingData.getLabels().getRow(i).toString().trim();
		String predicted = output.getRow(i).toString().trim();
		System.out.println("actual " + actual + " vs predicted " + predicted);
	    }

	    evaluation.eval(testingData.getLabels(), output);
	    System.out.println(evaluation.stats());
	}
	return this;
    }

    public void print()
    {
	try (final DataOutputStream out = new DataOutputStream(Files.newOutputStream(Paths.get("coefficients.bin"))))
	{
	    Nd4j.write(model.params(), out);
	}
	catch (IOException e)
	{
	    e.printStackTrace();
	}

	final String json = model.getLayerWiseConfigurations().toJson();
	final MultiLayerConfiguration newConfiguration = MultiLayerConfiguration.fromJson(json);
	final MultiLayerNetwork newNetwork = new MultiLayerNetwork(newConfiguration);
	newNetwork.init();
	try (final DataInputStream in = new DataInputStream(new FileInputStream("coefficients.bin"));)
	{
	    newNetwork.setParams(Nd4j.read(in));
	}
	catch (IOException e)
	{
	    e.printStackTrace();
	}

	System.out.println("Original network params: " + model.params());
	System.out.println("New network params: " + newNetwork.params());
    }

}
