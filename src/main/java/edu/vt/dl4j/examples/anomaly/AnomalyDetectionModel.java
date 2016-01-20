package edu.vt.dl4j.examples.anomaly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.ModelParameters;
import edu.vt.dl4j.tools.MnistVisualizer;

/**
 * Created by AmrAbed on Jan 20, 2016
 */
public class AnomalyDetectionModel extends Model
{
    private final AnomalyDetectionData data;

    public AnomalyDetectionModel(Data data, ModelParameters parameters)
    {
	super(data, parameters);
	this.data = (AnomalyDetectionData) data;
    }

    @Override
    public Model configure()
    {
	configuration = new NeuralNetConfiguration.Builder().seed(parameters.getSeed()).iterations(parameters.getIterations())
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(parameters.getLearningRate()).l2(0.001)
		.list(4)
		.layer(0,
			new DenseLayer.Builder().nIn(parameters.getInputSize()).nOut(250).weightInit(WeightInit.XAVIER)
				.updater(Updater.ADAGRAD).activation("relu").build())
		.layer(1,
			new DenseLayer.Builder().nIn(250).nOut(10).weightInit(WeightInit.XAVIER)
				.updater(Updater.ADAGRAD).activation("relu").build())
		.layer(2,
			new DenseLayer.Builder().nIn(10).nOut(250).weightInit(WeightInit.XAVIER)
				.updater(Updater.ADAGRAD).activation("relu").build())
		.layer(3,
			new OutputLayer.Builder().nIn(250).nOut(parameters.getInputSize()).weightInit(WeightInit.XAVIER)
				.updater(Updater.ADAGRAD).activation("relu")
				.lossFunction(LossFunctions.LossFunction.MSE).build())
		.pretrain(false).backprop(true).build();

	return this;
    }

    @Override
    public Model train()
    {
	int nEpochs = 30;
	for (int epoch = 0; epoch < nEpochs; epoch++)
	{
	    for (INDArray current : ((AnomalyDetectionData) data).getTrainingFeatures())
	    {
		model.fit(current, current);
	    }
	    System.out.println("Epoch " + epoch + " complete");
	}
	return this;
    }

    @Override
    public Model evaluate()
    {
	final Map<Integer, List<Triple<Double, Integer, INDArray>>> listsByDigit = new HashMap<>();
	for (int i = 0; i < 10; i++)
	{
	    listsByDigit.put(i, new ArrayList<Triple<Double, Integer, INDArray>>());
	}

	int count = 0;
	for (int i = 0; i < data.getTestingFeatures().size(); i++)
	{
	    final INDArray testData = data.getTestingFeatures().get(i);
	    final INDArray labels = data.getTestingLabels().get(i);
	    int nRows = testData.rows();
	    for (int j = 0; j < nRows; j++)
	    {
		INDArray example = testData.getRow(j);
		int label = (int) labels.getDouble(j);
		double score = model.score(new DataSet(example, example));
		listsByDigit.get(label).add(new ImmutableTriple<>(score, count++, example));
	    }
	}

	// Sort data by score, separately for each digit
	Comparator<Triple<Double, Integer, INDArray>> c = new Comparator<Triple<Double, Integer, INDArray>>()
	{
	    @Override
	    public int compare(Triple<Double, Integer, INDArray> o1, Triple<Double, Integer, INDArray> o2)
	    {
		return Double.compare(o1.getLeft(), o2.getLeft());
	    }
	};

	for (List<Triple<Double, Integer, INDArray>> list : listsByDigit.values())
	{
	    Collections.sort(list, c);
	}

	// Select the 5 best and 5 worst numbers (by reconstruction error) for
	// each digit
	List<INDArray> best = new ArrayList<>(50);
	List<INDArray> worst = new ArrayList<>(50);
	for (int i = 0; i < 10; i++)
	{
	    List<Triple<Double, Integer, INDArray>> list = listsByDigit.get(i);
	    for (int j = 0; j < 5; j++)
	    {
		best.add(list.get(j).getRight());
		worst.add(list.get(list.size() - j - 1).getRight());
	    }
	}
	
        //Visualize the best and worst digits
	new MnistVisualizer(2.0,best,"Best (Low Rec. Error)").visualize();
        new MnistVisualizer(2.0,worst,"Worst (High Rec. Error)").visualize();

	return this;
    }

}
