package edu.vt.dl4j.examples.dae;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.base.Model;
import edu.vt.dl4j.base.Parameters;

/**
 * Example Model for Deep Auto Encoder
 * 
 * @author AmrAbed
 */
public class DeepAutoEncoderModel extends Model
{

    public DeepAutoEncoderModel(Parameters parameters, Data data)
    {
	super(parameters, data);
    }

    @Override
    protected MultiLayerConfiguration getConfiguration()
    {
	final int[] hiddenLayerNodes = parameters.getHiddeLayerNodes();
	final int nLayers = hiddenLayerNodes.length;
	final ListBuilder list = new NeuralNetConfiguration.Builder().seed(parameters.getSeed())
		.iterations(parameters.getIterations()).optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
		.list(nLayers);
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
		final RBM hiddenLayer = new RBM.Builder().nIn(nIn).nOut(hiddenLayerNodes[i])
			.lossFunction(LossFunctions.LossFunction.RMSE_XENT).build();
		list.layer(i, hiddenLayer);
	    }
	    else
	    {
		final OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
			.nIn(nIn).nOut(parameters.getOutputSize()).build();
		list.layer(nLayers - 1, outputLayer);
	    }
	}
	return list.pretrain(true).backprop(true).build();
    }

    @Override
    public Model train()
    {
	final DataSetIterator iterator = data.getIterator();
	while (iterator.hasNext())
	{
	    DataSet next = iterator.next();
	    model.fit(new DataSet(next.getFeatureMatrix(), next.getFeatureMatrix()));
	}
	return this;
    }

    @Override
    public Model evaluate()
    {
	return this;
    }

}
