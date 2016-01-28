package edu.vt.dl4j.data;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

import org.apache.commons.io.FileUtils;

import edu.vt.dl4j.base.Data;
import edu.vt.dl4j.tools.CharacterIterator;

/**
 * Using data from Shakespeare for training LSTM model
 * 
 * @author AmrAbed
 */
public class ShakespeareData extends Data
{
    private final int miniBatchSize, examplesPerEpoch, exampleLength;

    /**
     * 
     * @param seed
     *            Seed for Random Number Generator
     * @param miniBatchSize
     *            Size of mini-batch to use when training
     * @param examplesPerEpoch
     *            Number of examples to learn on between generating samples
     * @param exampleLength
     *            Length of each training example
     */
    public ShakespeareData(int seed, int miniBatchSize, int examplesPerEpoch, int exampleLength)
    {
	super(seed);
	this.miniBatchSize = miniBatchSize;
	this.examplesPerEpoch = examplesPerEpoch;
	this.exampleLength = exampleLength;
    }

    @Override
    public Data load()
    {
	try
	{
	    iterator = getShakespeareIterator();
	}
	catch (Exception e)
	{
	    e.printStackTrace();
	}
	return this;
    }

    private CharacterIterator getShakespeareIterator() throws Exception
    {
	// The Complete Works of William Shakespeare
	// 5.3MB file in UTF-8 Encoding, ~5.4 million characters
	// https://www.gutenberg.org/ebooks/100
	final String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
	final String path = System.getProperty("java.io.tmpdir") + "/Shakespeare.txt";
	final File file = new File(path);
	if (!file.exists())
	{
	    FileUtils.copyURLToFile(new URL(url), file);
	    System.out.println("File downloaded to " + file.getAbsolutePath());
	}
	else
	{
	    System.out.println("Using existing text file at " + file.getAbsolutePath());
	}

	if (!file.exists())
	{
	    throw new IOException("File does not exist: " + path);
	}

	// Which characters are allowed? Others will be removed
	final char[] validCharacters = CharacterIterator.getMinimalCharacterSet();
	return new CharacterIterator(path, Charset.forName("UTF-8"), miniBatchSize, exampleLength, examplesPerEpoch,
		validCharacters, new Random(seed), true);
    }
}
