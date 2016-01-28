package edu.vt.dl4j.examples.cnn;

import edu.vt.dl4j.base.Parameters;

/**
 * Hyper-parameters associated with CNNs
 * 
 * @author AmrAbed
 *
 */
public class ConvulationalNetParameters extends Parameters
{
    private int channels, rows, columns;

    public int getChannels()
    {
        return channels;
    }

    public ConvulationalNetParameters setChannels(int channels)
    {
        this.channels = channels;
        return this;
    }

    public int getRows()
    {
        return rows;
    }

    public ConvulationalNetParameters setRows(int rows)
    {
        this.rows = rows;
        return this;
    }

    public int getColumns()
    {
        return columns;
    }

    public ConvulationalNetParameters setColumns(int columns)
    {
        this.columns = columns;
        return this;
    }

}
