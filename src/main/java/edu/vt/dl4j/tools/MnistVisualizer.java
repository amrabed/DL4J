package edu.vt.dl4j.tools;

import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MnistVisualizer
{
    private double imageScale;
    private List<INDArray> digits; // Digits (as row vectors), one per INDArray
    private String title;

    public MnistVisualizer(double imageScale, List<INDArray> digits, String title)
    {
	this.imageScale = imageScale;
	this.digits = digits;
	this.title = title;
    }

    public void visualize()
    {
	JFrame frame = new JFrame();
	frame.setTitle(title);
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	JPanel panel = new JPanel();
	panel.setLayout(new GridLayout(0, 5));

	List<JLabel> list = getComponents();
	for (JLabel image : list)
	{
	    panel.add(image);
	}

	frame.add(panel);
	frame.setVisible(true);
	frame.pack();
    }

    private List<JLabel> getComponents()
    {
	List<JLabel> images = new ArrayList<>();
	for (INDArray arr : digits)
	{
	    BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
	    for (int i = 0; i < 768; i++)
	    {
		bi.getRaster().setSample(i % 28, i / 28, 0, (int) (255 * arr.getDouble(i)));
	    }
	    ImageIcon orig = new ImageIcon(bi);
	    Image imageScaled = orig.getImage().getScaledInstance((int) (imageScale * 28), (int) (imageScale * 28),
		    Image.SCALE_REPLICATE);
	    ImageIcon scaled = new ImageIcon(imageScaled);
	    images.add(new JLabel(scaled));
	}
	return images;
    }
}
