package network;

import data.Image;
import data.MatrixUtility;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Layer> layers;
    private final double scaleFactor;
    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }
    private void linkLayers(){
        if(layers.size()<=1)return;

        for(int i=0;i<layers.size();i++){

            if(i==0){
                layers.get(i).setNextLayer(layers.get(i+1));
            }else if(i == layers.size()-1){
                layers.get(i).setPreviousLayer(layers.get(i-1));
            }else{
                layers.get(i).setPreviousLayer(layers.get(i-1));
                layers.get(i).setNextLayer(layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput,int correctAnswer){
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer]=1;

        return MatrixUtility.add(networkOutput,MatrixUtility.multiply(expected,-1));
    }

    private int getMaxIndex(double[] in){

        double max=0;
        int index=0;

        for(int i=0;i<in.length;i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }
        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(MatrixUtility.multiply(image.getData(),1.0/scaleFactor));

        double[] out = layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test(List<Image> images){
        int correct = 0;

        for(Image image:images){
            int guess = guess(image);

            if(guess == image.getLabel()){
                correct++;
            }
        }
        return ((float)correct)/images.size();
    }

    public void train(List<Image> images){

        for(Image image:images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(MatrixUtility.multiply(image.getData(),1.0/scaleFactor));

            double[] out = layers.get(0).getOutput(inList);
            double[] dLdO = getErrors(out,image.getLabel());

            layers.get(layers.size()-1).backward(dLdO);
        }
    }
}
