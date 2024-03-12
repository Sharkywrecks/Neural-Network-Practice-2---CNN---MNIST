import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        /*final long SEED = 123;
        System.out.println("Starting data loading...");
        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
        System.out.println("Images Train size: "+imagesTrain.size());
        System.out.println("Images Test size: "+imagesTest.size());
        NetworkBuilder networkBuilder = new NetworkBuilder(28,28,256*100);
        networkBuilder.addConvolutionLayer(8,5,1,0.1,SEED);
        networkBuilder.addMaxPoolLayer(3,2);
        networkBuilder.addFullConnectLayer(10,0.1,SEED);

        NeuralNetwork neuralNetwork = networkBuilder.build();

        float rate = neuralNetwork.test(imagesTest);
        System.out.println("Pre-training success rate: "+rate);

        for(int epoch=0;epoch<3;epoch++){
            Collections.shuffle(imagesTrain);
            neuralNetwork.train(imagesTrain);
            rate = neuralNetwork.test(imagesTest);
            System.out.println("Current iteration: "+epoch);
            System.out.println("Success rate: "+rate);
        }
        */

    }
}