package de.wi1.hohenheim.bachelor;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Random;

public class JavaMLClassifcation {
  public static void main(String[] args) throws IOException {
    // Training Data 2016
    String trainingData2016 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2016JavaML_Training.csv";
    // Test Data 2016
    String testData2016 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2016JavaML_Test.csv";
    // Training Data 2017
    String trainingData2017 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2017JavaML_Training.csv";
    // Test Data 2017
    String testData2017 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2017JavaML_Test.csv";

    (new JavaMLClassifcation()).initialize(trainingData2016, testData2016, ";", 45, 6);
    (new JavaMLClassifcation()).initialize(trainingData2017, testData2017, ";", 45, 7);
  }

  public void initialize(String trainingData, String testData, String split, int labels, int classes) throws IOException {
    // create training set from file
    Dataset training = FileHandler.loadDataset(new File(trainingData), labels, split);

    // normalize data (no changes to the data)
    NormalizeMidrange nmr = new NormalizeMidrange();
    nmr.build(training);

    // create a k-nearest neighbour classifier
    Classifier knn = new KNearestNeighbors(2);
    // create a new instance with the classifier
    knn.buildClassifier(training);

    // preform evaluation on the test data set
    Dataset test = FileHandler.loadDataset(new File(testData), labels, split);
    Map<Object, PerformanceMeasure> p = EvaluateDataset.testDataset(knn, test);

    /* Using the data splitting functionality (Training & Testing) of JavaML leads to errors */

    //Dataset[] dataSplit = training.folds(2, new Random());
    //Dataset tData = dataSplit[0];
    //Dataset eData = dataSplit[1];
    //knn.buildClassifier(tData);
    //Map<Object, PerformanceMeasure> p2 = EvaluateDataset.testDataset(knn, eData);

    evaluate(p, classes);
  }

  /*
    Evaluate the classification and calculate the mean values for all relevant metrics.
   */
  public void evaluate(Map<Object, PerformanceMeasure> p, int classes) {
    double accuracy = 0;
    double precision = 0;
    double recall = 0;
    double f1 = 0;

    System.out.println("Accuracy:");
    for (Object o : p.keySet()) {
      System.out.println(o + ": " + p.get(o).getAccuracy());
      accuracy += p.get(o).getAccuracy();
    }
    System.out.println("Precision:");
    for (Object o : p.keySet()) {
      System.out.println(o + ": " + p.get(o).getPrecision());
      precision += p.get(o).getPrecision();
    }
    System.out.println("Recall:");
    for (Object o : p.keySet()) {
      System.out.println(o + ": " + p.get(o).getRecall());
      recall += p.get(o).getRecall();
    }
    System.out.println("F1 Score:");
    for (Object o : p.keySet()) {
      System.out.println(o + ": " + p.get(o).getFMeasure());
      f1 += p.get(o).getFMeasure();
    }

    System.out.println("==================================================================");
    System.out.println("Mean Accuracy: " + accuracy / classes);
    System.out.println("Mean Precision: " + precision / classes);
    System.out.println("Mean Recall: " + recall / classes);
    System.out.println("Mean F1: " + f1 / classes);
    System.out.println("==================================================================");
  }

}
