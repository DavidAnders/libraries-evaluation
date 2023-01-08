package de.wi1.hohenheim.bachelor;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;


public class NeurophNeuralNetwork implements LearningEventListener {
  private static final Logger log = LoggerFactory.getLogger(NeurophNeuralNetwork.class);

  public static void main(String[] args) {
    // for 2016
    String inputFile2016 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2016Neuroph.csv";
    // for 2017
    String inputFile2017 = "C:\\Users\\test\\git\\Bachelor\\libraries-evaluation\\src\\main\\resources\\negoisst2017Neuroph.csv";

    (new NeurophNeuralNetwork()).initialize(inputFile2016, ";", 6, 45);
    log.info("==================================================================");
    (new NeurophNeuralNetwork()).initialize(inputFile2017, ";", 7, 45);
  }

  public void initialize(String file, String split, int classNum, int labels) {

    // create training set from file
    DataSet inputData = DataSet.createFromFile(file, labels, classNum, split, true);

    // split data in training and test (80% training)
    List<DataSet> subSets = Arrays.asList(inputData.split(0.8, 0.2));
    DataSet trainingData = subSets.get(0);
    DataSet testData = subSets.get(1);

    // normalize data (no changes to the data)
    MaxNormalizer maxNormalizer = new MaxNormalizer(inputData);
    maxNormalizer.normalize(trainingData);
    maxNormalizer.normalize(testData);

    // create MultiLayerPerceptron neural network
    MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(labels, labels, classNum);

    // apply learning rule and add listener for evaluation
    neuralNet.setLearningRule(new BackPropagation());
    BackPropagation learningRule = neuralNet.getLearningRule();
    learningRule.addListener(this);

    // set learning rate and max error
    learningRule.setLearningRate(0.1);
    // amount of cycles for training (improving) the model
    learningRule.setMaxIterations(2000);
    // train the network with the training data
    neuralNet.learn(trainingData);

    // evaluate the network
    evaluate(neuralNet, testData, classNum);
  }

  @Override
  /*
    Monitoring iterations and network error.
   */
  public void handleLearningEvent(LearningEvent event) {
    BackPropagation bp = (BackPropagation) event.getSource();
    if (bp.getCurrentIteration() % 100 == 0) {
      System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }
  }

  /*
    Custom evaluation method for measuring all relevant metrics.
    */
  public void evaluate(NeuralNetwork neuralNet, DataSet dataSet, int classNum) {
    Evaluation evaluation = new Evaluation();
    evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

    /*
      Defining the classes from the input data.
    */
    String[] classLabels;
    if (classNum == 6) {
      classLabels = new String[]{"Offer", "Counteroffer", "Question", "Clarification", "Accept", "Reject"};
    } else {
      classLabels = new String[]{"Offer", "Counteroffer", "Question", "Clarification", "Accept", "Reject", "Request"};
    }

    evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
    evaluation.evaluate(neuralNet, dataSet);

    ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
    ConfusionMatrix confusionMatrix = evaluator.getResult();
    System.out.println(confusionMatrix.toString());
    ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
    ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
    for (ClassificationMetrics cm : metrics) {
      System.out.println(cm.toString());
    }
    System.out.println(average);
  }
}
