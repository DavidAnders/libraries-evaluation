package de.wi1.hohenheim.bachelor;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DL4JNeuralNetwork {
  private static Logger log = LoggerFactory.getLogger(DL4JNeuralNetwork.class);

  /**
   * This method is used to train a neural network by a given CSV file.
   * Not runnable. This is a helper class.
   *
   * @param file     CSV file input to use for training the model
   * @param split    character that is used to split the data set values inside the CSV
   * @param classNum classes (e.g. types of messages) in the CSV data set.
   * @param labels   amount of values in each row of the CSV
   * @param batch    data set size: number of lines inside the CSV
   * @param skip     amount of lines that should be skipped in the CSV
   */
  public static Evaluation trainModelFromCSV(String file, char split, int classNum, int labels, int batch, int skip)
      throws Exception {
    // amount of cycles for training (improving) the model
    int epochs = 2000;

    // the RecordReaderDataSetIterator handles conversion of the input data
    RecordReader recordReader = new CSVRecordReader(skip, split);
    recordReader.initialize(new FileSplit(new ClassPathResource(file).getFile()));

    // receive loaded data, process all data and prepare for training
    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batch, labels, classNum);
    DataSet allData = iterator.next();

    // data set can be shuffeled for randomization
    // allData.shuffle(1);

    // using 80% of data for training
    SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);
    DataSet trainingData = testAndTrain.getTrain();
    DataSet testData = testAndTrain.getTest();

    // need to normalize the data
    DataNormalization normalizer = new NormalizerStandardize();

    // collect the statistics from the training data
    // This does not modify the input data
    normalizer.fit(trainingData);
    // Apply normalization (no changes to the data)
    normalizer.transform(trainingData);
    normalizer.transform(testData);

    log.info("Building model....");
    //building neural network
    MultiLayerConfiguration configuration
        = new NeuralNetConfiguration.Builder()
        //seed can be kept constant for reproducibility (constant results)
        .seed(1)
        // adapted learning rate caused meaningful improvements to performance scores
        // increasing the learning rate further may lead to unexpected prediction errors
        .updater(new Nesterovs(0.1, 0.9))
        .list()
        .layer(0, new DenseLayer.Builder().nIn(labels).nOut(labels).build())
        .layer(1, new DenseLayer.Builder().nIn(labels).nOut(labels).build())
        .layer(2, new OutputLayer.Builder().nIn(classNum).nOut(classNum).build())
        .build();

    // initialize the model
    MultiLayerNetwork model = new MultiLayerNetwork(configuration);
    model.init();
    model.setListeners(new ScoreIterationListener(100));

    // train the model
    for (int i = 0; i < epochs; i++) {
      model.fit(trainingData);
    }

    // evaluate the model
    Evaluation eval = new Evaluation(classNum);
    INDArray output = model.output(testData.getFeatures());
    eval.eval(testData.getLabels(), output);
    log.info(eval.stats());

    return eval;

    // model can be saved as a file
    // model.save(new File("model.zip"));
  }

}
