package de.wi1.hohenheim.bachelor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DL4JLibraryEvaluation {
  private static final Logger log = LoggerFactory.getLogger(DL4JLibraryEvaluation.class);

  public static void main(String[] args) throws Exception {
    // one session = one training session with fixed amount of epochs and randomly defined seeds
    // (valid for the complete session).
    int numSessions = 1;
    trainNeuralNetwork("negoisst2016DL4J.csv", ';', 6, 45, 1185, 1, numSessions);
    log.info("==================================================================");
    trainNeuralNetwork("negoisst2017DL4J.csv", ';', 7, 45, 1187, 1, numSessions);
  }

  /**
   * @param file        CSV file input to use for training the model
   * @param split       character that is used to split the data set values inside the CSV
   * @param classNum    classes (e.g. types of messages) in the CSV data set.
   * @param labels      amount of values in each row of the CSV
   * @param batch       data set size: number of lines inside the CSV
   * @param skip        amount of lines that should be skipped in the CSV
   * @param numSessions amount of training sessions with fixed amount of epochs and randomly defined seeds (valid for
   *                    the complete session)
   */
  public static void trainNeuralNetwork(String file, char split, int classNum, int labels, int batch, int skip,
                                        int numSessions)
      throws Exception {
    double accuracy = 0;
    double precision = 0;
    double recall = 0;
    double f1 = 0;

    // for additional (optinal) evaluation over multiple sessions with randomized seeds, etc.
    if (numSessions <= 1) {
      DL4JNeuralNetwork.trainModelFromCSV(file, split, classNum, labels, batch, skip);
    } else {
      // more than one session can be made to evaluate the average performance from multiple sessions.
      for (int i = 0; i < numSessions; i++) {
        org.nd4j.evaluation.classification.Evaluation eval = DL4JNeuralNetwork
            .trainModelFromCSV(file, split, classNum, labels, batch, skip);
        accuracy += eval.accuracy();
        precision += eval.precision();
        recall += eval.recall();
        f1 += eval.f1();
      }

      log.info("==================================================================");
      log.info("Mean Accuracy: " + accuracy / numSessions);
      log.info("Mean Precision: " + precision / numSessions);
      log.info("Mean Recall: " + recall / numSessions);
      log.info("Mean F1: " + f1 / numSessions);
      log.info("==================================================================");
    }
  }

}
