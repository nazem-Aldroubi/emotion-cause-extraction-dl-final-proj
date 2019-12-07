import tensorflow as tf
import numpy as np
from preprocess import get_data
from sklearn.linear_model import LogisticRegressionCV

class InterECModel(tf.keras.Model):
    """
    This model uses the Inter-EC Model architecture described
    in the paper to classify input clauses as cause clauses,
    emotion clauses, or neither.
    """
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 200
        self.batch_size = 32
        self.rnn_size = 100
        self.num_classes = 2

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
        self.lower_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.attention = tf.keras.layers.Attention()

        self.emotion_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.emotion_dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')

        self.cause_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.cause_dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    def call(self, clauses):
        embedding = tf.nn.embedding_lookup(self.E, clauses)
        # TODO: Finish writing model architecture!
        lower = self.attention(self.lower_biLSTM(embedding))

        emotion_seq, _, _  = self.emotion_biLSTM(lower)
        emotion_probs = self.emotion_dense(emotion_seq)

        cause_seq, _, _ = self.cause_biLSTM(lower)
        cause_probs = self.cause_dense(cause_seq)

        return emotion_probs, cause_probs

    def loss(self, cause_probabilities, cause_labels, emotion_probabilities, emotion_labels, alpha):
        cause_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(cause_labels, cause_probabilities))
        emotion_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(emotion_labels, emotion_probabilities))
        return alpha*emotion_loss + (1 - alpha)*cause_loss

    def train(self, train_clauses, train_cause_labels, train_emotion_labels):
        num_examples = train_clauses.shape[0]

        start_index = 0
        end_index = start_index + self.batch_size

        while (end_index <= num_examples):
            batch_clauses = train_clauses[start_index:end_index]
            batch_cause_labels = train_cause_labels[start_index:end_index]
            batch_emotion_labels = train_emotion_labels[start_index:end_index]

            with tf.GradientTape() as tape:
                cause_probabilities, emotion_probabilities = self.call(batch_clauses)
                loss = self.loss(cause_probabilities, batch_cause_labels, emotion_probabilities, batch_emotion_labels, 0.5)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            start_index = end_index
            end_index = start_index + self.batch_size

    def test(self, test_clauses, test_cause_labels, test_emotion_labels):
        num_examples = test_clauses.shape[0]

        start_index = 0
        end_index = start_index + self.batch_size

        loss = []
        while (end_index <= num_examples):
            batch_clauses = test_clauses[start_index:end_index]
            batch_cause_labels = test_cause_labels[start_index:end_index]
            batch_emotion_labels = test_emotion_labels[start_index:end_index]

            cause_probabilities, emotion_probabilities = self.call(batch_clauses)
            loss.append(self.loss(cause_probabilities, batch_cause_labels, emotion_probabilities, batch_emotion_labels, 0.5))

            start_index = end_index
            end_index = start_index + self.batch_size

        # TODO: This gives us loss for the test batches, but maybe we want to define an accuracy function for our extracted cause/emotion clauses?


class PairFilterModel():
    """
    This model is a Logistic Regression model that filters emotion-cause clause pairs
    to determine valid pairs in which the cause corresponds to the emotion. The emotion-cause pairs
    are obtained by taking the Cartesian product of the set of emotion clauses and the set
    of cause clauses that were extracted from the Inter-EC Model.
    """
    # TODO: Write this model! We can move this to another file, but I've defined it here for now!
    def __init__(self):
        self.model = LogisticRegressionCV(cv=10, fit_intercept=True, penalty="elasticnet")

    def fit(self, train_X, train_Y):
        """
        Fit the logistic model

        :param train_X: the training features, of the form (batch_size, num_features), with features being (s_e, s_c, v)
        :param train_Y: the training labels, 0 or 1 depending on whether the pair is an emotion-cause pair
        :return: null
        """
        train_X = train_X.numpy()
        train_Y = train_Y.numpy()
        self.model.fit(train_X, train_Y)


    def predict(self, test_X):
        """
        Predict the class for inputs

        :param test_X: the test features, of the form (batch_size, num_features)
        :return: the prediction whether the pair is significant (0 or 1)
        """
        test_X = test_X.numpy()
        return self.model.predict(test_X)

    def predict_proba(self, test_X):
        test_X = test_X.numpy()
        return self.model.predict_proba(test_X)

def main():
    train_clauses, test_clauses, train_emotion_labels, test_emotion_labels, \
    train_cause_labels, test_cause_labels, train_emotion_cause_pairs, \
    test_emotion_cause_pairs, word2id, pad_index = get_data("data.txt")

    ec_extract_model = InterECModel(len(word2id))
    # TODO: Train (and test) InterECModel.
    emotion_l, cause_l = ec_extract_model.call(train_clauses)


    # TODO: Obtain emotion clauses and cause clauses extracted from the InterECModel.

    # TODO: Apply Cartesian product to set of emotion clauses and set of cause clauses to obtain all possible pairs.

    # TODO: Instantiate PairFilterModel.

    # TODO: Train (and test) PairFilterModel.

if __name__ == '__main__':
    main()
