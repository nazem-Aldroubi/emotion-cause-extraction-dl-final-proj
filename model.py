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
        super(InterECModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 200
        self.batch_size = 32
        self.rnn_size = 100
        self.num_classes = 1

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, self.embedding_size)
        self.lower_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.attention = tf.keras.layers.Attention()

        self.emotion_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.emotion_dense = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')

        self.cause_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True))
        self.cause_dense = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')

        self.flatten = tf.keras.layers.Flatten()


    def call(self, clauses):
        embedding = self.embedding_layer(clauses)

        lower_biLSTM_out = self.lower_biLSTM(embedding)[0]

        out_with_atten = self.attention([lower_biLSTM_out, lower_biLSTM_out ,lower_biLSTM_out])
        out_with_atten += lower_biLSTM_out

        print(out_with_atten.shape)

        emotion_seq = self.emotion_biLSTM(out_with_atten)[0]
        emotion_seq = self.flatten(emotion_seq)
        print(emotion_seq.shape)
        emotion_probs = self.emotion_dense(emotion_seq)

        cause_seq = self.cause_biLSTM(out_with_atten)[0]
        cause_seq = self.flatten(cause_seq)
        cause_probs = self.cause_dense(cause_seq)

        # print(self.get_labels(emotion_probs))

        return emotion_probs, cause_probs
    
    # get the labels for clauses given probabilities
    # (num_clauses, 1)
    # return numpy.array
    def get_labels(self, probs):
        data = probs.numpy()
        return (data > 1/2)
    
    # get the embeddings of the clauses from the embedding layer
    # this will be used for the logistic regression
    def get_embeddings(self, clauses):
        return self.embedding_layer(clauses)
    
    def loss(self, cause_probabilities, cause_labels, emotion_probabilities, emotion_labels, alpha=1/2):
        cause_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(cause_labels, cause_probabilities))
        emotion_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(emotion_labels, emotion_probabilities))
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
                print("LOSS: ", loss)

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
    
    def __init__(self):
        self.model = LogisticRegressionCV(cv=10, fit_intercept=True, penalty="elasticnet")
    
    def get_cartesian_products(self, embedding_model, emotion_clauses, cause_clauses):
        """
        Given the set of emotion clauses and the set of cause clauses, obtain the embeddings from the embedding model
        Then produce the Cartesian product of both the clauses and their embeddings
        """
        emotion_embeddings = embedding_model.get_embeddings(emotion_clauses).numpy()
        cause_embeddings = embedding_model.get_embeddings(cause_clauses).numpy()
        clause_pairs = np.array([[(e, c) for c in cause_clauses] for e in emotion_clauses]).reshape(-1)
        embedding_pairs = np.array([np.append(e, c) for c in cause_embeddings] for e in emotion_embeddings)
        
        return clause_pairs, embedding_pairs
    
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

class NetworkFilterModel(tf.keras.Model):
    def __init__(self):
        super(NetworkFilterModel, self).__init__()
        self.batch_size = 5
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

    def call(self, inputs):
        return self.dense(inputs)
    
    def loss(self, labels, pred):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, pred))

    def train(self, train_inputs, train_labels):
        current_start = 0
        num_samples = train_inputs.shape[0]
        while current_start + self.batch_size < num_samples:
            current_batch = train_inputs[current_start:current_start + self.batch_size]
            current_labels = train_labels[current_start:current_start + self.batch_size]

            with tf.GradientTape() as t:
                batch_pred = self.call(current_batch)
                batch_loss = self.loss(current_labels, batch_pred)
                print(batch_loss)
            gradients = t.gradient(batch_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            current_start += self.batch_size
    
    def test(self, test_inputs):
        return self.call(test_inputs)

    def get_cartesian_products(self, embedding_model, emotion_clauses, cause_clauses, pair_dict):
        """
        Given the set of emotion clauses and the set of cause clauses, obtain the embeddings from the embedding model
        Then produce the Cartesian product of both the clauses and their embeddings
        """
        emotion_embeddings = embedding_model.get_embeddings(emotion_clauses).numpy()
        emotion_embeddings = emotion_embeddings.reshape((emotion_embeddings.shape[0], -1))
        cause_embeddings = embedding_model.get_embeddings(cause_clauses).numpy()
        cause_embeddings = cause_embeddings.reshape((cause_embeddings.shape[0], -1))
        if cause_embeddings.shape[0] == 0 or emotion_embeddings.shape[0] == 0:
            return np.array([]), np.array([])
        embedding_pairs = np.array([[np.append(e, c) for c in cause_embeddings] for e in emotion_embeddings])
        label_pairs = np.array([[([e,c] in pair_dict) for c in cause_clauses] for e in emotion_clauses])
        embedding_pairs = embedding_pairs.reshape((-1, embedding_pairs.shape[-1]))
        label_pairs = label_pairs.reshape((-1))

        return embedding_pairs, label_pairs

# return F1, recall, precision scores
# inputs: labels, pred are 1d np arrays
def scores(labels, pred):
    true_positive = sum(np.logical_and((labels==1), (pred==1)))
    true_negative = sum(np.logical_and((labels==0), (pred==0)))
    false_positive = sum(np.logical_and((labels==0), (pred==1)))
    false_negative = sum(np.logical_and((labels==1), (pred==0)))

    # precision represents the correct percentage of all positive predicitions
    precision = true_positive / (true_positive + false_positive)
    # recall represents how well we can predict results that should be positive
    recall = true_positive / (true_positive + false_negative)

    # f1 is the harmonic mean of precision and recall
    # intuitively, it represents the quality of how well we should trust a prediction that's positive
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
    
def main():
    train_clauses, test_clauses, train_emotion_labels, test_emotion_labels, \
    train_cause_labels, test_cause_labels, train_emotion_cause_pairs, \
    test_emotion_cause_pairs, word2id, pad_index = get_data("data.txt")

    ec_extract_model = InterECModel(len(word2id))
    # TODO: Train (and test) InterECModel.
    ec_extract_model.train(train_clauses, train_cause_labels, train_emotion_labels)

    # TODO: Obtain emotion clauses and cause clauses extracted from the InterECModel.

    # TODO: Apply Cartesian product to set of emotion clauses and set of cause clauses to obtain all possible pairs.

    # TODO: Instantiate PairFilterModel.

    # TODO: Train (and test) PairFilterModel.
    
    """
    # Train the EC Model
    ec_extract_model = InterECModel(len(word2id))
    ec_extract_model.train(train_clauses, train_cause_labels, train_emotion_labels)
    emotion_prob, cause_prob = ec_extract_model.call(train_clauses)

    # Extract clauses
    emotion_clauses = ec_extract_model.get_clauses(train_clauses, emotion_prob)
    cause_clauses = ec_extract_model.get_clauses(train_clauses, cause_prob)
    
    print(len(train_clauses))
    print(len(emotion_clauses))
    print(len(cause_clauses))

    # Create filter model
    pair_filter_model = NetworkFilterModel()
    # Create labels for pairs
    embedding_pairs, label_pairs = pair_filter_model.get_cartesian_products(ec_extract_model, emotion_clauses, cause_clauses, train_emotion_cause_pairs)
    # Train the logistic model
    pair_filter_model.train(embedding_pairs, label_pairs)
    """
if __name__ == '__main__':
    main()
