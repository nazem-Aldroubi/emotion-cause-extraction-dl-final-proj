import tensorflow as tf
import numpy as np
from preprocess import get_data
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test',
                    help='Can be "train" or "test"')
args = parser.parse_args()

class ECModel(tf.keras.Model):
    """
    This model classifies input clauses as cause clauses,
    emotion clauses, or neither.
    """
    def __init__(self, vocab_size, clause_size):
        super(ECModel, self).__init__()
        self.vocab_size = vocab_size
        self.clause_size = clause_size
        self.embedding_size = 200
        self.batch_size = 1
        self.rnn_size = 100
        self.num_classes = 1
        self.hidden_layer_size = 100

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        self.lower_model = tf.keras.Sequential()
        self.lower_model.add(tf.keras.layers.Embedding(vocab_size, self.embedding_size, input_length=clause_size))

        self.attention = tf.keras.layers.Attention()

        self.emotion_model = tf.keras.Sequential()
        self.emotion_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)))
        self.emotion_model.add(tf.keras.layers.Flatten())
        self.emotion_model.add(tf.keras.layers.Dense(self.hidden_layer_size, activation='relu'))
        self.emotion_model.add(tf.keras.layers.Dense(self.num_classes, activation='sigmoid'))

        self.cause_model = tf.keras.Sequential()
        self.cause_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)))
        self.cause_model.add(tf.keras.layers.Flatten())
        self.cause_model.add(tf.keras.layers.Dense(self.hidden_layer_size, activation='relu'))
        self.cause_model.add(tf.keras.layers.Dense(self.num_classes, activation='sigmoid'))


    def call(self, clauses):
        """
        Calls the model on the input clauses.
        """
        lower_output = self.lower_model(clauses)
        lower_output = self.attention([lower_output, lower_output, lower_output])

        emotion_probs = self.emotion_model(lower_output)
        cause_probs = self.cause_model(lower_output)

        return emotion_probs, cause_probs

    def get_embeddings(self, clauses):
        """
        Get the representation of the clauses from the lower-level model.
        """
        return self.lower_model(clauses)

    def get_likely_clauses(self, clauses, probs):
        """
        Get the clauses that are labeled as 1.
        """
        labels = np.squeeze(get_labels(probs))
        likely_indices = np.squeeze(np.argwhere(labels))
        return clauses[likely_indices].reshape((-1, self.clause_size))

    def loss(self, cause_probabilities, cause_labels, emotion_probabilities, emotion_labels, alpha=0.5):
        """
        Calculates loss as a weighted sum of the losses for cause classification and emotion classification.
        """
        cause_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(cause_labels, cause_probabilities))
        emotion_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(emotion_labels, emotion_probabilities))
        return alpha*emotion_loss + (1 - alpha)*cause_loss

    def train(self, train_clauses, train_cause_labels, train_emotion_labels):
        """
        Trains the model.
        """
        num_examples = train_clauses.shape[0]

        start_index = 0
        end_index = start_index + self.batch_size

        while (end_index <= num_examples):
            batch_clauses = train_clauses[start_index:end_index]
            batch_cause_labels = train_cause_labels[start_index:end_index]
            batch_emotion_labels = train_emotion_labels[start_index:end_index]

            with tf.GradientTape(persistent=True) as tape:
                emotion_probabilities, cause_probabilities = self.call(batch_clauses)
                loss = self.loss(cause_probabilities, batch_cause_labels, emotion_probabilities, batch_emotion_labels, 0.5)
                # print("EC MODEL TRAIN LOSS: ", loss)

            emotion_trainable_variables = self.lower_model.trainable_variables + self.attention.trainable_variables + self.emotion_model.trainable_variables
            cause_trainable_variables = self.lower_model.trainable_variables + self.attention.trainable_variables + self.cause_model.trainable_variables

            emotion_gradients = tape.gradient(loss, emotion_trainable_variables)
            cause_gradients = tape.gradient(loss, cause_trainable_variables)

            self.optimizer.apply_gradients(zip(emotion_gradients, emotion_trainable_variables))
            self.optimizer.apply_gradients(zip(cause_gradients, cause_trainable_variables))

            start_index = end_index
            end_index = start_index + self.batch_size

    def test(self, test_clauses, test_cause_labels, test_emotion_labels):
        """
        Tests the model. Returns the mean loss from the test data.
        """
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

        return np.mean(loss)


class FilterModel(tf.keras.Model):
    """
    This model is a logistic regression model that filters emotion-cause pairs to obtain the valid pairs in which the cause
    clauses correspond to the emotion clauses.
    """
    def __init__(self):
        super(FilterModel, self).__init__()
        self.batch_size = 1
        self.hidden_layer_size = 100
        self.dense_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

    def call(self, inputs):
        """
        Calls the model on the input pairs.
        """
        return self.dense_3(self.dense_2(self.dense_1(inputs)))

    def loss(self, labels, pred):
        """
        Calculates the binary cross entropy loss for valid pair classification.
        """
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, pred))

    def train(self, train_inputs, train_labels):
        """
        Train the model for the given input pairs, and binary labels (1 if the pair is valid, 0 otherwise).
        """
        current_start = 0
        num_samples = train_inputs.shape[0]
        while current_start + self.batch_size <= num_samples:
            current_batch = train_inputs[current_start:current_start + self.batch_size]
            current_labels = train_labels[current_start:current_start + self.batch_size]

            with tf.GradientTape() as t:
                batch_pred = self.call(current_batch)
                batch_loss = self.loss(current_labels, batch_pred)
                # print("Filter Model Train Loss: ", batch_loss)
            gradients = t.gradient(batch_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            current_start += self.batch_size

    def get_cartesian_products(self, embedding_model, emotion_clauses, cause_clauses, real_pairs):
        """
        Given the set of emotion clauses and the set of cause clauses, obtain the embeddings from the embedding model
        Then produce the Cartesian product of both the clauses and their embeddings.
        """
        emotion_embeddings = embedding_model.get_embeddings(emotion_clauses).numpy()
        emotion_embeddings = emotion_embeddings.reshape((emotion_embeddings.shape[0], -1))
        cause_embeddings = embedding_model.get_embeddings(cause_clauses).numpy()
        cause_embeddings = cause_embeddings.reshape((cause_embeddings.shape[0], -1))
        if cause_embeddings.shape[0] == 0 or emotion_embeddings.shape[0] == 0:
            return np.array([]), np.array([])
        embedding_pairs = np.array([[np.append(e, c) for c in cause_embeddings] for e in emotion_embeddings])
        label_pairs = np.array([[self.is_real_pair(e, c, real_pairs) for c in cause_clauses] for e in emotion_clauses])
        embedding_pairs = embedding_pairs.reshape((-1, embedding_pairs.shape[-1]))
        label_pairs = label_pairs.reshape((-1))

        return embedding_pairs, label_pairs

    def is_real_pair(self, emotion_clause, cause_clause, real_pairs):
        """
        Returns a Boolean representing whether or not the given emotion-cause pair
        is in the list of real emotion-cause pairs.
        """
        for i in range(len(real_pairs)):
            if np.all(real_pairs[i] == [emotion_clause, cause_clause]):
                return True
        return False

def get_labels(probs):
    """
    Get the binary labels for inputs given probabilities. Samples a label (0/1)
    based on the probability for each label, as computed by the model.
    """
    data = probs.numpy()
    labels = np.zeros(data.shape)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels[i][j] = np.random.choice(2, 1, p=[1-data[i][j], data[i][j]])
    return labels

def scores(labels, pred):
    """
    Calculates the precision, recall, and F1 scores.

    :param labels: 1D np.array of actual labels.
    :param pred: 1D np.array of predicted labels.
    :return tuple of precision, recall, and F1 scores.
    """
    true_positive = np.sum(np.logical_and((labels==1), (pred==1)))
    true_negative = np.sum(np.logical_and((labels==0), (pred==0)))
    false_positive = np.sum(np.logical_and((labels==0), (pred==1)))
    false_negative = np.sum(np.logical_and((labels==1), (pred==0)))

    # Precision represents the correct percentage of all positive predicitions
    precision = true_positive / (true_positive + false_positive)
    # Recall represents how well we can predict results that should be positive
    recall = true_positive / (true_positive + false_negative)

    # F1 is the harmonic mean of precision and recall
    # intuitively, it represents the quality of how well we should trust a prediction that's positive
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def main():
    # Obtain preprocessed data.
    train_clauses, test_clauses, train_emotion_labels, test_emotion_labels, \
    train_cause_labels, test_cause_labels, train_emotion_cause_pairs, \
    test_emotion_cause_pairs, word2id, pad_index, clause_size = get_data("data.txt")

    ec_extract_model = ECModel(len(word2id), clause_size)
    train_emotion_labels = np.asarray(train_emotion_labels).astype('float32').reshape((-1,1))
    test_emotion_labels = np.asarray(test_emotion_labels).astype('float32').reshape((-1,1))
    train_cause_labels = np.asarray(train_cause_labels).astype('float32').reshape((-1,1))
    test_cause_labels = np.asarray(test_cause_labels).astype('float32').reshape((-1,1))
    
    # Train ECModel.
    num_epochs = 5

    # For saving/loading models.
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(ec_extract_model=ec_extract_model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    checkpoint.restore(manager.latest_checkpoint)

    if args.mode == "train":
        for e in range(num_epochs):
            print("EC MODEL EPOCH: ", e)
            ec_extract_model.train(train_clauses, train_cause_labels, train_emotion_labels)
        manager.save()

    # Test ECModel.
    num_examples = 30
    train_clauses = train_clauses[:num_examples]
    test_clauses = test_clauses[:num_examples]
    train_emotion_probs, train_cause_probs = ec_extract_model.call(train_clauses)
    test_emotion_probs, test_cause_probs = ec_extract_model.call(test_clauses)

    # Extract emotion and cause clauses.
    train_emotion_clauses = ec_extract_model.get_likely_clauses(train_clauses, train_emotion_probs)
    train_cause_clauses = ec_extract_model.get_likely_clauses(train_clauses, train_cause_probs)

    print("NUMBER OF TRAIN CLAUSES: ", train_clauses.shape)
    print("NUMBER OF EXTRACTED TRAIN EMOTION CLAUSES: ", train_emotion_clauses.shape)
    print("NUMBER OF EXTRACTED TRAIN CAUSE CLAUSES: ", train_cause_clauses.shape)

    test_emotion_clauses = ec_extract_model.get_likely_clauses(test_clauses, test_emotion_probs)
    test_cause_clauses = ec_extract_model.get_likely_clauses(test_clauses, test_cause_probs)

    print("NUMBER OF TEST CLAUSES: ", test_clauses.shape)
    print("NUMBER OF EXTRACTED TEST EMOTION CLAUSES: ", test_emotion_clauses.shape)
    print("NUMBER OF EXTRACTED TEST CAUSE CLAUSES: ", test_cause_clauses.shape)

    # Create filter model.
    pair_filter_model = FilterModel()

    # Apply Cartesian product to set of emotion clauses and set of cause clauses to obtain all possible pairs.
    train_embedding_pairs, train_label_pairs = pair_filter_model.get_cartesian_products(ec_extract_model, train_emotion_clauses, train_cause_clauses, train_emotion_cause_pairs)
    test_embedding_pairs, test_label_pairs = pair_filter_model.get_cartesian_products(ec_extract_model, test_emotion_clauses, test_cause_clauses, test_emotion_cause_pairs)
    train_label_pairs = np.asarray(train_cause_labels).astype('float32').reshape((-1,1))
    test_label_pairs = np.asarray(test_cause_labels).astype('float32').reshape((-1,1))
    
    # Train filter model.
    num_epochs = 25
    for e in range(num_epochs):
        print("FILTER MODEL EPOCH ", e)
        pair_filter_model.train(train_embedding_pairs, train_label_pairs)

    # Test filter model.
    train_pair_probs = pair_filter_model.call(train_embedding_pairs)
    train_predicted_label_pairs = np.squeeze(get_labels(train_pair_probs))

    test_pair_probs = pair_filter_model.call(test_embedding_pairs)
    test_predicted_label_pairs = np.squeeze(get_labels(test_pair_probs))

    # Calculate F1 Score.
    train_f1_score = scores(train_label_pairs, train_predicted_label_pairs)
    print("Train F1 Score: ", train_f1_score)

    test_f1_score = scores(test_label_pairs, test_predicted_label_pairs)
    print("Test F1 Score: ", test_f1_score)

if __name__ == '__main__':
    main()
