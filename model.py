import tensorflow as tf
import numpy as np
from preprocess import get_data

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

        self.emotion_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_state=True, return_sequences=True))
        self.emotion_dense_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.emotion_dense_2 = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')

        self.cause_biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_state=True, return_sequences=True))
        self.cause_dense_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.cause_dense_2 = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')

        self.flatten = tf.keras.layers.Flatten()

    def call(self, clauses):
        lower_output = self.lower_model(clauses)
        lower_output = self.attention([lower_output, lower_output, lower_output])

        emotion_seq = self.emotion_biLSTM(lower_output)[0]
        emotion_seq = self.flatten(emotion_seq)
        emotion_probs = self.emotion_dense_2(self.emotion_dense_1(emotion_seq))

        cause_seq = self.cause_biLSTM(lower_output)[0]
        cause_seq = self.flatten(cause_seq)
        cause_probs = self.cause_dense_2(self.cause_dense_1(cause_seq))

        return emotion_probs, cause_probs 
    
    def get_embeddings(self, clauses):
        """
        Get the representation of the clauses from the lower-level model.
        """
        return self.lower_model(clauses)
    
    def get_likely_clauses(self, clauses, probs):
        """
        Get the clauses with greater than 50% probability of having label 1.
        """
        labels = np.squeeze(get_labels(probs))
        likely_indices = np.squeeze(np.argwhere(labels))
        return clauses[likely_indices].reshape((-1, self.clause_size))
        
    def loss(self, cause_probabilities, cause_labels, emotion_probabilities, emotion_labels, alpha=0.5):
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
                emotion_probabilities, cause_probabilities = self.call(batch_clauses)
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

        return np.mean(loss)


class FilterModel(tf.keras.Model):
    def __init__(self):
        super(FilterModel, self).__init__()
        self.batch_size = 1
        self.hidden_layer_size = 100
        self.dense_1 = tf.keras.layers.Dense(self.hidden_layer_size)
        self.dense_2 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))
    
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

    def get_cartesian_products(self, embedding_model, emotion_clauses, cause_clauses, real_pairs):
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
        label_pairs = np.array([[self.is_real_pair(e, c, real_pairs) for c in cause_clauses] for e in emotion_clauses])
        embedding_pairs = embedding_pairs.reshape((-1, embedding_pairs.shape[-1]))
        label_pairs = label_pairs.reshape((-1))

        return embedding_pairs, label_pairs

    def is_real_pair(self, emotion_clause, cause_clause, real_pairs):
        for i in range(len(real_pairs)):
            if np.all(real_pairs[i] == [emotion_clause, cause_clause]):
                return True
        return False

def get_labels(probs):
    """
    Get the binary labels for clauses given probabilities.
    """
    data = probs.numpy()
    return (data > 1/2)

# return F1, recall, precision scores
# inputs: labels, pred are 1d np arrays
def scores(labels, pred):
    print("LABELS: ", labels)
    print("PRED: ", pred)
    true_positive = np.sum(np.logical_and((labels==1), (pred==1)))
    true_negative = np.sum(np.logical_and((labels==0), (pred==0)))
    false_positive = np.sum(np.logical_and((labels==0), (pred==1)))
    false_negative = np.sum(np.logical_and((labels==1), (pred==0)))

    print("TRUE P: ", true_positive)
    print("TRUE N: ", true_negative)
    print("FALSE P: ", false_positive)
    print("FALSE N: ", false_negative)

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
    test_emotion_cause_pairs, word2id, pad_index, clause_size = get_data("data.txt")
    
    ec_extract_model = ECModel(len(word2id), clause_size)
    # Train ECModel.
    num_epochs = 1

    # This block of code is to test very small batches to make sure the model is learning.
    num_examples = 5
    train_clauses = train_clauses[:num_examples]
    train_emotion_labels = train_emotion_labels[:num_examples]
    train_cause_labels = train_cause_labels[:num_examples]
    num_epochs = 10
    # Test block ends here.

    for e in range(num_epochs):
        ec_extract_model.train(train_clauses, train_cause_labels, train_emotion_labels)

    # Test ECModel.
    emotion_probs, cause_probs = ec_extract_model.call(train_clauses)

    # Extract emotion and cause clauses.
    emotion_clauses = ec_extract_model.get_likely_clauses(train_clauses, emotion_probs)
    cause_clauses = ec_extract_model.get_likely_clauses(train_clauses, cause_probs)

    print("NUMBER OF TRAIN CLAUSES: ", train_clauses.shape)
    print("NUMBER OF EXTRACTED EMOTION CLAUSES: ", emotion_clauses.shape)
    print("NUMBER OF EXTRACTED CAUSE CLAUSES: ", cause_clauses.shape)

    # Create filter model.
    pair_filter_model = FilterModel()

    # Apply Cartesian product to set of emotion clauses and set of cause clauses to obtain all possible pairs.
    embedding_pairs, label_pairs = pair_filter_model.get_cartesian_products(ec_extract_model, emotion_clauses, cause_clauses, train_emotion_cause_pairs)

    # Train filter model.
    for e in range(num_epochs):
        pair_filter_model.train(embedding_pairs, label_pairs)
    
    # Test filter model.
    pair_probs = pair_filter_model.call(embedding_pairs) 
    predicted_label_pairs = np.squeeze(get_labels(pair_probs))

    # Calculate F1 Score.
    f1_score = scores(label_pairs, predicted_label_pairs)
    print("F1 Score: ", f1_score)

if __name__ == '__main__':
    main()
