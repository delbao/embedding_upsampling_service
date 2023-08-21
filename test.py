import tensorflow as tf
import numpy as np
from common import cosine_similarity_loss

class EmbeddingModelTests(tf.test.TestCase):

    def test_cosine_similarity_loss(self):
        # Predefined cosine similarity matrix
        cosine_angle_similarity_matrix = np.array([
            [1.0, 0.5, 0.2, 0.8],
            [0.5, 1.0, 0.6, 0.3],
            [0.2, 0.6, 1.0, 0.4],
            [0.8, 0.3, 0.4, 1.0]
        ], dtype=np.float32)

        # Test case with 2D embeddings
        embeddings = np.array([
            [0.6, 0.8],
            [0.5, 0.5]
        ], dtype=np.float32)

        y_true = tf.constant(np.arange(len(embeddings)), dtype=tf.int32)
        y_pred = tf.constant(embeddings, dtype=tf.float32)
        
        computed_loss = cosine_similarity_loss(cosine_angle_similarity_matrix)(y_true, y_pred)

        expected_loss = 0.120025
        self.assertAllClose(computed_loss.numpy(), expected_loss, atol=1e-6)
    
    def test_model_output_similarity(self):
        cosine_angle_similarity_matrix = np.load('og_train_cos_theta.npy')
        # Load the pre-trained model
        with tf.keras.utils.custom_object_scope({'loss': cosine_similarity_loss(cosine_angle_similarity_matrix)}):
            model = tf.keras.models.load_model('./trained')
        # Load the input embeddings (assuming they are saved in a numpy format)
        embeddings = np.load('./projected_train_embs.npy')
        np.random.seed(None)
        # Randomly pick two embeddings
        indices = np.random.choice(embeddings.shape[0], 2)
        tf.print(f'{indices=}')
        selected_embeddings = embeddings[indices]
        output_embeddings = model.predict(selected_embeddings)
        output_embeddings_normalized = tf.nn.l2_normalize(output_embeddings, axis=-1)
        computed_similarity = tf.reduce_sum(output_embeddings_normalized[0] * output_embeddings_normalized[1])

        # Fetch the corresponding cosine similarity value from the provided matrix
        expected_similarity = cosine_angle_similarity_matrix[indices[0], indices[1]]

        # Assert that the computed cosine similarity is close to the value from the matrix
        self.assertAllClose(computed_similarity.numpy(), expected_similarity, atol=1e-2)

# Run the test
tf.test.main()
