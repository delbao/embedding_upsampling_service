import tensorflow as tf


def cosine_similarity_loss(cosine_angle_similarity_matrix):
    cosine_angle_similarity_matrix = tf.cast(
        cosine_angle_similarity_matrix, tf.float32)

    def loss(y_true, y_pred):
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)
        predicted_similarity_matrix = tf.matmul(
            y_pred_normalized, y_pred_normalized, transpose_b=True)
        # Generate all combinations of indices in y_true
        i, j = tf.meshgrid(y_true, y_true)
        indices = tf.stack([tf.reshape(i, [-1]), tf.reshape(j, [-1])], axis=1)
        # Fetch the corresponding cosine similarities
        target_similarity_values = tf.gather_nd(
            cosine_angle_similarity_matrix, indices)
        target_similarity_matrix = tf.reshape(
            target_similarity_values, tf.shape(predicted_similarity_matrix))
        return tf.reduce_mean(tf.square(predicted_similarity_matrix - target_similarity_matrix))

    return loss
