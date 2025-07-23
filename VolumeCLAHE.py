import tensorflow as tf

def VolumeCLAHE(args):
    
    data, label = args
    
    # Calculate the maximum value in the data tensor
    max_val_patch = tf.cast(tf.reduce_max(data), tf.int32)

    # Compute the histogram of the data tensor
    histogram = tf.histogram_fixed_width(data, value_range=[0.0, tf.cast(max_val_patch + 1, dtype=tf.float16)], nbins=max_val_patch + 1)

    # Clip the histogram values
    clipped_histogram = tf.clip_by_value(histogram, clip_value_min=0, clip_value_max=2)

    # Calculate the redistribution values
    redistribute = tf.cast((histogram - clipped_histogram) / (max_val_patch + 1), tf.int32)

    # Calculate the redistributed histogram
    redistribute_hist = clipped_histogram + redistribute

    # Compute the cumulative distribution function (CDF)
    cdf = tf.cumsum(redistribute_hist)

    # Calculate the maximum value in the CDF
    cdf_max = tf.reduce_max(cdf)

    # Normalize the CDF
    normalized_cdf = cdf / cdf_max

    # Use the normalized CDF to equalize the data
    equalized_data = tf.gather(normalized_cdf, tf.cast(data, tf.int32))

    return equalized_data, label