# -*- coding: utf-8 -*-

# File:           losses.py
# COMMENTS:       Maximum Mean Discrepancy (MMD). Gretton, Arthur, et al.,
#                 A kernel two-sample test. The Journal of Machine Learning Research, 2012.
# AUTHOR:         Konstantinos Bousmalis
# PAPER:          Bousmalis, K., Trigeorgis, G., Silberman, N., Krishnan, D., & Erhan, D. (2016). Domain separation
#                 networks. In Advances in neural information processing systems (pp. 343-351).
# REPOSITORY:     https://github.com/tensorflow/models/tree/master/research/domain_adaptation/domain_separation


from functools import partial
import tensorflow as tf
slim = tf.contrib.slim

################################################################################
# SIMILARITY LOSS
################################################################################

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.

  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]

  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].

  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.

  We create a sum of multiple gaussian kernels each having a width sigma_i.

  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))



def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples):
  """MARGINAL MMD. Get the Maximum discrepancy between two representations.

  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]

  Returns:
    a scalar value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  # declare a function based on partial
  gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
  # obtain maximun mean discrepancy
  mmd_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
  mmd_value = tf.maximum(1e-4, mmd_value)
  return mmd_value
