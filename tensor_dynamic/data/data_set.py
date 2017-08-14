import numpy


class DataSet(object):
    def __init__(self, features, labels, fake_data=False,
                 flatten=False,
                 to_binary=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert features.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (features.shape,
                                                       labels.shape))
            self._num_examples = features.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns]
            if flatten:
                assert features.shape[3] == 1
                features = features.reshape(features.shape[0],
                                            features.shape[1] * features.shape[2])

            if to_binary:
                # Convert from [0, 255] -> [0.0, 1.0].
                features = features.astype(numpy.float32)
                features = numpy.multiply(features, 1.0 / 255.0)

        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        """Returns np.Array of features for this dataset, the size of the first dimension should match that of the
        labels property"""
        return self._features

    @property
    def labels(self):
        """Returns np.Array of labels for this dataset, the size of the first dimension should match that of the
        features property"""
        return self._labels

    @property
    def num_examples(self):
        """Returns int for number of examples in this dataset"""
        return self._num_examples

    @property
    def epochs_completed(self):
        """Returns int for the number of epoch of training we have gone through using either the next_batch or
        one_iteration in batches methods"""
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set.

        Args:
            batch_size (int):
        """
        assert batch_size <= self._num_examples

        if self._index_in_epoch == 0 and self._epochs_completed > 0:
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self._num_examples:
            end = None

            # Finished epoch
            self._epochs_completed += 1
            self._index_in_epoch = 0
        else:
            end = self._index_in_epoch

            # we will overrun next run
            if end + batch_size > self._num_examples:
                self._epochs_completed += 1
                self._index_in_epoch = 0

        return self._features[start:end], self._labels[start:end]

    def one_iteration_in_batches(self, batch_size):
        """ This uses the next_batch method, but in contrast to that method this returns a genertor that will terminate
        after exactly one epoch of batches.

        Args:
            batch_size (int):

        Returns:
            Generator of tuple of feautres and labels for each batch
        """
        self._index_in_epoch = 0
        starting_epoch = self._epochs_completed

        while starting_epoch == self._epochs_completed:
            yield self.next_batch(batch_size)

    def reset(self):
        """Reset the epoch count and our position in current epoch"""
        self._index_in_epoch = 0
        self._epochs_completed = 0
