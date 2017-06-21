import numpy


class DataSet(object):
    def __init__(self, features, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert features.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (features.shape,
                                                       labels.shape))
            self._num_examples = features.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns]
            if features.shape[3] == 1:
                features = features.reshape(features.shape[0],
                                            features.shape[1] * features.shape[2])
                # Convert from [0, 255] -> [0.0, 1.0].
                features = features.astype(numpy.float32)
                features = numpy.multiply(features, 1.0 / 255.0)

        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set.

        Args:
            fake_data (bool): If True create dummy data of 1. for everything
            batch_size (int):
        """
        if fake_data:
            fake_image = [1.0 for _ in xrange(self._features.shape[1])]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]

        assert batch_size <= self._num_examples

        if self._index_in_epoch == 0 and self._epochs_completed > 0:
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            end = -0

            # Finished epoch
            self._epochs_completed += 1
            self._index_in_epoch = 0
        else:
            end = self._index_in_epoch

            # we will overrun next run
            if start + batch_size == self._num_examples:
                self._epochs_completed += 1
                self._index_in_epoch = 0

        return self._features[start:end], self._labels[start:end]

    def one_iteration_in_batches(self, batch_size):
        self._index_in_epoch = 0
        starting_epoch = self._epochs_completed

        while starting_epoch == self._epochs_completed:
            yield self.next_batch(batch_size)

    def reset(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
