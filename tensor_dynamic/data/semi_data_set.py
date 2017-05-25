import numpy

from tensor_dynamic.data.data_set import DataSet

# TODO: Fix this it's broken
class SemiDataSet(object):
    def __init__(self, features, labels, unlabeled_features):
        self.unlabeled_features = unlabeled_features

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(features, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        features = features[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(10)[l == 1][0] for l in labels])
        idx = indices[y == 0][:5]
        n_classes = y.max() + 1
        n_from_each_class = unlabeled_features / n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y == c][:n_from_each_class]
            i_labeled += list(i)
        l_images = features[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.unlabeled_features:
            labeled_images, labels = self.labeled_ds.next_batch(self.unlabeled_features)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels