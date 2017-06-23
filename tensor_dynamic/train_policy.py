import operator

import sys

from tensor_dynamic.layers.duel_state_relu_layer import DuelStateReluLayer
from utils import train_till_convergence


class TrainPolicy(object):
    def __init__(self, trainer, data_set, batch_size=100,
                 max_iterations=10000,
                 max_hidden_nodes=None,
                 stop_accuracy=None,
                 grow_after_turns_without_improvement=None,
                 start_grow_epoch=None,
                 learn_rate_decay=1.,
                 learn_rate_boost=None,
                 back_loss_on_misclassified_only=False):
        """Class for training networks

        Args:
            trainer (tensor_dynamic.CategoricalTrainer):
            data_set:
            batch_size (int):
            max_iterations (int):
            max_hidden_nodes:
            stop_accuracy:
            grow_after_turns_without_improvement:
            start_grow_epoch:
            learn_rate_decay (float):
            learn_rate_boost (float):
            back_loss_on_misclassified_only (bool):

        Returns:

        """
        self.learn_rate_decay = learn_rate_decay
        self.batch_size = batch_size
        self._data_set = data_set
        self._trainer = trainer
        self.max_iterations = max_iterations
        self.max_hidden_nodes = max_hidden_nodes
        self.stop_accuracy = stop_accuracy
        self.grow_after_epochs_without_improvement = grow_after_turns_without_improvement
        self.start_grow_epoch = start_grow_epoch
        self.learn_rate_boost = learn_rate_boost
        self.back_loss_on_misclassified_only = back_loss_on_misclassified_only

    def train_one_epoch(self):
        cost = self._trainer.train_one_epoch(self._data_set, self.batch_size)
        self._trainer.learn_rate *= self.learn_rate_decay

        return cost

    def train_till_convergence(self, continue_epochs=3, use_validation=True, max_epochs=10000):
        if use_validation:
            def train_one_epoch_validation():
                self.train_one_epoch()
                _, validation_loss = self._trainer.accuracy(self._data_set.validation)
                return validation_loss

            train_till_convergence(train_one_epoch_validation, continue_epochs=continue_epochs, max_epochs=max_epochs)
        else:
            train_till_convergence(self.train_one_epoch, continue_epochs=continue_epochs, max_epochs=max_epochs)

    @property
    def validation_accuracy(self):
        self._trainer.predict(self._data_set.validation.features, self._data_set.validation.labels)

    def run_full(self, verbose=True):
        best_validation_loss = sys.float_info.max
        epochs_since_validation_improvement = 0

        if self.start_grow_epoch:
            for _ in range(self.start_grow_epoch):
                train_loss = self.train_one_epoch()
                print("burn in train loss %s" % train_loss)

        while True:
            train_loss = self.train_one_epoch()
            validation_accuracy, validation_loss = self._trainer.accuracy(self._data_set.validation)
            if verbose:
                print(self._data_set.train.epochs_completed, train_loss, validation_accuracy, validation_loss)

            if self.stop_accuracy and validation_accuracy >= self.stop_accuracy:
                print("hit stopping accuracy with validation score of %s" % validation_accuracy)
                return

            if self._data_set.train.epochs_completed >= self.max_iterations:
                print("hit max iterations %s" % self.max_iterations)
                return

            if best_validation_loss > validation_loss:
                print("new best validation loss %s" % validation_loss)
                best_validation_loss = validation_loss
                epochs_since_validation_improvement = 0
            elif self.grow_after_epochs_without_improvement and epochs_since_validation_improvement > self.grow_after_epochs_without_improvement:
                if self.max_hidden_nodes and sum(
                        [x.output_nodes for x in self._trainer.net.all_layers]) > self.max_hidden_nodes:
                    print("hit stopping number of hidden nodes %s" % self.max_hidden_nodes)
                    return

                if not self.grow_net():
                    print("stopped because we did not grow")
                    return

                epochs_since_validation_improvement = 0
                best_validation_loss = 10000000.
                if self.learn_rate_boost:
                    self._trainer.learn_rate += self.learn_rate_boost
            else:
                epochs_since_validation_improvement += 1

    def grow_net(self):
        # find layer with highest reconstruction error
        if self.back_loss_on_misclassified_only:
            back_losses_per_layer = self._trainer.back_losses_per_layer(self._data_set.train.features)
        else:
            back_losses_per_layer = self._trainer.back_losses_per_layer(self._data_set.train.features,
                                                                        misclassification_only=True,
                                                                        labels=self._data_set.train.labels)

        print("back_losses %s" % [(k.layer_number, v) for k, v in back_losses_per_layer.iteritems()])
        max_layer = max(back_losses_per_layer.iteritems(), key=operator.itemgetter(1))[0]
        print("adding node to layer %s", max_layer.layer_number)
        # TODO: reset best validation loss?
        max_layer.resize(max_layer.output_nodes + 1)
        print("New shape = %s", [x.output_nodes for x in self._trainer.net.all_layers])

        return True


class DuelStateReluTrainPolicy(TrainPolicy):
    def __init__(self, trainer, data_set, batch_size,
                 max_iterations=10000,
                 max_hidden_nodes=None,
                 stop_accuracy=None,
                 grow_after_turns_without_improvement=None,
                 start_grow_epoch=None,
                 learn_rate_decay=1.,
                 learn_rate_boost=None,
                 # back_loss_on_misclassified_only=False
                 ):
        super(DuelStateReluTrainPolicy, self).__init__(trainer, data_set, batch_size,
                                                       max_iterations, max_hidden_nodes, stop_accuracy,
                                                       grow_after_turns_without_improvement,
                                                       start_grow_epoch,
                                                       learn_rate_decay,
                                                       learn_rate_boost)

    def grow_net(self):
        duel_state_relu_layers = [x for x in self._trainer.net.all_layers if isinstance(x, DuelStateReluLayer)]
        grew = False
        for layer in duel_state_relu_layers:
            if layer.inactive_nodes() == 0:
                # this layer has no inactive nodes so add one grow
                layer.resize(layer.output_nodes + 1)
                grew = True
                print("New shape = %s", [x.output_nodes for x in self._trainer.net.all_layers])
            else:
                print("Have inactive nodes")

        return grew
