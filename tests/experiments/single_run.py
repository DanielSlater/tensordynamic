import math
import tensorflow as tf

from tensor_dynamic.data.cifar_data import get_cifar_100_data_set_collection
from tensor_dynamic.data.mnist_data import get_mnist_data_set_collection
from tensor_dynamic.layers.categorical_output_layer import CategoricalOutputLayer
from tensor_dynamic.layers.convolutional_layer import ConvolutionalLayer
from tensor_dynamic.layers.flatten_layer import FlattenLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.max_pool_layer import MaxPoolLayer

data_set_collection = get_cifar_100_data_set_collection(validation_ratio=.15)
from tensor_dynamic.node_importance import node_importance_optimal_brain_damage, node_importance_by_removal, \
    node_importance_by_real_activation_from_input_layer_variance

# data_set_collection = get_mnist_data_set_collection(validation_ratio=.15)

for run_number in range(0, 30):
    with tf.Session() as session:
        non_liniarity = tf.nn.relu

        regularizer_coeff = 0.01
        noise = 1.
        last_layer = InputLayer(data_set_collection.features_shape,
                                # drop_out_prob=.5,
                                # layer_noise_std=1.
                                )

        last_layer = ConvolutionalLayer(last_layer, (4, 4, 3), session=session,
                                        node_importance_func=node_importance_by_real_activation_from_input_layer_variance,
                                        batch_normalize_input=True,
                                        layer_noise_std=noise)

        last_layer = MaxPoolLayer(last_layer)

        if len(data_set_collection.features_shape) > 1:
            last_layer = FlattenLayer(last_layer, session)

        for _ in range(1):
            last_layer = HiddenLayer(last_layer, 10, session, non_liniarity=non_liniarity,
                                     node_importance_func=node_importance_by_real_activation_from_input_layer_variance,
                                     layer_noise_std=noise,
                                     batch_normalize_input=True)

        output = CategoricalOutputLayer(last_layer, data_set_collection.labels_shape, session,
                                        batch_normalize_input=True,
                                        loss_cross_entropy_or_log_prob=False,
                                        layer_noise_std=noise,
                                        regularizer_weighting=regularizer_coeff)


        def get_file_root():
            return data_set_collection.name + "_flat_noise_" + str(noise)


        def loss_comparison_evaluation(model, data_set):
            """Use bayesian model comparison to evaluate a trained model

            Args:
                model (OutputLayer): Trained model to evaluate
                data_set (DataSet): data set this model was trained on, tends to be test set, but can be train if set up so

            Returns:
                float : log_probability_og_model_generating_data - log(number_of_parameters)
            """
            loss = -model.session.run(model.last_layer.target_loss_op_predict,
                                      feed_dict={model.input_placeholder: data_set.features,
                                                 model.last_layer.target_placeholder: data_set.labels})
            if math.isinf(loss):
                print("got here")
            with open("cifar-conv/run_" + str(run_number) + "_" + get_file_root() + ".csv", "a") as f:
                f.write(
                    str(model.get_resizable_dimension_size_all_layers()).replace(',', '-') + ", " + str(loss) + "\n")

            return loss


        output.learn_structure_layer_by_layer(data_set_collection.train, data_set_collection.validation,
                                              model_evaluation_function=loss_comparison_evaluation,
                                              start_learn_rate=0.0001, continue_learn_rate=0.0001,
                                              add_layers=True,
                                              save_checkpoint_path='cifar-conv/checkpoint')

        train_log_prob, train_acc, train_error = output.evaluation_stats(data_set_collection.train)

        val_log_prob, val_acc, val_error = output.evaluation_stats(data_set_collection.validation)

        test_log_prob, test_acc, test_error = output.evaluation_stats(data_set_collection.test)

        with open("cifar-conv/results_" + get_file_root() + ".csv", "a") as myfile:
            myfile.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (train_error, train_acc,
                                                        val_error, val_acc,
                                                        test_error, test_acc,
                                                        str(output.get_resizable_dimension_size_all_layers())
                                                        .replace(',', '-'),
                                                        output.get_parameters_all_layers()))

            # (7508.6528, 0.97310001)
            # INFO:tensor_dynamic.layers.output_layer:iterations = 23 error = 7508.65
