import tensorflow as tf
from td_or_not_td.alg.layers import NetworkCreator


class Agent:
    """Builds one of the worker graphs"""
    def __init__(self, rank, session, graph, rm):
        self._p = graph.p
        self._sess = session

        if self._p.data_format == "NHWC":
            data_shape = [None, self._p.screen_res_y, self._p.screen_res_x,
                          self._p.input_image_number]
        else:
            data_shape = [None, self._p.input_image_number, self._p.screen_res_y,
                          self._p.screen_res_x]

        self._placeholder_dict = {
            "learning_rate": graph.learning_rate,
            "image": tf.placeholder(tf.float32, data_shape, name="image"),
            "vector": tf.placeholder(tf.float32, [None, self._p.input_vector_size],
                                     name="vector"),
            "action": tf.placeholder(tf.float32, [None, self._p.number_of_actions], name="action"),
            "target_v": tf.placeholder(tf.float32, [None], name="target_v"),
            "target_r": tf.placeholder(tf.float32, [None, self._p.target_v_size],
                                       name="target_r"),
        }

        self._command_list_dict = graph.build_agent_graph(rank, self._placeholder_dict, rm)

    def run(self, key):
        """Runs one of the available tensorflow ops.

        All used tensorflow ops are created by Graph.build_agent_graph.
        All placeholders are fed by methods of the ReplayMemory.

        Available keys:
          "sync": copy global network weights to local weights
          "action": get action for current state
          "loss": get current loss value
          "train": run one network training step

        Additional key for TD algorithms (Q and a3c):
          "v_list": get value predictions for bootstrapping

        Additional key for the Q algorithm:
          "sync_target": update target network weights


        Args:
          key (str): tensorflow op key
        """

        # command structure:
        #     [tensorflow_op, [list_of_placeholders], [list_of_replay_memory_methods]]
        return self._sess.run(self._command_list_dict[key][0], feed_dict=dict(zip(
            self._command_list_dict[key][1], [f() for f in self._command_list_dict[key][2]]
        )))


class Graph:
    def __init__(self, parameter):
        """creates global and target network weights"""
        self.p = parameter
        self.learning_rate = tf.placeholder(tf.float32, [])

        self._optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                                                    decay=0.99,
                                                    epsilon=1e-1)

        if self.p.data_format == "NHWC":
            data_shape = [None, self.p.screen_res_y, self.p.screen_res_x, self.p.input_image_number]
        else:
            data_shape = [None, self.p.input_image_number, self.p.screen_res_y, self.p.screen_res_x]
        input_image = tf.placeholder(tf.float32, data_shape)
        input_vector = tf.placeholder(tf.float32, [None, self.p.input_vector_size])
        input_action = tf.placeholder(tf.float32, [None, self.p.number_of_actions])

        perception = self.perception_network(
            input_image, "main_perception", self.p.data_format, reuse=False, print_shape=True)

        vector = self.vector_network(
            input_vector, "main_vector", self.p.data_format, reuse=False, print_shape=True)

        self.network(perception, vector, input_action, "main_network", self.p.data_format,
                     reuse=False, print_shape=True)

        self.main_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main_perception") +\
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main_vector") +\
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main_network")

        if self.p.algorithm == "Q":
            perception = self.perception_network(
                input_image, "target_perception", self.p.data_format, reuse=False,
                print_shape=False)

            vector = self.vector_network(
                input_vector, "target_vector", self.p.data_format, reuse=False, print_shape=False)

            self.network(perception, vector, input_action, "target_network", self.p.data_format,
                         reuse=False, print_shape=False)

    def build_agent_graph(self, rank, pl, rm):
        """creates a local networks and the tensorflow computation graph

        Args:
          rank: Thread rank
          pl: Agent,_placeholder_dict
          rm: Instance of ReplayMemory
        """

        command_dict = {}
        # command structure:
        #    [tensorflow_op, [list_of_placeholders], [list_of_replay_memory_methods]]

        network_pre_name = "agent_"
        network_name_ending = "_{}".format(rank)
        reuse = False

        # network creation:

        perception = self.perception_network(
            pl["image"],
            network_pre_name + "perception" + network_name_ending, self.p.data_format, reuse=reuse
        )

        vector = self.vector_network(
            pl["vector"],
            network_pre_name + "vector" + network_name_ending, self.p.data_format, reuse=reuse
        )

        out_list = self.network(
            perception, vector, pl["action"],
            network_pre_name + "network" + network_name_ending, self.p.data_format,
            reuse=reuse
        )

        command_dict["sync"] = [
            self._get_copy_weights_list_operator(
                ["main_perception", "main_vector", "main_network"],
                ["agent_perception_{}".format(rank), "agent_vector_{}".format(rank),
                 "agent_network_{}".format(rank)]),
            [],
            []
        ]

        # computation graph creation:

        def huber_loss(diff):
            clipped_diff = tf.clip_by_value(diff, -1.0, 1.0)
            return tf.stop_gradient(clipped_diff) * diff

        if self.p.algorithm == "a3c":
            pol, v = out_list

            command_dict["v_list"] = [
                v,
                [pl["image"], pl["vector"]],
                [rm.state_list_for_v, rm.vector_list_for_v]
            ]

            command_dict["action"] = [
                tf.multinomial(pol, 1)[:, 0][0],
                [pl["image"], pl["vector"]],
                [rm.next_state, rm.next_vector]
            ]

            softmax_pol = tf.nn.softmax(pol)
            log_output_a = tf.log(softmax_pol + 1e-10)
            entropy = - tf.reduce_sum(tf.multiply(log_output_a, softmax_pol), 1)
            cross_entropy = tf.reduce_sum(tf.multiply(log_output_a, pl["action"]), 1)

            p_loss = - tf.reduce_sum(
                tf.multiply(cross_entropy, tf.stop_gradient(pl["target_v"] - v)) + 0.01 * entropy)

            v_loss = 0.5 * tf.reduce_sum(
                tf.square(pl["target_v"] - v))

            command_dict["loss"] = [
                [v_loss, p_loss],
                [pl["image"], pl["vector"], pl["action"], pl["target_v"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch,
                 rm.value_target_batch]
            ]

            loss = p_loss + 0.5 * v_loss

            grad, __ = zip(*self._optimizer.compute_gradients(loss))
            grad = [x for x in grad if x is not None]
            grad = tf.clip_by_global_norm(grad, 40.)[0]

            gradvar = list(zip(grad, self.main_var))

            if rank == 0:
                print("\n###### learned parameters: ######")
                [print(j) for i, j in gradvar if i is not None]
                print("#################################")

            command_dict["train"] = [
                self._optimizer.apply_gradients(gradvar),
                [pl["image"], pl["vector"], pl["action"], pl["target_v"], pl["learning_rate"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch,
                 rm.value_target_batch, rm.learning_rate]
            ]

        elif self.p.algorithm == "Q":
            q_list, a = out_list

            target_perception = self.perception_network(
                pl["image"], "target_perception", self.p.data_format,
                reuse=True, print_shape=False
            )

            target_vector = self.vector_network(
                pl["vector"], "target_vector", self.p.data_format,
                reuse=True, print_shape=False
            )

            target_q_list, __ = self.network(
                target_perception, target_vector, tf.one_hot(a, self.p.number_of_actions),
                "target_network", self.p.data_format,
                reuse=True, print_shape=False
            )

            command_dict["sync_target"] = [
                self._get_copy_weights_list_operator(
                    ["main_perception", "main_vector", "main_network"],
                    ["target_perception", "target_vector", "target_network"]),
                [],
                []
            ]

            command_dict["action"] = [
                a[0],
                [pl["image"], pl["vector"]],
                [rm.next_state, rm.next_vector]
            ]

            command_dict["v_list"] = [
                target_q_list,
                [pl["image"], pl["vector"]],
                [rm.state_list_for_v, rm.vector_list_for_v]
            ]

            loss = tf.reduce_sum(huber_loss(q_list - pl["target_v"]))

            command_dict["loss"] = [
                [loss],
                [pl["image"], pl["vector"], pl["action"], pl["target_v"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch, rm.value_target_batch]
            ]

            grad, __ = zip(*self._optimizer.compute_gradients(loss))
            grad = [x for x in grad if x is not None]

            gradvar = list(zip(grad, self.main_var))

            if rank == 0:
                print("\n###### learned parameters: ######")
                [print(j) for i, j in gradvar if i is not None]
                print("#################################")

            command_dict["train"] = [
                self._optimizer.apply_gradients(gradvar),
                [pl["image"], pl["vector"], pl["action"], pl["target_v"], pl["learning_rate"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch, rm.value_target_batch,
                 rm.learning_rate]
            ]

        elif self.p.algorithm == "Qmc":
            v_list, a = out_list

            command_dict["action"] = [
                a[0],
                [pl["image"], pl["vector"]],
                [rm.next_state, rm.next_vector]
            ]

            zero_const = tf.constant(0.0, dtype=tf.float32, shape=[self.p.batch_size,
                                                                   self.p.target_v_size])

            loss = tf.reduce_sum(tf.where(
                tf.equal(pl["target_r"], self.p.qmc_no_target_available_encoding),
                zero_const,
                huber_loss(v_list - pl["target_r"])
            ))

            command_dict["loss"] = [
                [loss],
                [pl["image"], pl["vector"], pl["action"], pl["target_r"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch, rm.value_target_batch]
            ]

            grad, __ = zip(*self._optimizer.compute_gradients(loss))
            grad = [x for x in grad if x is not None]
            gradvar = list(zip(grad, self.main_var))

            if rank == 0:
                print("\n###### learned parameters: ######")
                [print(j) for i, j in gradvar if i is not None]
                print("#################################")

            command_dict["train"] = [
                self._optimizer.apply_gradients(gradvar),
                [pl["image"], pl["vector"], pl["action"], pl["target_r"], pl["learning_rate"]],
                [rm.state_batch, rm.vector_batch, rm.action_batch, rm.value_target_batch,
                 rm.learning_rate]
            ]

        else:
            raise ValueError("unknown algorithm")

        return command_dict

    def perception_network(self, input_image, name, data_format, reuse=False, print_shape=False):
        if self.p.use_screen:
            nc = NetworkCreator(name, data_format, reuse, input_image, print_shape)

            nc.conv_layer(32, 8, 4)
            nc.relu()

            nc.conv_layer(64, 4, 2)
            nc.relu()

            nc.conv_layer(64, 3, 1)
            nc.relu()

            nc.fc_layer(512)
            return nc.relu()
        else:
            return None

    def vector_network(self, input_m, name, data_format, reuse=False, print_shape=False):
        if self.p.use_vector_input:
            nc = NetworkCreator(name, data_format, reuse, input_m, print_shape)

            nc.fc_layer(128)
            nc.relu()

            nc.fc_layer(128)
            nc.relu()

            nc.fc_layer(128)
            return nc.relu()
        else:
            return None

    def network(self, perception, vector, action_input, name, data_format,
                reuse=False, print_shape=False):
        nc = NetworkCreator(name, data_format, reuse, perception, print_shape)

        if self.p.use_vector_input and self.p.use_screen:
            assert vector is not None
            tmp = nc.extend(vector)
        elif self.p.use_vector_input:
            tmp = vector
        elif self.p.use_screen:
            tmp = perception
        else:
            raise ValueError("both use_vector_input and use_screen are False")

        if self.p.algorithm == "a3c":
            pol = nc.fc_layer(self.p.number_of_actions, tmp)
            v = tf.reshape(nc.fc_layer(1, tmp), [-1])
            return [pol, v]

        elif self.p.algorithm == "Qmc":
            assert action_input is not None

            if self.p.use_screen:
                nc.fc_layer(512, tmp)
                nc.relu()
                r_a = nc.fc_layer(self.p.number_of_actions * self.p.number_of_predictions)
                r_a = tf.reshape(r_a, [-1, self.p.number_of_actions, self.p.number_of_predictions])

                a = tf.argmax(tf.reduce_sum(self.p.prediction_steps_usage * r_a, 2), 1)

                r_a = r_a - tf.reduce_mean(r_a, reduction_indices=1, keep_dims=True)
                action_input = tf.reshape(action_input, tf.concat([tf.shape(action_input), [1]], 0))
                r_a = tf.reduce_sum(r_a * action_input, 1)

                nc.fc_layer(512, tmp)
                nc.relu()
                r = nc.fc_layer(self.p.number_of_predictions)

                r_list = r + r_a
                return [r_list, a]

            else:
                r_a = nc.fc_layer(self.p.number_of_actions * self.p.number_of_predictions, tmp)
                r_a = tf.reshape(r_a, [-1, self.p.number_of_actions, self.p.number_of_predictions])

                action_input = tf.reshape(action_input, tf.concat([tf.shape(action_input), [1]], 0))
                r_list = tf.reduce_sum(r_a * action_input, 1)

                a = tf.argmax(tf.reduce_sum(self.p.prediction_steps_usage * r_a, 2), 1)
                return [r_list, a]

        elif self.p.algorithm == "Q":
            assert action_input is not None

            if self.p.use_screen:
                nc.fc_layer(512, tmp)
                nc.relu()
                q_a = nc.fc_layer(self.p.number_of_actions)

                a = tf.argmax(q_a, 1)

                q_a = q_a - tf.reduce_mean(q_a, reduction_indices=1, keep_dims=True)
                q_a = tf.reduce_sum(q_a * action_input, 1)

                nc.fc_layer(512, tmp)
                nc.relu()
                q = nc.fc_layer(1)

                q = tf.reshape(q, [-1])

                q_list = q + q_a

                return [q_list, a]

            else:
                q_a = nc.fc_layer(self.p.number_of_actions, tmp)

                a = tf.argmax(q_a, 1)

                q_list = tf.reduce_sum(q_a * action_input, 1)
                return [q_list, a]

        else:
            raise ValueError("unknown algorithm")

    @staticmethod
    def _get_copy_weights_list_operator(list_with_names_c, list_with_names_v):
        """copy weights from network_c to network_v

        Args:
            list_with_names_c: target scopes
            list_with_names_v: source scopes"""
        assign_ops = []
        for network_name_c, network_name_v in zip(list_with_names_c, list_with_names_v):
            vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name_c)
            vars_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name_v)
            for i in range(len(vars_v)):
                assign_ops.append(tf.assign(vars_v[i], vars_c[i]))
        assign_all = tf.group(*assign_ops)
        return assign_all
