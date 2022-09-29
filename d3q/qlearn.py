# autopep8: off
#
#         .o8    .oooo.
#        "888  .dP""Y88b
#    .oooo888        ]8P'  .ooooo oo
#   d88' `888      <88b.  d88' `888
#   888   888       `88b. 888   888
#   888   888  o.   .88P  888   888
#   `Y8bod88P" `8bd88P'   `V8bod888
#                               888.
#                               8P'
#                               "

import numpy as np
import tensorflow as tf

from d3q.logging import log
from d3q.replaymemory import ReplayMemory


class QTrainer:
    def __init__(self,
                 game,
                 model: tf.keras.Model,
                 replaymemory: ReplayMemory):
        self.game = game
        self.model = model
        self.replaymemory = replaymemory

        self.loss_fn = game.make_loss_fn()
        self.optimizer = game.make_optimizer()

        # log.info('Broadcasting initial state...')
        # import horovod.tensorflow as hvd
        # self.hvd = hvd
        # hvd.broadcast_variables(self.model.variables + self.optimizer.variables(), root_rank=0)

    def optimize(self):
        num_steps = self.game.NUM_OPTIMIZATION_STEPS_PER_ITER
        num_substeps = self.game.NUM_OPTIMIZATION_SUBSTEPS
        local_batch_size = self.game.LOCAL_BATCH_SIZE

        num_steps = min(num_steps, self.replaymemory.size // local_batch_size)
        assert num_steps > 0

        log.info(f"Running {num_steps} x {num_substeps} optimization steps with a local batch size of {local_batch_size}.")

        total_loss = 0.0

        for step in range(num_steps):
            if num_substeps == 1:
                target_model = self.model
            else:
                target_model = tf.keras.models.clone_model(self.model)

            for substep in range(num_substeps):
                # Sample a random batch of experiences. It is best to have them uncorrelated.
                with tf.device('/device:CPU:0'):
                    observations, actions, rewards, next_observations, nonterminals = self.replaymemory.sample_random()

                #loss = optimizer_step(self.model, target_model, observations, actions, rewards, next_observations, nonterminals, self.loss_fn, self.optimizer, self.game.Q_GAMMA, self.hvd)

                with tf.device('/device:CPU:0'):
                    # Compute the best possible Q-value which can be executed from the successor state.
                    Q_next_action_values = target_model(next_observations)
                    Q_next_best_actions = tf.math.argmax(Q_next_action_values, axis=1)
                    Q_next_best_action_values = tf.gather(Q_next_action_values, Q_next_best_actions, axis=1, batch_dims=1)

                    # Based on the best Q-value of the successor state, compute the expected value (a value to be learnt) of chosen action on the predecessor state.
                    expected_Q_chosen_action_value = rewards + tf.cast(nonterminals, dtype=tf.float32) * self.game.Q_GAMMA * Q_next_best_action_values

                    with tf.GradientTape() as tape:
                        # Compute the Q-value of the actually executed action (so called Pi, or "policy" function value) on the predecessor state.
                        Q_action_values = self.model(observations)
                        Q_chosen_action_value = tf.gather(Q_action_values, tf.cast(actions, dtype=tf.int64), axis=1, batch_dims=1)

                        # Compute the loss as a criterion function between the actual Q-values and the expected (future reward discounted by time).
                        loss = self.loss_fn(Q_chosen_action_value, expected_Q_chosen_action_value)

                        # Retrieve the gradients.
                        gradients = tape.gradient(loss, self.model.trainable_variables)

                    #gradients = [self.hvd.allreduce(gradient, device_dense='/device:HPU:0') for gradient in gradients]

                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                total_loss += float(loss)

        average_loss = total_loss / (num_steps * num_substeps)
        log.info(f"Average Loss: {average_loss}")


# @tf.function
# def optimizer_step(model, target_model, observations, actions, rewards, next_observations, nonterminals, loss_fn, optimizer, q_gamma, hvd):
#     with tf.device('/device:HPU:0'):
#         # Compute the best possible Q-value which can be executed from the successor state.
#         Q_next_action_values = target_model(next_observations)
#         Q_next_best_actions = tf.math.argmax(Q_next_action_values, axis=1)
#         Q_next_best_action_values = tf.gather(Q_next_action_values, Q_next_best_actions, axis=1, batch_dims=1)

#         # Based on the best Q-value of the successor state, compute the expected value (a value to be learnt) of chosen action on the predecessor state.
#         expected_Q_chosen_action_value = rewards + tf.cast(nonterminals, dtype=tf.float32) * q_gamma * Q_next_best_action_values

#         with tf.GradientTape() as tape:
#             # Compute the Q-value of the actually executed action (so called Pi, or "policy" function value) on the predecessor state.
#             Q_action_values = model(observations)
#             Q_chosen_action_value = tf.gather(Q_action_values, tf.cast(actions, dtype=tf.int64), axis=1, batch_dims=1)

#             # Compute the loss as a criterion function between the actual Q-values and the expected (future reward discounted by time).
#             loss = loss_fn(Q_chosen_action_value, expected_Q_chosen_action_value)

#             # Retrieve the gradients.
#             gradients = tape.gradient(loss, model.trainable_variables)

#         gradients = [hvd.allreduce(gradient, device_dense='/device:HPU:0') for gradient in gradients]

#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     return loss


def evaluate(game, model: tf.keras.Model):
    score = 0.0
    env = game.make_env()

    for round in range(game.NUM_EVAL_ROUNDS):
        reward_sum = 0.0
        state0, _ = env.reset()
        for step in range(game.MAX_EVAL_ROUND_STEPS):
            action_values = model(tf.expand_dims(state0, axis=0)).numpy()
            action = np.argmax(action_values)
            state1, reward, terminal, info, _ = game.env_step(env, action)
            reward_sum += reward
            if terminal:
                break
            else:
                state0 = state1
        score += reward_sum

    if game.NUM_EVAL_ROUNDS > 0:
        score /= game.NUM_EVAL_ROUNDS

    return score
