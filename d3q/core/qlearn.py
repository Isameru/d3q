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
from d3q.core.logging import log
from d3q.experiencereplay.replaymemory_service import \
    ReplayMemoryServiceController


class QTrainer:
    def __init__(self,
                 game,
                 model: tf.keras.Model,
                 replaymemory_srvc: ReplayMemoryServiceController):
        self.game = game
        self.model = model
        self.replaymemory_srvc = replaymemory_srvc

        self.loss_fn = game.make_loss_fn()
        self.optimizer = game.make_optimizer()

        # TODO: In case of multi-worker training, broadcast initial state from rank:0 to other trainers.

    def optimize(self):
        num_steps = self.game.NUM_OPTIMIZATION_STEPS_PER_ITER
        num_substeps = self.game.NUM_OPTIMIZATION_SUBSTEPS
        local_batch_size = self.game.LOCAL_BATCH_SIZE

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
                (observations, actions, rewards, next_observations, nonterminals), virt_indices = self.replaymemory_srvc.fetch_sampled_memories(as_tf_tensor_on_device='/device:CPU:0')

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
                    sample_losses = self.loss_fn(tf.reshape(Q_chosen_action_value, (-1, 1)), tf.reshape(expected_Q_chosen_action_value, (-1, 1)))
                    loss = tf.math.reduce_mean(sample_losses)

                    # Retrieve the gradients.
                    gradients = tape.gradient(loss, self.model.trainable_variables)

                # TODO: In case of multi-worker training, all-reduce the gradients among trainers.

                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Update sampled memory priorities based on losses.
                sample_priorities = tf.math.pow(sample_losses, self.game.LOSS_TO_PRIORITY_BETA)
                self.replaymemory_srvc.update_priorities(virt_indices, sample_priorities.numpy())

                total_loss += float(loss)

        average_loss = total_loss / (num_steps * num_substeps)
        log.info(f"Average Loss: {average_loss}")


def evaluate(game, model: tf.keras.Model):
    score = 0.0
    env = game.make_env()

    for round in range(game.NUM_EVAL_ROUNDS):
        reward_sum = 0.0
        state0, _ = env.reset()
        for step in range(game.MAX_ROUND_STEPS):
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
