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

import time

import tensorflow as tf
from d3q.core.logging import log
from d3q.experiencereplay.replaymemory_service import \
    ReplayMemoryServiceController
from d3q.sim.sim_service import SimPoolServiceController

MAX_PRIORITY = 100.0  # A failsafe for a large sum of priorities.


class DQNTrainer:
    """ Implements the Q-learning training loop.

        Interacts with replay memory service to fetch sample batches and update their loss-scaled priorities.
        Interacts with simulator pool service to set the action prediction model and request full-game evaluation.
    """

    def __init__(self,
                 game,
                 model: tf.keras.Model,
                 save_model_path: str,
                 replaymemory_srvc: ReplayMemoryServiceController,
                 simpool_srvc: SimPoolServiceController,
                 summary_writer: tf.summary.SummaryWriter,
                 initial_random_policy_threshold):
        self.game = game
        self.model = model
        self.save_model_path = save_model_path
        self.replaymemory_srvc = replaymemory_srvc
        self.simpool_srvc = simpool_srvc
        self.summary_writer = summary_writer

        self.loss_fn = game.make_loss_fn()
        self.optimizer = game.make_optimizer()

        self.random_policy_threshold = initial_random_policy_threshold
        self.epoch = 0
        self.step = 0
        self.train_steps_since_last_sync = None
        self.num_samples_processed = 0

        # TODO: In case of multi-worker training, broadcast initial state from rank:0 to other trainers.

    def optimize_to_goal(self):
        """ Enters the training loop until the score reaches the goal or other limits are hit.

            In this script, epoch means a series of steps between simulators' target model update.
        """
        best_avg_score_so_far = 0.0
        start_time = time.time()

        while True:
            if self.train_steps_since_last_sync is None or self.train_steps_since_last_sync >= self.game.NUM_STEPS_PER_EPOCH:
                if self.train_steps_since_last_sync is not None:
                    self.random_policy_threshold = self.game.set_random_policy_threshold(self.random_policy_threshold)
                    self.epoch += 1

                self.simpool_srvc.set_target_network(self.model, True, self.random_policy_threshold)
                expect_eval_at_x = self.num_samples_processed

                self.train_steps_since_last_sync = 0

                if self.save_model_path:
                    log.debug(f'Saving model: {self.save_model_path}')
                    self.model.save_weights(self.save_model_path)
            else:
                expect_eval_at_x = None

            self.optimize_steps()
            self.step += 1
            self.train_steps_since_last_sync += 1

            if expect_eval_at_x is not None:
                scores = self.simpool_srvc.receive_evalution_scores()

                best_score = max(scores)
                avg_score = sum(scores)/len(scores)

                with self.summary_writer.as_default(step=expect_eval_at_x):
                    tf.summary.scalar("avg_score", avg_score)
                    tf.summary.scalar("best_score", best_score)

                best_avg_score_so_far = max(best_avg_score_so_far, avg_score)

            if best_avg_score_so_far >= self.game.GOAL_SCORE:
                log.info(f'The training succeeded: the evaluation reached an average score goal of {best_avg_score_so_far} after processing {expect_eval_at_x} samples.')
                expect_eval_at_x = max(expect_eval_at_x, 1)  # Prevent division by zero if the model is already well-trained.
                return best_avg_score_so_far * (self.game.GAME_LIMIT_IN_SAMPLES / expect_eval_at_x)

            if self.num_samples_processed >= self.game.GAME_LIMIT_IN_SAMPLES:
                log.info(f'The training failed: the evaluation reached an average score goal of {best_avg_score_so_far} after reaching the game sample limit after processing {expect_eval_at_x} samples.')
                return best_avg_score_so_far

            if time.time() - start_time >= self.game.GAME_LIMIT_IN_SEC:
                log.info(f'The training failed: the evaluation reached an average score goal of {best_avg_score_so_far} after reaching the game time limit of {self.game.GAME_LIMIT_IN_SEC} sec after processing {expect_eval_at_x} samples.')
                return best_avg_score_so_far

    def optimize_steps(self):
        """ Performs a series of Q-learning steps.
        """
        num_optimization_steps = self.game.NUM_OPTIMIZATION_STEPS
        num_optimization_substeps = self.game.NUM_OPTIMIZATION_SUBSTEPS
        local_batch_size = self.game.LOCAL_BATCH_SIZE

        log.debug(f"Running {num_optimization_steps} optimization steps with a local batch size of {local_batch_size}.")

        total_loss = 0.0

        for step in range(num_optimization_steps):
            if num_optimization_substeps == 1:
                target_model = self.model
            else:
                assert num_optimization_substeps > 1
                target_model = tf.keras.models.clone_model(self.model)

            for substep in range(num_optimization_substeps):
                # Sample a random batch of experiences. It is best to have them uncorrelated.
                observations, actions, rewards, next_observations, nonterminals, virt_indices, sampling_saturation = \
                    self.replaymemory_srvc.fetch_sampled_memories(as_tf_tensor_on_device='/device:CPU:0')

                # Perform the optimization step.
                loss, sample_priorities = \
                    optimize_step(self.game,
                                  self.model,
                                  target_model,
                                  self.optimizer,
                                  self.loss_fn,
                                  observations,
                                  actions,
                                  rewards,
                                  next_observations,
                                  nonterminals)

                if not tf.math.is_finite(loss):
                    raise RuntimeError(f'Non-finite training loss')

                # Update sampled memory priorities based on losses.
                self.replaymemory_srvc.update_priorities(virt_indices, sample_priorities.numpy())

                self.num_samples_processed += len(virt_indices)
                with self.summary_writer.as_default(step=self.num_samples_processed):
                    tf.summary.scalar("loss", loss)
                    tf.summary.scalar("sampling_saturation", sampling_saturation)

                total_loss += float(loss)

        average_loss = total_loss / (num_optimization_steps * num_optimization_substeps)
        log.info(f"Average Loss: {average_loss}")


@tf.function
def optimize_step(game,
                  model,
                  target_model,
                  optimizer,
                  loss_fn,
                  observations,
                  actions,
                  rewards,
                  next_observations,
                  nonterminals):
    """ Performs a single Q-learning step.
    """

    # Compute the best possible Q-value which can be executed from the successor state.
    Q_next_action_values = target_model(next_observations, training=False)
    Q_next_best_actions = tf.math.argmax(Q_next_action_values, axis=1)
    Q_next_best_action_values = tf.gather(Q_next_action_values, Q_next_best_actions, axis=1, batch_dims=1)

    # Based on the best Q-value of the successor state, compute the expected value (a value to be learnt) of chosen action on the predecessor state.
    expected_Q_chosen_action_value = rewards + tf.cast(nonterminals, dtype=tf.float32) * game.Q_GAMMA * Q_next_best_action_values

    with tf.GradientTape() as tape:
        # Compute the Q-value of the actually executed action (so called Pi, or "policy" function value) on the predecessor state.
        Q_action_values = model(observations, training=True)
        Q_chosen_action_value = tf.gather(Q_action_values, tf.cast(actions, dtype=tf.int64), axis=1, batch_dims=1)

        # Compute the loss as a criterion function between the actual Q-values and the expected (future reward discounted by time).
        sample_losses = loss_fn(tf.reshape(Q_chosen_action_value, (-1, 1)), tf.reshape(expected_Q_chosen_action_value, (-1, 1)))
        loss = tf.math.reduce_mean(sample_losses)

        # Retrieve the gradients.
        gradients = tape.gradient(loss, model.trainable_variables)

    # TODO: In case of multi-worker training, all-reduce the gradients among trainers.

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    sample_priorities = tf.math.pow(tf.cast(sample_losses, tf.float64), game.LOSS_TO_PRIORITY_EXP_BETA)
    sample_priorities = tf.math.minimum(sample_priorities, MAX_PRIORITY)

    return loss, sample_priorities
