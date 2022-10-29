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

import tensorflow as tf
from d3q.core.logging import log
from d3q.experiencereplay.replaymemory_service import \
    ReplayMemoryServiceController
from d3q.sim.sim_service import SimPoolServiceController


class DQNTrainer:
    def __init__(self,
                 game,
                 model: tf.keras.Model,
                 save_model_path: str,
                 replaymemory_srvc: ReplayMemoryServiceController,
                 simpool_srvc: SimPoolServiceController,
                 summary_writer: tf.summary.SummaryWriter,
                 initial_random_policy_threshold: float = 1.0):
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
        """ epoch = steps between target model update and evaluation
        """
        best_avg_score_so_far = 0.0

        while True:
            if self.train_steps_since_last_sync is None or self.train_steps_since_last_sync >= self.game.NUM_STEPS_PER_EPOCH:
                if self.train_steps_since_last_sync is not None:
                    self.random_policy_threshold = self.game.set_random_policy_threshold(self.random_policy_threshold)
                    self.epoch += 1

                self.simpool_srvc.set_target_network(self.model, True, self.random_policy_threshold)
                expect_eval_at_x = self.num_samples_processed

                self.train_steps_since_last_sync = 0

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
                return best_avg_score_so_far * (self.game.GAME_LIMIT_IN_SAMPLES / expect_eval_at_x)

            if self.num_samples_processed >= self.game.GAME_LIMIT_IN_SAMPLES:
                return best_avg_score_so_far

    def optimize_steps(self):
        num_steps = self.game.NUM_OPTIMIZATIONS_PER_STEP
        local_batch_size = self.game.LOCAL_BATCH_SIZE

        log.debug(f"Running {num_steps} optimization steps with a local batch size of {local_batch_size}.")

        total_loss = 0.0

        for step in range(num_steps):
            # Sample a random batch of experiences. It is best to have them uncorrelated.
            observations, actions, rewards, next_observations, nonterminals, virt_indices, sampling_saturation = \
                self.replaymemory_srvc.fetch_sampled_memories(as_tf_tensor_on_device='/device:CPU:0')

            # Perform the optimization step.
            loss, sample_priorities = \
                optimize_step(self.game,
                              self.model,
                              self.optimizer,
                              self.loss_fn,
                              observations,
                              actions,
                              rewards,
                              next_observations,
                              nonterminals)

            # Update sampled memory priorities based on losses.
            self.replaymemory_srvc.update_priorities(virt_indices, sample_priorities.numpy())

            self.num_samples_processed += len(virt_indices)
            with self.summary_writer.as_default(step=self.num_samples_processed):
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("sampling_saturation", sampling_saturation)

            total_loss += float(loss)

        average_loss = total_loss / num_steps
        log.info(f"Average Loss: {average_loss}")


@tf.function
def optimize_step(game,
                  model,
                  optimizer,
                  loss_fn,
                  observations,
                  actions,
                  rewards,
                  next_observations,
                  nonterminals):
    # Compute the best possible Q-value which can be executed from the successor state.
    Q_next_action_values = model(next_observations)
    Q_next_best_actions = tf.math.argmax(Q_next_action_values, axis=1)
    Q_next_best_action_values = tf.gather(Q_next_action_values, Q_next_best_actions, axis=1, batch_dims=1)

    # Based on the best Q-value of the successor state, compute the expected value (a value to be learnt) of chosen action on the predecessor state.
    expected_Q_chosen_action_value = rewards + tf.cast(nonterminals, dtype=tf.float32) * game.Q_GAMMA * Q_next_best_action_values

    with tf.GradientTape() as tape:
        # Compute the Q-value of the actually executed action (so called Pi, or "policy" function value) on the predecessor state.
        Q_action_values = model(observations)
        Q_chosen_action_value = tf.gather(Q_action_values, tf.cast(actions, dtype=tf.int64), axis=1, batch_dims=1)

        # Compute the loss as a criterion function between the actual Q-values and the expected (future reward discounted by time).
        sample_losses = loss_fn(tf.reshape(Q_chosen_action_value, (-1, 1)), tf.reshape(expected_Q_chosen_action_value, (-1, 1)))
        loss = tf.math.reduce_mean(sample_losses)

        # Retrieve the gradients.
        gradients = tape.gradient(loss, model.trainable_variables)

    # TODO: In case of multi-worker training, all-reduce the gradients among trainers.

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    sample_priorities = tf.math.pow(sample_losses, game.LOSS_TO_PRIORITY_BETA)

    return loss, sample_priorities
