import numpy as np
import tensorflow as tf

from models.note_rnn import rl_tuner_ops

NOTE_OFF = 0
NO_EVENT = 1

class Evaluator():
    def __init__(self, sequence_length=32):
        """Starts the models internal composition over at beat 0, with no notes.
        Also resets statistics about whether the composition is in the middle of a
        melodic leap.
        """
        self.beat = 0
        self.composition = []
        self.composition_direction = 0
        self.leapt_from = None
        self.steps_since_last_leap = 0
        self.num_notes_in_melody = sequence_length
    
    def reset_composition(self):
        self.beat = 0
        self.composition = []
        self.composition_direction = 0
        self.leapt_from = None
        self.steps_since_last_leap = 0

    def reward_music_theory(self, action):
        """Computes cumulative reward for all music theory functions.

        Args:
            action: A one-hot encoding of the chosen action.
        Returns:
            Float reward value.
        """
        reward = self.reward_key(action)
        tf.logging.debug('Key: %s', reward)
        prev_reward = reward

        reward += self.reward_tonic(action)
        if reward != prev_reward:
            tf.logging.debug('Tonic: %s', reward)
        prev_reward = reward

        reward += self.reward_penalize_repeating(action)
        if reward != prev_reward:
            tf.logging.debug('Penalize repeating: %s', reward)
        prev_reward = reward

        reward += self.reward_penalize_autocorrelation(action)
        if reward != prev_reward:
            tf.logging.debug('Penalize autocorr: %s', reward)
        prev_reward = reward

        reward += self.reward_motif(action)
        if reward != prev_reward:
            tf.logging.debug('Reward motif: %s', reward)
        prev_reward = reward

        reward += self.reward_repeated_motif(action)
        if reward != prev_reward:
            tf.logging.debug('Reward repeated motif: %s', reward)
        prev_reward = reward

        # New rewards based on Gauldin's book, "A Practical Approach to Eighteenth
        # Century Counterpoint"
        reward += self.reward_preferred_intervals(action)
        if reward != prev_reward:
            tf.logging.debug('Reward preferred_intervals: %s', reward)
        prev_reward = reward

        reward += self.reward_leap_up_back(action)
        if reward != prev_reward:
            tf.logging.debug('Reward leap up back: %s', reward)
        prev_reward = reward

        reward += self.reward_high_low_unique(action)
        if reward != prev_reward:
            tf.logging.debug('Reward high low unique: %s', reward)

        return reward

    def random_reward_shift_to_mean(self, reward):
        """Modifies reward by a small random values s to pull it towards the mean.

        If reward is above the mean, s is subtracted; if reward is below the mean,
        s is added. The random value is in the range 0-0.2. This function is helpful
        to ensure that the model does not become too certain about playing a
        particular note.

        Args:
            reward: A reward value that has already been computed by another reward
                function.
        Returns:
            Original float reward value modified by scaler.
        """
        s = np.random.randint(0, 2) * .1
        if reward > .5:
            reward -= s
        else:
            reward += s
        return reward

    def reward_scale(self, obs, action, scale=None):
        """Reward function that trains the model to play a scale.

        Gives rewards for increasing notes, notes within the desired scale, and two
        consecutive notes from the scale.

        Args:
            obs: A one-hot encoding of the observed note.
            action: A one-hot encoding of the chosen action.
            scale: The scale the model should learn. Defaults to C Major if not
                provided.
        Returns:
            Float reward value.
        """

        if scale is None:
            scale = rl_tuner_ops.C_MAJOR_SCALE

        obs = np.argmax(obs)
        action = np.argmax(action)
        reward = 0
        if action == 1:
            reward += .1
        if action > obs and action < obs + 3:
            reward += .05

        if action in scale:
            reward += .01
            if obs in scale:
                action_pos = scale.index(action)
                obs_pos = scale.index(obs)
                if obs_pos == len(scale) - 1 and action_pos == 0:
                    reward += .8
                elif action_pos == obs_pos + 1:
                    reward += .8

        return reward

    def reward_key_distribute_prob(self, action, key=None):
        """Reward function that rewards the model for playing within a given key.

        Any note within the key is given equal reward, which can cause the model to
        learn random sounding compositions.

        Args:
            action: One-hot encoding of the chosen action.
            key: The numeric values of notes belonging to this key. Defaults to C
                Major if not provided.
        Returns:
            Float reward value.
        """
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY

        reward = 0

        action_note = np.argmax(action)
        if action_note in key:
            num_notes_in_key = len(key)
            extra_prob = 1.0 / num_notes_in_key

            reward = extra_prob

        return reward

    def reward_key(self, action, penalty_amount=-1.0, key=None):
        """Applies a penalty for playing notes not in a specific key.

        Args:
            action: One-hot encoding of the chosen action.
            penalty_amount: The amount the model will be penalized if it plays
                a note outside the key.
            key: The numeric values of notes belonging to this key. Defaults to
                C-major if not provided.
        Returns:
            Float reward value.
        """
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY

        reward = 0

        action_note = np.argmax(action)
        if action_note not in key:
            reward = penalty_amount

        return reward

    def reward_tonic(self, action, tonic_note=rl_tuner_ops.C_MAJOR_TONIC,
                                     reward_amount=3.0):
        """Rewards for playing the tonic note at the right times.

        Rewards for playing the tonic as the first note of the first bar, and the
        first note of the final bar.

        Args:
            action: One-hot encoding of the chosen action.
            tonic_note: The tonic/1st note of the desired key.
            reward_amount: The amount the model will be awarded if it plays the
                tonic note at the right time.
        Returns:
            Float reward value.
        """
        action_note = np.argmax(action)
        first_note_of_final_bar = self.num_notes_in_melody - 4

        if self.beat == 0 or self.beat == first_note_of_final_bar:
            if action_note == tonic_note:
                return reward_amount
        elif self.beat == first_note_of_final_bar + 1:
            if action_note == NO_EVENT:
                return reward_amount
        elif self.beat > first_note_of_final_bar + 1:
            if action_note == NO_EVENT or action_note == NOTE_OFF:
                return reward_amount
        return 0.0

    def reward_non_repeating(self, action):
        """Rewards the model for not playing the same note over and over.

        Penalizes the model for playing the same note repeatedly, although more
        repeititions are allowed if it occasionally holds the note or rests in
        between. Reward is uniform when there is no penalty.

        Args:
            action: One-hot encoding of the chosen action.
        Returns:
            Float reward value.
        """
        penalty = self.reward_penalize_repeating(action)
        if penalty >= 0:
            return .1

    def detect_repeating_notes(self, action_note):
        """Detects whether the note played is repeating previous notes excessively.

        Args:
            action_note: An integer representing the note just played.
        Returns:
            True if the note just played is excessively repeated, False otherwise.
        """
        num_repeated = 0
        contains_held_notes = False
        contains_breaks = False

        # Note that the current action yas not yet been added to the composition
        for i in range(len(self.composition)-1, -1, -1):
            if self.composition[i] == action_note:
                num_repeated += 1
            elif self.composition[i] == NOTE_OFF:
                contains_breaks = True
            elif self.composition[i] == NO_EVENT:
                contains_held_notes = True
            else:
                break

        if action_note == NOTE_OFF and num_repeated > 1:
            return True
        elif not contains_held_notes and not contains_breaks:
            if num_repeated > 4:
                return True
        elif contains_held_notes or contains_breaks:
            if num_repeated > 6:
                return True
        else:
            if num_repeated > 8:
                return True

        return False

    def reward_penalize_repeating(self,
                                action,
                                penalty_amount=-100.0):
        """Sets the previous reward to 0 if the same is played repeatedly.

        Allows more repeated notes if there are held notes or rests in between. If
        no penalty is applied will return the previous reward.

        Args:
            action: One-hot encoding of the chosen action.
            penalty_amount: The amount the model will be penalized if it plays
                repeating notes.
        Returns:
            Previous reward or 'penalty_amount'.
        """
        action_note = np.argmax(action)
        is_repeating = self.detect_repeating_notes(action_note)
        if is_repeating:
            return penalty_amount
        else:
            return 0.0

    def reward_penalize_autocorrelation(self,
                                        action,
                                        penalty_weight=3.0):
        """Reduces the previous reward if the composition is highly autocorrelated.

        Penalizes the model for creating a composition that is highly correlated
        with itself at lags of 1, 2, and 3 beats previous. This is meant to
        encourage variety in compositions.

        Args:
            action: One-hot encoding of the chosen action.
            penalty_weight: The default weight which will be multiplied by the sum
                of the autocorrelation coefficients, and subtracted from prev_reward.
        Returns:
            Float reward value.
        """
        composition = self.composition + [np.argmax(action)]
        lags = [1, 2, 3]
        sum_penalty = 0
        for lag in lags:
            coeff = rl_tuner_ops.autocorrelate(composition, lag=lag)
            if not np.isnan(coeff):
                if np.abs(coeff) > 0.15:
                    sum_penalty += np.abs(coeff) * penalty_weight
        return -sum_penalty

    def detect_last_motif(self, composition=None, bar_length=8):
        """Detects if a motif was just played and if so, returns it.

        A motif should contain at least three distinct notes that are not note_on
        or note_off, and occur within the course of one bar.

        Args:
            composition: The composition in which the function will look for a
                recent motif. Defaults to the model's composition.
            bar_length: The number of notes in one bar.
        Returns:
            None if there is no motif, otherwise the motif in the same format as the
            composition.
        """
        if composition is None:
            composition = self.composition

        if len(composition) < bar_length:
            return None, 0

        last_bar = composition[-bar_length:]

        actual_notes = [a for a in last_bar if a != NO_EVENT and a != NOTE_OFF]
        num_unique_notes = len(set(actual_notes))
        if num_unique_notes >= 3:
            return last_bar, num_unique_notes
        else:
            return None, num_unique_notes

    def reward_motif(self, action, reward_amount=3.0):
        """Rewards the model for playing any motif.

        Motif must have at least three distinct notes in the course of one bar.
        There is a bonus for playing more complex motifs; that is, ones that involve
        a greater number of notes.

        Args:
            action: One-hot encoding of the chosen action.
            reward_amount: The amount that will be returned if the last note belongs
                to a motif.
        Returns:
            Float reward value.
        """

        composition = self.composition + [np.argmax(action)]
        motif, num_notes_in_motif = self.detect_last_motif(composition=composition)
        if motif is not None:
            motif_complexity_bonus = max((num_notes_in_motif - 3)*.3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_repeated_motif(self, action, bar_length=8):
        """Detects whether the last motif played repeats an earlier motif played.

        Args:
            action: One-hot encoding of the chosen action.
            bar_length: The number of beats in one bar. This determines how many beats
                the model has in which to play the motif.
        Returns:
            True if the note just played belongs to a motif that is repeated. False
            otherwise.
        """
        composition = self.composition + [np.argmax(action)]
        if len(composition) < bar_length:
            return False, None

        motif, _ = self.detect_last_motif(
                composition=composition, bar_length=bar_length)
        if motif is None:
            return False, None

        prev_composition = self.composition[:-(bar_length-1)]

        # Check if the motif is in the previous composition.
        for i in range(len(prev_composition) - len(motif) + 1):
            for j in range(len(motif)):
                if prev_composition[i + j] != motif[j]:
                    break
            else:
                return True, motif
        return False, None

    def reward_repeated_motif(self,
                            action,
                            bar_length=8,
                            reward_amount=4.0):
        """Adds a big bonus to previous reward if the model plays a repeated motif.

        Checks if the model has just played a motif that repeats an ealier motif in
        the composition.

        There is also a bonus for repeating more complex motifs.

        Args:
            action: One-hot encoding of the chosen action.
            bar_length: The number of notes in one bar.
            reward_amount: The amount that will be added to the reward if the last
                note belongs to a repeated motif.
        Returns:
            Float reward value.
        """
        is_repeated, motif = self.detect_repeated_motif(action, bar_length)
        if is_repeated:
            actual_notes = [a for a in motif if a != NO_EVENT and a != NOTE_OFF]
            num_notes_in_motif = len(set(actual_notes))
            motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_sequential_interval(self, action, key=None):
        """Finds the melodic interval between the action and the last note played.

        Uses constants to represent special intervals like rests.

        Args:
            action: One-hot encoding of the chosen action
            key: The numeric values of notes belonging to this key. Defaults to
                C-major if not provided.
        Returns:
            An integer value representing the interval, or a constant value for
            special intervals.
        """
        if not self.composition:
            return 0, None, None

        prev_note = self.composition[-1]
        action_note = np.argmax(action)

        c_major = False
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY
            c_notes = [2, 14, 26]
            g_notes = [9, 21, 33]
            e_notes = [6, 18, 30]
            c_major = True
            tonic_notes = [2, 14, 26]
            fifth_notes = [9, 21, 33]

        # get rid of non-notes in prev_note
        prev_note_index = len(self.composition) - 1
        while (prev_note == NO_EVENT or
                     prev_note == NOTE_OFF) and prev_note_index >= 0:
            prev_note = self.composition[prev_note_index]
            prev_note_index -= 1
        if prev_note == NOTE_OFF or prev_note == NO_EVENT:
            tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)
            return 0, action_note, prev_note

        tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)

        # get rid of non-notes in action_note
        if action_note == NO_EVENT:
            if prev_note in tonic_notes or prev_note in fifth_notes:
                return (rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH,
                                action_note, prev_note)
            else:
                return rl_tuner_ops.HOLD_INTERVAL, action_note, prev_note
        elif action_note == NOTE_OFF:
            if prev_note in tonic_notes or prev_note in fifth_notes:
                return (rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH,
                                action_note, prev_note)
            else:
                return rl_tuner_ops.REST_INTERVAL, action_note, prev_note

        interval = abs(action_note - prev_note)

        if c_major and interval == rl_tuner_ops.FIFTH and (
                prev_note in c_notes or prev_note in g_notes):
            return rl_tuner_ops.IN_KEY_FIFTH, action_note, prev_note
        if c_major and interval == rl_tuner_ops.THIRD and (
                prev_note in c_notes or prev_note in e_notes):
            return rl_tuner_ops.IN_KEY_THIRD, action_note, prev_note

        return interval, action_note, prev_note

    def reward_preferred_intervals(self, action, scaler=5.0, key=None):
        """Dispenses reward based on the melodic interval just played.

        Args:
            action: One-hot encoding of the chosen action.
            scaler: This value will be multiplied by all rewards in this function.
            key: The numeric values of notes belonging to this key. Defaults to
                C-major if not provided.
        Returns:
            Float reward value.
        """
        interval, _, _ = self.detect_sequential_interval(action, key)
        tf.logging.debug('Interval:', interval)

        if interval == 0:    # either no interval or involving uninteresting rests
            tf.logging.debug('No interval or uninteresting.')
            return 0.0

        reward = 0.0

        # rests can be good
        if interval == rl_tuner_ops.REST_INTERVAL:
            reward = 0.05
            tf.logging.debug('Rest interval.')
        if interval == rl_tuner_ops.HOLD_INTERVAL:
            reward = 0.075
        if interval == rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
            reward = 0.15
            tf.logging.debug('Rest interval after 1st or 5th.')
        if interval == rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH:
            reward = 0.3

        # large leaps and awkward intervals bad
        if interval == rl_tuner_ops.SEVENTH:
            reward = -0.3
            tf.logging.debug('7th')
        if interval > rl_tuner_ops.OCTAVE:
            reward = -1.0
            tf.logging.debug('More than octave.')

        # common major intervals are good
        if interval == rl_tuner_ops.IN_KEY_FIFTH:
            reward = 0.1
            tf.logging.debug('In key 5th')
        if interval == rl_tuner_ops.IN_KEY_THIRD:
            reward = 0.15
            tf.logging.debug('In key 3rd')

        # smaller steps are generally preferred
        if interval == rl_tuner_ops.THIRD:
            reward = 0.09
            tf.logging.debug('3rd')
        if interval == rl_tuner_ops.SECOND:
            reward = 0.08
            tf.logging.debug('2nd')
        if interval == rl_tuner_ops.FOURTH:
            reward = 0.07
            tf.logging.debug('4th')

        # larger leaps not as good, especially if not in key
        if interval == rl_tuner_ops.SIXTH:
            reward = 0.05
            tf.logging.debug('6th')
        if interval == rl_tuner_ops.FIFTH:
            reward = 0.02
            tf.logging.debug('5th')

        tf.logging.debug('Interval reward', reward * scaler)
        return reward * scaler

    def detect_high_unique(self, composition):
        """Checks a composition to see if the highest note within it is repeated.

        Args:
            composition: A list of integers representing the notes in the piece.
        Returns:
            True if the lowest note was unique, False otherwise.
        """
        max_note = max(composition)
        if list(composition).count(max_note) == 1:
            return True
        else:
            return False

    def detect_low_unique(self, composition):
        """Checks a composition to see if the lowest note within it is repeated.

        Args:
            composition: A list of integers representing the notes in the piece.
        Returns:
            True if the lowest note was unique, False otherwise.
        """
        no_special_events = [x for x in composition
                                                 if x != NO_EVENT and x != NOTE_OFF]
        if no_special_events:
            min_note = min(no_special_events)
            if list(composition).count(min_note) == 1:
                return True
        return False

    def reward_high_low_unique(self, action, reward_amount=3.0):
        """Evaluates if highest and lowest notes in composition occurred once.

        Args:
            action: One-hot encoding of the chosen action.
            reward_amount: Amount of reward that will be given for the highest note
                being unique, and again for the lowest note being unique.
        Returns:
            Float reward value.
        """
        if len(self.composition) + 1 != self.num_notes_in_melody:
            return 0.0

        composition = np.array(self.composition)
        composition = np.append(composition, np.argmax(action))

        reward = 0.0

        if self.detect_high_unique(composition):
            reward += reward_amount

        if self.detect_low_unique(composition):
            reward += reward_amount

        return reward

    def detect_leap_up_back(self, action, steps_between_leaps=6):
        """Detects when the composition takes a musical leap, and if it is resolved.

        When the composition jumps up or down by an interval of a fifth or more,
        it is a 'leap'. The model then remembers that is has a 'leap direction'. The
        function detects if it then takes another leap in the same direction, if it
        leaps back, or if it gradually resolves the leap.

        Args:
            action: One-hot encoding of the chosen action.
            steps_between_leaps: Leaping back immediately does not constitute a
                satisfactory resolution of a leap. Therefore the composition must wait
                'steps_between_leaps' beats before leaping back.
        Returns:
            0 if there is no leap, 'LEAP_RESOLVED' if an existing leap has been
            resolved, 'LEAP_DOUBLED' if 2 leaps in the same direction were made.
        """
        if not self.composition:
            return 0

        outcome = 0

        interval, action_note, prev_note = self.detect_sequential_interval(action)

        if action_note == NOTE_OFF or action_note == NO_EVENT:
            self.steps_since_last_leap += 1
            tf.logging.debug('Rest, adding to steps since last leap. It is'
                                             'now: %s', self.steps_since_last_leap)
            return 0

        # detect if leap
        if interval >= rl_tuner_ops.FIFTH or interval == rl_tuner_ops.IN_KEY_FIFTH:
            if action_note > prev_note:
                leap_direction = rl_tuner_ops.ASCENDING
                tf.logging.debug('Detected an ascending leap')
            else:
                leap_direction = rl_tuner_ops.DESCENDING
                tf.logging.debug('Detected a descending leap')

            # there was already an unresolved leap
            if self.composition_direction != 0:
                if self.composition_direction != leap_direction:
                    tf.logging.debug('Detected a resolved leap')
                    tf.logging.debug('Num steps since last leap: %s',
                                                     self.steps_since_last_leap)
                    if self.steps_since_last_leap > steps_between_leaps:
                        outcome = rl_tuner_ops.LEAP_RESOLVED
                        tf.logging.debug('Sufficient steps before leap resolved, '
                                                         'awarding bonus')
                    self.composition_direction = 0
                    self.leapt_from = None
                else:
                    tf.logging.debug('Detected a double leap')
                    outcome = rl_tuner_ops.LEAP_DOUBLED

            # the composition had no previous leaps
            else:
                tf.logging.debug('There was no previous leap direction')
                self.composition_direction = leap_direction
                self.leapt_from = prev_note

            self.steps_since_last_leap = 0

        # there is no leap
        else:
            self.steps_since_last_leap += 1
            tf.logging.debug('No leap, adding to steps since last leap. '
                                             'It is now: %s', self.steps_since_last_leap)

            # If there was a leap before, check if composition has gradually returned
            # This could be changed by requiring you to only go a 5th back in the
            # opposite direction of the leap.
            if (self.composition_direction == rl_tuner_ops.ASCENDING and
                    action_note <= self.leapt_from) or (
                            self.composition_direction == rl_tuner_ops.DESCENDING and
                            action_note >= self.leapt_from):
                tf.logging.debug('detected a gradually resolved leap')
                outcome = rl_tuner_ops.LEAP_RESOLVED
                self.composition_direction = 0
                self.leapt_from = None

        return outcome

    def reward_leap_up_back(self, action, resolving_leap_bonus=5.0,
                                                    leaping_twice_punishment=-5.0):
        """Applies punishment and reward based on the principle leap up leap back.

        Large interval jumps (more than a fifth) should be followed by moving back
        in the same direction.

        Args:
            action: One-hot encoding of the chosen action.
            resolving_leap_bonus: Amount of reward dispensed for resolving a previous
                leap.
            leaping_twice_punishment: Amount of reward received for leaping twice in
                the same direction.
        Returns:
            Float reward value.
        """

        leap_outcome = self.detect_leap_up_back(action)
        if leap_outcome == rl_tuner_ops.LEAP_RESOLVED:
            tf.logging.debug('Leap resolved, awarding %s', resolving_leap_bonus)
            return resolving_leap_bonus
        elif leap_outcome == rl_tuner_ops.LEAP_DOUBLED:
            tf.logging.debug('Leap doubled, awarding %s', leaping_twice_punishment)
            return leaping_twice_punishment
        else:
            return 0.0