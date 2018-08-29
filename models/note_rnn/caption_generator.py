from models.brs_seqgan.generator import Generator
import heapq
import math
import numpy as np
import tensorflow as tf


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
            sentence: List of word ids in the caption.
            state: Model state after generating the previous word.
            logprob: Log-probability of the caption.
            score: Score of the caption.
            metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
            sort: Whether to return the elements in descending sorted order.

        Returns:
            A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    def __init__(self,
               model,
               beam_size=3,
               max_caption_length=20,
               length_normalization_factor=0.0):
        if isinstance(model, Generator):
            self.batch_size = model.batch_size
            self.max_caption_length = max_caption_length
            self.model = model
            self.beam_size = beam_size
        else:
            exit()
        
    def generate_beam(self, sess, image_batch):
        # init_state = sess.run(self.model.lstm.zero_state(batch_size=1, dtype=tf.float32))
        # init_word = np.zeros((self.batch_size, ), dtype=int)
        # last_output, last_state = sess.run([self.model.infer_state, self.model.infer_output], 
        #     feed = {self.model.state_feed: init_state, self.model.word_feed: init_word})
        # partial_list = [] #(batch_size x beam_size x sentence_length)
        # for i in range(self.max_caption_length):
        #     word_candidate = np.argsort(last_output, axis=1)
        #     last_output = []
        #     for i in range(self.beam_size):
        #         last_word = word_candidate[:, i]
        #         partial_list[x].append(word_candidate)
        #         new_output, last_state = sess.run([self.model.infer_state, self.model.infer_output],
        #             feed = {self.model.state_feed: last_state, self.model.word_feed: last_word})
        #         for x in range(self.batch_size):
        #             captions[x].sentence
        # import pdb; pdb.set_trace()
        _, state = self.model.lstm(self.model.image_emb, 
                self.model.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32))
        state = tf.concat(axis=1, values=state)
        initial_state = sess.run(state, feed_dict={self.model.image: image_batch}) # (batch_size * hidden)
        captions = []
        for x in range(self.batch_size):
            captions.append(self.beam_single(sess, initial_state[x]))
        return captions

    def beam_single(self, sess, initial_state):

        initial_beam = Caption(
            sentence=[0],
            state=initial_state,
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)

        for i in range(self.max_caption_length):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

            softmax, new_states = sess.run(
                [self.model.infer_output, self.model.infer_state], 
                feed_dict = { self.model.state_feed: state_feed,
                            self.model.word_feed: input_feed}
                )

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    # if metadata:
                    #     metadata_list = partial_caption.metadata + [metadata[i]]
                    # else:
                    metadata_list = None
                    if w == self.model.n_words: # ending word id
                    #     if self.length_normalization_factor > 0:
                    #         score /= len(sentence)**self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    # else:
                    beam = Caption(sentence, state, logprob, score, metadata_list)
                    partial_captions.push(beam)
        if not complete_captions.size():
            complete_captions = partial_captions

            return complete_captions.extract(sort=True)
