import random
from copy import deepcopy

syllables = ["wa", "ba", "ra", "ta", "na", "ka", "la", "pa", "za", "ma", "we", "be", "re", "te",
             "ne", "ke", "le", "pe", "ze", "me", "wi", "bi", "ri", "ti", "ni", "ki", "li", "pi",
             "zi", "mi", "wo", "bo", "ro", "to", "no", "ko", "lo", "po", "zo", "mo"]


class Agent:
    """ Base class for the agents """

    def __init__(self, identifier, num_objects):
        self.brain = None
        self.identifier = identifier
        self.vocabulary = [[] for _ in range(num_objects)]

    def get_vocabulary(self):
        """ Returns the vocabulary """
        return self.vocabulary

    def print_vocabulary(self):
        """ Returns the vocabulary as a formatted string. """
        return '[' + '] ['.join(map(lambda v: ', '.join(v), self.vocabulary)) + ']'

    def speak(self, obj):
        """ Returns a tuple of a word, and if it was invented. """
        num_words = len(self.vocabulary[obj])

        if num_words != 0:
            return self.vocabulary[obj][random.randint(0, num_words - 1)], False
        else:
            return self._invent_word(), True

    def listen(self, obj, word):
        """ Checks if the word for the given object is in the vocabulary. Returns -1 if not found. """
        return obj if word in self.vocabulary[obj] else -1

    def adopt(self, obj, word):
        """ Adopts the word for the given object, discards all other existing words. """
        self.vocabulary[obj] = [word]

    def add_word(self, obj, word):
        """ Add a word for the given object to the vocabulary """
        self.vocabulary[obj].append(word)

    def _invent_word(self):
        """ Invent a word from a set of syllabels """
        return ''.join(random.choices(syllables, k=4))
