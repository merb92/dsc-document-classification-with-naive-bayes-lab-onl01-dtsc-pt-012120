def bag_it(text):
    """
    Create a dictionary of word frequencies in the parameter text
    """
    bag = {}
    for word in text.split():
        bag[word] = bag.get(word, 0) + 1
    return bag

class naive_bayes(object):
    """docstring for .naive_bayes"""

    # __init__(self):


    def fit(self, X, y):

        self.X = X
        self.y = y

        self.p_classes = dict(self.y.value_counts(normalize=True))

        self.word_frequency_dict = {}
        for class_ in self.X.unique():
            freq_dict_for_class = {}
            for text in self.X[self.X == class_]['text']:
                for word in text.split():
                    freq_dict_for_class[word] = freq_dict_for_class.get(word, 0) + 1
            word_frequency_dict[class_] = freq_dict_for_class

        vocab = set()
        for text in self.X:
            for word in text.split():
                vocab.add(word)
        self.V = len(vocab)

    def predict(self, doc):
        classes = [] # list for the classes b/c dict.keys() does not guarentee the order of the list of keys
        posteriors = [] # list of probabilities for each class

        bag = bag_it(doc)

        for class_ in self.word_frequency_dict.keys():
            # get P(class)
            p = np.log(self.p_classes[class_]) # take log to avoid underflow
            # get the conditional log probabilities for P(word|class) for all the words
            # using log probabilities to avoid underflow
            for word in bag.keys():
                numerator = bag[word] + 1
                denominator = self.word_frequency_dict[class_].get(word, 0) + V
                p += np.log(numerator / denominator)
            classes.append(class_)
            posteriors.append(p)
        if return_posteriors:
            print(posteriors)
        return classes[np.argmax(posteriors)]
