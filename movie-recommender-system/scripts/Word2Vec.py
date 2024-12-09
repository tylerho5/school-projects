### Tyler Ho, Quynh Nguyen
### CS439
### Final Project
### Fall 2024

import numpy as np
import random
from collections import defaultdict

class Word2Vec:
    """
    learns word embeddings from corpus of sentences

    based on Word2Vec using Skip-Gram with Negative Sampling from gensim and tensorflow
    """

    def __init__(self):

        # dimension of word vectors
        self.vecSize = 10
        # max distance between target and surrounding words
        self.window = 5
        # min number of occurences required for unique word
        self.minCount = 1
        # number of passes
        self.epochs = 10
        # number of negative samples to draw for each positive sample
        self.negSamples = 5
        # step size for updating weights
        self.learnRate = 0.025
        # stores unique words
        self.vocab = {}
        # counts of each unique word
        self.wordCountDict = defaultdict(int)
        # maps words to index
        self.word2index = dict()
        # reverse of last dict, maps index to word
        self.index2word = dict()
        # matrix to hold input word embeddings
        self.embedInput = None 
        # matrix to hold output word embeddings
        self.embedOutput = None 
        # token to hold rare or unknown words
        self.UNK = '<UNK>'

    def buildVocab(self, sentences):
        """
        build the vocabulary from the input sentence
        
        :param sentences: list of sublists containing words in sentence
        """

        # count frequency of each word
        for sentence in sentences:
            for word in sentence:
                self.wordCountDict[word] += 1

        # filter words that appear infrequently as set in initalized class object
        filteredWords = [word for word, count in self.wordCountDict.items() if count >= self.minCount]
        if not filteredWords:
            filteredWords = [self.UNK]

        # maps words to indices and indices to words
        self.word2index = {word: idx for idx, word in enumerate(filteredWords)}
        self.word2index[self.UNK] = len(self.word2index)
        self.index2word = {idx: word for word, idx in self.word2index.items()}
        vocabSize = len(self.word2index)

        # initialize input embeddings with random values
        self.embedInput = np.random.uniform(-0.5 / self.vecSize, 0.5 / self.vecSize, 
                                                 (vocabSize, self.vecSize))
                                                 
        # initialize output embedding with zero
        self.embedOutput = np.zeros((vocabSize, self.vecSize))

    def getTrainingData(self, sentences):
        """
        generate training data pairs based on window size
        
        :param sentences: list of sublists containing words in sentence
        :return: list of tuples each holding target word index and context word index
        """

        trainingData = []
        for sentence in sentences:

            # replace rare words with UNK token
            sentence = [word if word in self.word2index else self.UNK for word in sentence]
            for i, targetWord in enumerate(sentence):
                targetIndex = self.word2index[targetWord]

                # set context window
                start = max(0, i - self.window)
                end = min(len(sentence), i + self.window + 1)
                contextWords = sentence[start:i] + sentence[i+1:end]
                for context_word in contextWords:
                    contextIndex = self.word2index[context_word]
                    trainingData.append((targetIndex, contextIndex))

        return trainingData

    def getNegSamples(self, posIndices, numNegSamples):
        """
        generate negative samples for positive context indices
        
        :param posIndices: positive context word indices to avoid
        :param numNegSamples number of negative samples to generate
        :return: list of negative sample indices.
        """

        negatives = []
        vocabSize = len(self.word2index)
        while len(negatives) < numNegSamples:
            negativeIndex = random.randint(0, vocabSize - 1)
            if negativeIndex not in posIndices:
                negatives.append(negativeIndex)

        return negatives

    def train(self, sentences):
        """
        train Word2Vec model on sentences
        
        :param sentences: list of sublists containing words in sentence
        """

        #build vocab
        self.buildVocab(sentences)  

        # generate training pairs
        trainingData = self.getTrainingData(sentences) 
        vocab_size = len(self.word2index)

        # iterate over each epoch
        for epoch in range(self.epochs):

            #shuffle data
            random.shuffle(trainingData)  
            loss = 0

            # iterate over each (target, context) pair
            for targetIndex, contextIndex in trainingData:

                # positive sample
                zPositive = np.dot(self.embedInput[targetIndex], self.embedOutput[contextIndex])
                sigmoidPositive = self.sigmoid(zPositive)

                # cumulative loss
                loss += -np.log(sigmoidPositive + 1e-10)
                gradPositive = self.learnRate * (1 - sigmoidPositive)

                # update output embedding for context word 
                self.embedOutput[contextIndex] += gradPositive * self.embedInput[targetIndex]

                # update input embedding for target word 
                self.embedInput[targetIndex] += gradPositive * self.embedOutput[contextIndex]

                # negative sampling
                negativeIndices = self.getNegSamples({contextIndex}, self.negSamples)
                for negativeIndex in negativeIndices:
                    zNegative = np.dot(self.embedInput[targetIndex], self.embedOutput[negativeIndex])
                    sigmoidNegative = self.sigmoid(zNegative)

                    # cumulative loss
                    loss += -np.log(1 - sigmoidNegative + 1e-10) 
                    gradNegative = self.learnRate * (0 - sigmoidNegative)

                    # update output embedding for negative sample
                    self.embedOutput[negativeIndex] += gradNegative * self.embedInput[targetIndex]

                    # update input embedding for target word
                    self.embedInput[targetIndex] += gradNegative * self.embedOutput[negativeIndex]

    def sigmoid(self, x):
        """
        calculate sigmoid function
        
        :param x: input value/numpy array
        :return: sigmoid of x
        """

        return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

    def getEmbedding(self, word):
        """
        get word's embedding vector
        
        :param word: word to get embedding vector for 
        :return: word's embedding vector
        """

        if word in self.word2index:
            return self.embedInput[self.word2index[word]]
        else:
            return np.zeros(self.vecSize)