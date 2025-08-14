import numpy as np
import random

# The class implements topic modeling (Latent dirichlet allocation) algorithm using collapsed gibbs sampling as in inference. 
class LDA:
    # topics to extract from the data (Components)
    _numTopics = None
    # vocabulary (unique words) in the dataset
    _arrVocab = None
    #size of vocabulary (count of unique words)
    _numVocabSize = None
    # dataset
    _arrDocs = []
    # dataset size (number of documents)
    _numDocSize = None
    # dirichlet prior (document to topic prior)
    _numAlpha = None
    # dirichlet prior (topic to word prior)
    _numBeta = None
    _ifScalarHyperParameters = True
    # Gibb sampler iterations
    _numGSIterations = None
    # The iterations for initial burnin (update of parameters)
    _numNBurnin = None
    # The iterations for continuous burnin (update of parameters)
    _numSampleLag = None
    
    
    
    # The following attributes are for internal working
    __numTAlpha = None  
    __numVBeta = None   
    __arrTheta = None
    __arrThetaSum = None
    __arrPhi = None
    __arrPhiSum = None
    __arrNDT = None
    __arrNDSum = []
    __arrNTW = None
    __arrNTSum = []
    __arrZ = []
    
    # for alpha to be a list, its size must be equal to the size of the dataset, has value for each doc
    # for beta to be a list, its size must be equal to the number of topics, has value for each topic  
    def __init__(self, config): # numTopics = 2, numAlpha = 1.0, numBeta = 0.01, 
        #numGSIterations = 1000, numNBurnin = 50, numSampleLag = 20, wordsPerTopic = 10):
        self._numTopics = config["numTopics"]
        self._numAlpha = config["numAlpha"]
        self._numBeta = config["numBeta"]
        self._numGSIterations = config["numGSIterations"]
        self._numNBurni = config["numNBurnin"]
        self._numSampleLag = config["numSampleLag"]
        self.__wordsPerTopic = config["wordsPerTopic"]
            
    #load data as integer encoding of words in a sequence (no padding or truncation)
    def getData(self, file_path):
        with open(file_path, 'r') as file:
            integer_encoded_data = file.read()
        self.__loadData(integer_encoded_data)
        self.__loadVocab()
        self.__prepareCollections()

    #load docs and docSize from the dataset
    def __loadData(self, integer_encoded_data):
        rows = integer_encoded_data.split('\n')
         
        #read dataset as documents of words IDs
        for row in rows:
            swordlist = row.split('\t')
            swordlist = list(filter(None, swordlist))   #remove empty items from list
            if len(swordlist) > 0:
                iwordlist = [eval(w) for w in swordlist]    
                self._arrDocs.append(iwordlist)

        # determine dataset size
        self._numDocSize = len(self._arrDocs)
        
        
    #Determine unique words (vocabulary) and count of unique words (vocabSize)    
    def __loadVocab(self):
        #determine unique vocabulary
        uniqueWords = []
        for doc in self._arrDocs:
            for word in doc:
                if word not in uniqueWords:
                    uniqueWords.append(word)
        self._arrVocab = uniqueWords
        self._numVocabSize = len(self._arrVocab)    

    def __prepareCollections(self):
        self.__arrNDSum = np.array([0] * self._numDocSize)
        self.__arrTheta = np.array([[0] * self._numTopics] * self._numDocSize)
        self.__arrThetasum = np.array([[0] * self._numTopics] * self._numDocSize)
        self.__arrNDT = np.array([[0] * self._numTopics] * self._numDocSize)
        
        self.__arrNTSum = np.array([0] * self._numTopics)
        self.__arrPhi = np.array([[0] * self._numVocabSize] * self._numTopics)
        self.__arrPhisum = np.array([[0] * self._numVocabSize] * self._numTopics)
        self.__arrNTW = np.array([[0] * self._numVocabSize] * self._numTopics)

        #Assign values to parameters based on hyper-parameters
        self.__numTAlpha = self._numTopics*self._numAlpha  
        self.__numVBeta = self._numVocabSize*self._numBeta   

        
        for d in range(0, self._numDocSize):
            rowOfZeros = [0] * len(self._arrDocs[d])
            self.__arrZ.append(rowOfZeros)
                
    # Initialize first markov chain randomly
    def randomMarkovChainInitialization(self):
        
        for d in range(self._numDocSize):
            wta = []                        #wta - word topic assignment
            doc = self._arrDocs[d]
            for ind in range(len(doc)): 
                randtopic = random.randint(0, self._numTopics - 1)      # generate a topic number at random
                self.__arrZ[d][ind] = randtopic
                self.__arrNDT[d][randtopic] += 1
                self.__arrNDSum[d] += 1
                wordid = self._arrDocs[d][ind]
                self.__arrNTW[randtopic][wordid] += 1
                self.__arrNTSum[randtopic] += 1
            
    
    #Inference (Collapsed Gibbs Sampling)
    def gibbsSampling(self):
        tAlpha = self._numAlpha * self._numTopics
        vBeta = self._numBeta * self._numVocabSize            
                    
        for it in range(self._numGSIterations):
            for d in range(self._numDocSize):
                dsize = len(self._arrDocs[d])
                for ind in range(dsize):
                    # remove old topic from a word instance
                    oldTopic = self.__arrZ[d][ind]
                    wordid = self._arrDocs[d][ind]
                    self.__arrNDT[d][oldTopic] -= 1
                    self.__arrNDSum[d] -= 1
                    self.__arrNTW[oldTopic][wordid] -= 1
                    self.__arrNTSum[oldTopic] -= 1   

                    # find a new more appropriate tpoic for the word instanc as per current state of the model
                    prob = [0] * self._numTopics
                    
                    for t in range(self._numTopics):
                        prob[t] = ((self.__arrNDT[d][t] + self._numAlpha) / (self.__arrNDSum[d] + tAlpha)) * \
                            (self.__arrNTW[t][wordid] + self._numBeta) / (self.__arrNTSum[t] + vBeta)
                    
                    #cumulate multinomial
                    cdf = prob
                    for x in range(1, len(cdf)):
                        cdf[x] += cdf[x-1]
                    
                    cutoff = random.random() * cdf[-1]
                    newTopic = 0
                    for i in range(len(cdf)):
                        if cdf[i] > cutoff:
                            newTopic = i
                            break
                    #update as per new topic
                    self.__arrZ[d][ind] = newTopic
                    self.__arrNDT[d][newTopic] += 1
                    self.__arrNDSum[d] += 1
                    self.__arrNTW[newTopic][wordid] += 1
                    self.__arrNTSum[newTopic] += 1
                
    def getTopicsPerDocument(self):
        results = ''
        results += "***Topics per Document***\n"
        for d in range(self._numDocSize):
            results += "Document " + str(d) + ":\n"
            for t in range(self._numTopics):
                val = (self.__arrNDT[d][t]+self._numAlpha)/(self.__arrNDSum[d]+self.__numTAlpha)
                results += "Topic " + str(t) + ":" + str(val) + '\t'
            results += '\n'
        #print(results)
        file = open('data/output-data/document-topic-distribution.txt', 'w')
        file.write(results)
        return results
                    
   
    def getWordsPerTopic(self, revdictionary):
        results = {}
        
        for t in range(self._numTopics):
            #results += "\nTopic " + str(t) + ":"
            #flag = 0
            wpt = {}
            for v in range(self._numVocabSize):
                val = (self.__arrNTW[t][v]+self._numBeta)/(self.__arrNTSum[t]+self.__numVBeta)
                wpt[revdictionary[str(v)]] = float(val)
             #   flag += 1
             #   if flag == self.__wordsPerTopic:
             #       break
            wpt = sorted(wpt.items(), key=lambda x: x[1], reverse=True)[:self.__wordsPerTopic]
            results[t] = wpt
        #print(results)
        return results
        
    
    def printall(self):
        print("topics: ", self._numTopics)
        print("dataset: ", self._arrDocs)
        print("dataset size: ", self._numDocSize)
        print("vocab: ", self._arrVocab)
        print("vocab size: ", self._numVocabSize)
        print("ndt: ", self.__arrNDT)
        print("ndsum: ", self.__arrNDSum)
        print("ntw: ", self.__arrNTW)
        print("ntsum: ", self.__arrNTSum)
        print("z: ", self.__arrZ)