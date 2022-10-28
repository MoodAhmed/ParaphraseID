# Import the beautifulsoup
# and request libraries of python.
import requests
import bs4
import json

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
##nltk.download()

from sklearn.linear_model import LogisticRegression
from sklearn import svm

import numpy as np

def countCaps(str):
    cap_count=0
    for i in str:
        if i.isupper():
            cap_count+=1
    return cap_count
def countDigits(str):
    digit_count=0
    for i in str:
        if i.isnumeric():
            digit_count+=1
    return digit_count

def shiftDown(str):
    shifted=[]
    for i in str:
            shifted.append(i.lower())
    return shifted
def comparison(int1, int2):
    if(int1==int2):
        return 1.0
    else:
        if((int1==0) or (int2==0)):
            return (1/(int1+int2))
        else:
            return float(min(int1, int2)/(int1+int2))
def similarity(set, input1, input2,suff):
    countIn1=0
    countIn2=0
    sim=0
    for i in set:
        ##if this input contains an entry from the set list
        for j in input1:
            if(not suff):
                if(j==i):
                    countIn1+=1
            else:
                ##the set to compare is a list of suffixes
                if(j.endswith(i)):
                    countIn1+=1
        for k in input2:
            if (not suff):
                if(k==i):
                    countIn2+=1
            else:
                if(k.endswith(i)):
                    countIn2+=1

        ##find the similarity between the two inputs with regard to this member of the given set
        sim+= comparison(countIn1, countIn2)
        countIn1=0
        countIn2=0

    ##return the average similarity
    return sim/len(set)

##Sort the input into relevant substrings, with and without the comparison set
def divide(sieve, whole, top):
    keep=[]
    ##only return elements included in the given set
    if(top==True):
        for i in whole:
            if i in (set(whole) & set(sieve)):
                keep.append(i)
    ##Only return elements excluded from the given set
    ##Eventually, this will yield keywords, once the basic parts of speech have been sorted out
    else:
        for j in whole:
            if j in (set(whole) - set(sieve)):
                keep.append(j)
    return keep

def synonymList(word):

    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
             synonyms.append(lm.name())#adding into synonyms
    return (set(synonyms))
#Represents the average similarity of the second sentences given the repetitions of words from the first and their synonyms.
def synSim(input1, input2):
    synFreqAve=0
    for i in input1:
        firstOccurences= 0
        secOccurences= 0

        word= i
        synonyms= synonymList(word)
        for j in input1:
            otherWord= j
            if (otherWord in synonyms or otherWord==word):
               ##how often the word in question appears (by definition) in its own sentence
                firstOccurences+=1

        for j in input2:
            otherWord= j
            if (otherWord in synonyms or otherWord==word ):
               ##how often the word in question appears (by definition) in the other sentence
                secOccurences+=1

        ##this value is close to 1 if the two sentence share similar meaning with regard to this word
        synonymFreq= float(secOccurences/firstOccurences)

        synFreqAve+= synonymFreq

    ##A much longer sentence is unlikely to be a paraphrase even if it's composed of synonyms from the original input
    return float(synFreqAve/(len(input1)))
##Averages the relationship in meaning as determined in the previous function. It is important to consider both sentences as the first input, in case one i smuch longer.
def synSimOverall(input1,input2):
    sim1to2 = synSim(input1,input2)
    sim2to1 = synSim(input2,input1)

    return float((sim2to1+sim1to2)/2)


def googleSearch(str):
    results= []
    url = 'https://google.com/search?q=' + str

    request_result = requests.get(url)

    # Creating soup from the fetched request
    soup = bs4.BeautifulSoup(request_result.text,
                         "html.parser")
    # soup.find.all( h3 ) to grab
    # all major headings of our search result,
    heading_object = soup.find_all('h3')

    # Iterate through the object
    # and print it as a string.
    for info in heading_object:
        link =info.getText()
        results.append(link)
    return results
def googleSim(input1, input2):
    results1= googleSearch(input1)
    results2= googleSearch(input2)

    if(len(results1)==0 and len(results2)==0):
        return 0
    else:
        ##refers to links found from both searches
        sameLinks= set(results1) & set(results2)

        ##return the number of similar links over the average number of results for each search
        return float(len(sameLinks)/ ((len(results1)+(len(results2)))/2))

def expectedExtraction(file1,skip, training):
    training_file = open(file1, 'r', encoding='utf-8')

    expResults = []
    skipped=0

    for line in training_file:
        line = line.split("\t")
        exp = float(line[3].strip("\n"))
        if(training==1):
            if (skipped < skip):
                if (exp == 1):
                    skipped += 1
                continue
        expResults.append(exp)

    return expResults
def featureExtraction(file1, skip, testing):
    punc=["'", '!', '.', '?', '$', '(',')', '<','>', '%','$',';',',',':','/','-']

    ##hardcode lists for parts of speech which are less likely to be paraphrased
    articles= ['a','an','the']
    conjunctions= ['for','and','nor','but', 'or','yet','so']
    prepositions= ['about', 'above', 'across', 'after','against','along', 'among', 'ago', 'as', 'at', 'before', 'behind','below','beneath', 'beside','by', 'down','during','except', 'for', 'from','in','into','inside','of', 'off','on','over','past','since','than','through','to','towards','under','until','up','upon','via','with']

    persPronouns= ['i','we','you','he','she','it']
    demoPronouns= ['they','this','these','that','those']
    interPronouns= ['who', 'whoever','whom','whomever','which','whichever']
    relaPronouns= ['each','all','everyone', 'either','one','both','any','such']
    reflPronouns= ['myself','herself','himself','themselves','itself']
    possPronouns= ['yours','mine','theirs', 'his','hers']

    nounSuffs=['acy','al','ance','ence','dom','er','or','ism','ist','ity','ty','ment','ness','ship','sion','tion']
    verbSuffs= ['ate','en','ify','fy','ize', 'ise']
    adjSuffs= ['able','ible','al','esque','ful','ic','ical','ious','ous','ish','ive','less','y']

    training_file= open(file1, 'r', encoding='utf-8')

    firstLines=[]
    secLines=[]
    truth= []
    skipped=0

    #Sort training data into two inputs and the golden truth value
    for line in training_file:
        line=line.split("\t")
        firstLines.append(line[1])
        secLines.append(line[2])
        if(testing==0):
            truth= expectedExtraction(file1,0,1)

    feature_set=[]
    #loop through each sample while adding to the feature set
    for i in range(len(firstLines)):
        feature_row= []

        if (skipped < skip):
            if (truth[i] == 1):
                skipped += 1
            continue

        input1=firstLines[i]
        input2=secLines[i]
        input1=nltk.word_tokenize(input1)
        input2= nltk.word_tokenize(input2)

        len1 = len(input1)
        len2 = len(input2)
        lenSim = comparison(len1, len2)
        feature_row.append(lenSim)

        caps1= countCaps(input1)
        caps2= countCaps(input2)
        capsSim= comparison(caps1, caps2)
        feature_row.append(capsSim)

        ##leave inputs in lowercase for easier comparison
        input1= shiftDown(input1)
        input2= shiftDown(input2)

        digs1= countDigits(input1)
        digs2= countDigits(input2)
        digsSim= comparison(digs1, digs2)
        feature_row.append(digsSim)

        #Sort by punctuation
        punc1 = divide(punc, input1, True)
        punc2 = divide(punc, input2, True)
        puncSim = similarity(punc, punc1, punc2, False)
        feature_row.append(puncSim)

        input1 = divide(punc, input1, False)
        input2 = divide(punc, input2, False)

        ##Sort the inputs by articles
        art1= divide(articles, input1, True)
        art2= divide(articles, input2, True)
        ##Calculate how similar these subsets are
        artSim= similarity(articles, art1, art2, False)
        feature_row.append(artSim)

        input1= divide(articles, input1, False)
        input2 = divide(articles, input2, False)


        ##Repeat this process with all the parts of speech until only keywords remain in the input
        ##Conjunctions
        conj1 = divide(conjunctions, input1, True)
        conj2 = divide(conjunctions, input2, True)
        conjSim = similarity(conjunctions, conj1, conj2, False)
        feature_row.append(conjSim)

        input1 = divide(conjunctions, input1, False)
        input2 = divide(conjunctions, input2, False)

        ##Prepositions
        prep1 = divide(prepositions, input1, True)
        prep2 = divide(prepositions, input2, True)
        prepSim = similarity(prepositions, prep1, prep2, False)
        feature_row.append(prepSim)

        input1 = divide(prepositions, input1, False)
        input2 = divide(prepositions, input2, False)

        ##Personal Pronouns
        pers1 = divide(persPronouns, input1, True)
        pers2 = divide(persPronouns, input2, True)
        persSim = similarity(persPronouns, pers1, pers2, False)
        feature_row.append(persSim)

        input1 = divide(persPronouns, input1, False)
        input2 = divide(persPronouns, input2, False)

        ##Demonstrative Pronouns
        dem1 = divide(demoPronouns, input1, True)
        dem2 = divide(demoPronouns, input2, True)
        demSim = similarity(demoPronouns, dem1, dem2, False)
        feature_row.append(demSim)

        input1 = divide(demoPronouns, input1, False)
        input2 = divide(demoPronouns, input2, False)

        ##Interrogative Pronouns
        intP1 = divide(interPronouns, input1, True)
        intP2 = divide(interPronouns, input2, True)
        intPSim = similarity(interPronouns,intP1, intP2, False)
        feature_row.append(intPSim)

        input1 = divide(interPronouns, input1, False)
        input2 = divide(interPronouns, input2, False)

        ##Relative Pronouns
        rela1 = divide(relaPronouns, input1, True)
        rela2 = divide(relaPronouns, input2, True)
        relaSim = similarity(relaPronouns,rela1, rela2, False)
        feature_row.append(relaSim)

        input1 = divide(relaPronouns, input1, False)
        input2 = divide(relaPronouns, input2, False)

        ##Reflexive Pronouns
        refl1 = divide(reflPronouns, input1, True)
        refl2 = divide(reflPronouns, input2, True)
        reflSim = similarity(reflPronouns, refl1, refl2, False)
        feature_row.append(reflSim)

        input1 = divide(reflPronouns, input1, False)
        input2 = divide(reflPronouns, input2, False)

        ##Possessive Pronouns
        poss1 = divide(possPronouns, input1, True)
        poss2 = divide(possPronouns, input2, True)
        possSim = similarity(possPronouns, poss1, poss2, False)
        feature_row.append(possSim)

        input1 = divide(possPronouns, input1, False)
        input2 = divide(possPronouns, input2, False)

        ##Now that the parsing is done, analyze the suffices of the remaining words.
        nounSuffSim= similarity(nounSuffs, input1, input2, True)
        feature_row.append(nounSuffSim)

        verbSuffSim= similarity(verbSuffs, input1, input2, True)
        feature_row.append(verbSuffSim)

        adjSuffSim= similarity(adjSuffs, input1, input2, True)
        feature_row.append(adjSuffSim)

        ##Recombine the list of remaining keywords into one search query
        ##stitch1= " ".join(input1)
        ##stitch2= " ".join(input2)
        #searchSim= googleSim(stitch1,stitch2)
        ##feature_row.append(searchSim)

        synonymity = synSimOverall(input1,input2)
        feature_row.append(synonymity)
        feature_set.append(feature_row)
    return feature_set

def round(values):
    rounded=[]
    for i in values:
        if (i<0.5):
            rounded.append(0.0)
        else:
            rounded.append(1.0)
    return rounded
def balance(set):
    ones=0
    nones=0
    for i in set:
        if i==0.0:
           nones+=1
        else:
            ones+=1
    return float(ones/nones)

def writeResults(sampleList, predictions):
    file = open(sampleList, 'r', encoding='utf-8')

    samples = []

    for line in file:
        line = line.split("\t")
        samples.append(line[0])

    results= open("MahmoodAhmed_test_result.txt",'w')
    for i in range(0,len(samples)):
        results.write(samples[i])
        results.write('\t')
        results.write(str(predictions[int(i)]))
        results.write('\n')
    results.close()


fileToTrain= "train_with_label.txt"
fileToDev= "dev_with_label.txt"
fileToTest= "test_without_label.txt"



try:
    training_truths=np.load("trainingT.npy")
except FileNotFoundError:
    training_truths= expectedExtraction(fileToTrain,2000,1)
    training_truths = np.array(training_truths)
    np.save("trainingT.npy", training_truths)
    training_truths = np.load("trainingT.npy")

try:
    dev_truth=np.load("devT.npy")
except FileNotFoundError:
    dev_truth= expectedExtraction(fileToDev,0,0)
    dev_truth = np.array(dev_truth)
    np.save("devT.npy", dev_truth)
    dev_truth = np.load("devT.npy")

try:
    dev_features=np.load("devF.npy")
except FileNotFoundError:
    dev_features= featureExtraction(fileToDev,0,0)
    dev_features = np.array(dev_features)
    np.save("devF.npy", dev_features)
    dev_features = np.load("devF.npy")

try:
    test_features=np.load("testF.npy")
except FileNotFoundError:
    test_features= featureExtraction(fileToTest,0,1)
    test_features = np.array(test_features)
    np.save("testF.npy", test_features)
    test_features = np.load("testF.npy")

try:
    training_features = np.load("trainingF.npy")
except FileNotFoundError:
    training_features = featureExtraction(fileToTrain,2000,0)
    training_features = np.array(training_features)
    np.save("trainingF.npy", training_features)
    training_features = np.load("trainingF.npy")

##Matrices of the features and expected results have all been loaded into the program
## Create the model
model = LogisticRegression()
model.fit(training_features, training_truths)

prediction= model.predict(dev_features)

##accuracy of the model
correct=0
for i in range(0,len(dev_truth)):
    if (prediction[i]) == (dev_truth[i]):
        correct+=1
print(float(correct/len(dev_features)))

##In the training set there were 2,000 more positive results than negatives, so the model was biased
b= balance(training_truths)
a= len(training_truths)
numOfNones = a/(b+1)
balancedSet= numOfNones*2

##Create file with test results
predictTest= model.predict(test_features)
writeResults("test_without_label.txt", predictTest)


