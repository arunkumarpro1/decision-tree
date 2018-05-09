import pandas as pd
import math
import sys
import random
import copy

'''Calculate the entropy of the given node'''


def entropy(df, n, target_attribute):
    entropy = 0.0
    for _, p in df[target_attribute].value_counts().iteritems():
        entropy += (-p / n) * math.log(p / n, 2) 
    return entropy


'''ID3 algorithm recursive routine'''


def id3(df, attributes, depth):
    global noNodes, noLeafNodes
    uniques = df.Class.unique()
    if len(uniques) == 1:
        print(uniques[0], end="", flush=True)
        noLeafNodes += 1
        return uniques[0]
    elif len(attributes) == 0:
        mostCommonValue = df[target_attribute].value_counts().idxmax()
        print(mostCommonValue, end="", flush=True)
        noLeafNodes += 1
        return mostCommonValue
    else:
        igMax, splitAttribute = 0, attributes[0]
        n = len(df.index)
        parentEntropy = entropy(df, n, target_attribute)
        for column in attributes:
            childEntropy = 0.0
            for value in values:
                df1 = df[df[column] == value]
                n1 = len(df1.index) 
                childEntropy += (n1 / n) * entropy(df1, n1, target_attribute)
            ig = parentEntropy - childEntropy
            if ig > igMax:
                igMax = ig
                splitAttribute = column 
        newNode = {splitAttribute:{}}  
        for value in values:
            noNodes += 1
            print()
            print('| ' * depth, end="", flush=True)
            print("%s = %d : " % (splitAttribute, value), end="", flush=True)
            newAttributes = attributes[:]
            newAttributes.remove(splitAttribute)
            newDf = df[df[splitAttribute] == value]
            if len(newDf.index) == 0:
                newNode[splitAttribute][value] = df[target_attribute].value_counts().idxmax()
            else:
                childNode = id3(newDf, newAttributes, depth + 1) 
                newNode[splitAttribute][value] = childNode   
        return newNode


'''Calculate the accuracy of a model with the given dataset'''


def accuracy(df, root):
    success = 0.0
    for row in df.itertuples():
        current = root.copy()
        if getattr(row, target_attribute) == classify(row, current):
            success += 1
    return success / len(df.index)


'''Predicts the classification for the given example'''


def classify(row, current):
    while isinstance(current, dict):
        attribute = list(current.keys())[0]
        value = getattr(row, attribute)
        current = current[list(current.keys())[0]]
        current = current[value]
    return current


'''Prunes the decision tree according to the given pruning factor and check accuracy'''


def prune(df, pruneFactor, root):
    global valAccuracy
    newRoot = copy.deepcopy(root)
    count = 0
    while count < pruneFactor:
        newRoot, _, done = removeNodes(df, newRoot, 0, random.randint(1, noNodes))
        if not done:
            continue
        elif (accuracy(dfVal, newRoot) * 100) < valAccuracy:
            newRoot = copy.deepcopy(root)
        else:
            valAccuracy = accuracy(dfVal, newRoot) * 100
            root = copy.deepcopy(newRoot)
        count += 1
    return newRoot            
    

'''Remove the sub tree of a particular node'''


def removeNodes(df, node, label, curr):
    if label == curr:
        if isinstance(node, dict):
            node = df[target_attribute].value_counts().idxmax()
            return node, label, True
    elif isinstance(node, dict):
        for key in node.keys():
            for value in values:
                node[key][value], label, done = removeNodes(df[df[key] == value], node[key][value], label + 1, curr)
                if done:
                    return node, label, True
    return node, label, False


'''To count the number of nodes in the tree'''


def getCount(node):
    count, leafCount = 1, 0
    if isinstance(node, dict):
        for value in values:
            c, lc = getCount(node[list(node.keys())[0]][value])
            count += c
            leafCount += lc
    else:
        leafCount = 1
    return count, leafCount


def plotTree(node, depth):
    if isinstance(node, dict):
        for value in values:
            print()
            print('| ' * depth, end="", flush=True)
            print("%s = %d : " % (list(node.keys())[0], value), end="", flush=True)
            plotTree(node[list(node.keys())[0]][value], depth + 1) 
    else:
        print(node, end="", flush=True)


values = [0, 1]
noNodes, noLeafNodes = 0, 0
target_attribute = 'Class'
df = pd.read_csv(sys.argv[1], skip_blank_lines=True).dropna()
trainingAttributes = list(df)
trainingAttributes.remove(target_attribute)
print("Plot of pre-pruned decision tree model")
print("--------------------------------------")
root = id3(df, trainingAttributes, 0)
print()
print("\n\nPre-Pruned Accuracy")
print("-------------------\n")
print("Number of training instances = %d" % len(df.index))
print("Number of training attributes = %d" % len(trainingAttributes))
print("Total number of nodes in the tree = %d" % noNodes)
print("Number of leaf nodes in the tree = %d" % noLeafNodes)
print("Accuracy of the model on the training dataset = %.2f%%" % (accuracy(df, root) * 100))

dfVal = pd.read_csv(sys.argv[2], skip_blank_lines=True).dropna()
validationAttributes = list(dfVal)
validationAttributes.remove('Class')
valAccuracy = accuracy(dfVal, root) * 100
print("\nNumber of validation instances = %d" % len(dfVal.index))
print("Number of validation attributes = %d" % len(validationAttributes))
print("Accuracy of the model on the validation dataset before pruning = %.2f%%" % valAccuracy)

dfTest = pd.read_csv(sys.argv[3], skip_blank_lines=True).dropna()
testingAttributes = list(dfTest)
testingAttributes.remove('Class')
print("\nNumber of testing instances = %d" % len(dfTest.index))
print("Number of testing attributes = %d" % len(testingAttributes))
print("Accuracy of the model on the testing dataset before pruning = %.2f%%" % (accuracy(dfTest, root) * 100))

root = prune(df, int(float(sys.argv[4]) * noNodes), root)

print("\nPlot of post-pruned decision tree model")
print("--------------------------------------")
plotTree(root, 0)
print()
print("\n\nPost-Pruned Accuracy")
print("-------------------\n")
print("Number of training instances = %d" % len(df.index))
print("Number of training attributes = %d" % len(trainingAttributes))
newNodeCount, newLeafCount = getCount(root)
print("Total number of nodes in the tree = %d" % newNodeCount)
print("Number of leaf nodes in the tree = %d" % newLeafCount)
print("Accuracy of the model on the training dataset = %.2f%%" % (accuracy(df, root) * 100))

print("\nNumber of validation instances = %d" % len(dfVal.index))
print("Number of validation attributes = %d" % len(validationAttributes))
print("Accuracy of the model on the validation dataset after pruning = %.2f%%" % (accuracy(dfVal, root) * 100))

print("\nNumber of testing instances = %d" % len(dfTest.index))
print("Number of testing attributes = %d" % len(testingAttributes))
print("Accuracy of the model on the testing dataset after pruning = %.2f%%" % (accuracy(dfTest, root) * 100))
