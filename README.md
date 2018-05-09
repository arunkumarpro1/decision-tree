# decision-tree
**Implementation of Decision tree classfier using ID3 algorithm**

The program is designed to work well on mutli valued attributes. The values array in the code must be updated with the permissible values for the attributes. The fully grown decision tree is pruned according to the given prune factor.

**Sample Output**

<Decision Tree>
  
**Pre-Pruned Accuracy**
Number of training instances = 100
Number of training attributes = 5
Total number of nodes in the tree = 20
Number of leaf nodes in the tree = 8
Accuracy of the model on the training dataset = 81.2%
Number of validation instances = 50
Number of validation attributes = 5
Accuracy of the model on the validation dataset before pruning = 72.1%
Number of testing instances = 20
Number of testing attributes = 5
Accuracy of the model on the testing dataset = 60.8%

**Post-Pruned Accuracy**
Number of training instances = 100
Number of training attributes = 5
Total number of nodes in the tree = 20
Number of leaf nodes in the tree = 8
Accuracy of the model on the training dataset = 81.2%
Number of validation instances = 50
Number of validation attributes = 5
Accuracy of the model on the validation dataset before pruning = 72.1%
Number of testing instances = 20
Number of testing attributes = 5
Accuracy of the model on the testing dataset = 60.8%

Language Used:
-------------
Python v3.6


Third Party Libraries Used:
--------------------------
Pandas (for reading dataset and pre-processing)
To install Pandas - execute the command "pip3 install pandas"

How to compile and run?
----------------------
Navigate to the correct folder in command prompt and execute the following command
"python id3DecisionTree.py <training_set_location> <validation_set_location> <test_set_location> <pruning_factor>

Or

Eclipse-PyDev IDE can be used.
