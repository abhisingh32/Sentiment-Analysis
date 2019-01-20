# Sentiment-Analysis

Here at the provided assignment, we have to develop a predictive model that can determine the classes (here sentiments) where the social media message would fall in. These medical abstracts describe the current sentiment of the patient. Here I have tried to design assistive technology that can identify, with high precision, the class of problems described in the abstract. In the given dataset, abstracts from 11 different sentiments have been included: from very sad to very happy sentiment. The goal is to develop predictive models that can determine, given a medical abstract, which one of 11 classes it belongs to. 

Here I have tried multiple approaches and analysis the outcome for all thereby able to do comparative analysis and was able to select the best suitable model for the problem that provides best solution. I have taken the following approach/methodology to perform this classification. 

• Import the required modules and libraries 
• Read both the input data files  
• Pre-process the medical abstracts. 
1. Convert abstract into a list of words. 
2. Remove numerical values. 
3. Remove stop words. 
4. Stem the words. 
5. Remove punctuations
6. Convert the text to lower case
7. Lemmatize the words (verbs mostly) with context 
e.g. ‘descending’, ‘descends’, ‘descended’ ‘descend’ 

The aim of both stemming and lemmatization here in this context is to reduce inflectional forms  of words and many a times derivationally related word forms to a common base form. 
• Covert the documents data from the array to the compressed sparse row format. I have chosen compressed sparse row matrix format due to its time and memory efficiency. 
• Slice out the training data and testing data. 
• Normalize the rows of a CSR matrix by their L-2 norm. If copy is True, returns a copy of the normalized matrix.
• Compute cosine similarity of each document in testing set with whole training set.
o	Classify vector x using kNN and majority vote rule given training data and associated classes
• Select the k most similar documents. Extract their class information. 
• If there is not a tie between most similar k documents’ class, then find the majority class among those classes and predict that one. 
• If there is a tie among top similar documents, then predict the class of documents having highest average similarity. 
• Repeat this process for all the documents in the test class and write the output list to the text file. 

Modules and parameters for the code: 
• K=50 (Considering 50 neighbours to find class) 
• NumPy, pandas, SciPy, ScikitLearn modules for data frames and matrix operations 
• Scipy.sparse.csr_matrix for sparse data similarity findings 
• NLTK module functions for data pre-processing – e.g. stop words, lemmatize 
• Collections module for counter containers 

Challenges faced while coding: 
• The model does not converge properly. It seems to have multiple local minima. It became really challenging to find the optimum value to K nearest neighbour to obtain the best accuracy. The pattern is not stable and with limited number of attempts to try on leader board, it was cumbersome to find the appropriate value of K to obtain best F1 score.

There are no special instructions to run the code. I have included the code to download required NLTK packages. 
