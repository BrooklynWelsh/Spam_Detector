import pandas, nltk, math, numpy
import torch as t
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter
from torch import tensor, nn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import OrderedDict
from operator import itemgetter
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

def df_to_tensor(df):
    npMatrix = df.to_numpy()
    return t.from_numpy(npMatrix)

def split_and_get_word_freq(x_train, y_train):
    # Spam word frequency
    spam_word_counts = dict()
    ham_word_counts = dict()
    spam_df = pandas.DataFrame()
    ham_df = pandas.DataFrame()

    for i in range(len(x_train)):
        sentence = word_tokenize(x_train.iloc[i,0])  # Tokenize the current sentence so nltk can remove stop words
        if (y_train.iloc[i, 0] == 'spam'):
            spam_df = spam_df.append(x_train.iloc[i])
            for w in sentence:
                    if w in spam_word_counts:  # Also update word_counts if word isn't a stop word
                        spam_word_counts[w] = spam_word_counts[w] + 1
                    else:
                        spam_word_counts[w] = 1
        else:
            ham_df = ham_df.append(x_train.iloc[i])
            for w in sentence:
                    if w in ham_word_counts:  # Also update word_counts if word isn't a stop word
                        ham_word_counts[w] = ham_word_counts[w] + 1
                    else:
                        ham_word_counts[w] = 1

    return spam_word_counts, ham_word_counts, spam_df, ham_df

def Get_Word_Freq(data, ps):
    word_freqs = dict()
    for row in data:
        sentence = row  # Tokenize the current sentence so nltk can remove stop words
        for w in sentence:
            stemmed_word = ps.stem(w)
            if stemmed_word in word_freqs:  # Stop words already removed
                word_freqs[stemmed_word] = word_freqs[stemmed_word] + 1
            else:
                word_freqs[stemmed_word] = 1

    # Sort the word_counts dictionary
    sorted_word_freqs = OrderedDict(sorted(word_freqs.items(), key=itemgetter(1), reverse=True))

    result = dict()
    i = 0
    # Gets only the first 50 most frequent words
    for x in sorted_word_freqs:
        if i > 49:
            break
        result[x] = sorted_word_freqs[x]
        i = i + 1

    return result


def BayesClassifier(percentTest, df, x, y):
    counts = [0, 0] # Totals for both spam and ham classes
    # Spam will be represented by counts[0], ham by counts[1]

    # Get rid of stop words
    stop_words = set(stopwords.words('english'))  # Set stopwords dictionary to english
    ps = PorterStemmer()  # Initialize for stemming in for loop
    for row in x.itertuples():
        sentence = word_tokenize(row[1])  # Tokenize the current sentence so nltk can remove stop words
        filtered_sentence = []
        for w in sentence:
            stemmed_word = ps.stem(w)   # Stem the word
            if stemmed_word not in stop_words and stemmed_word.isalnum():   # isalnum checks to make sure word isn't punctuation or symbol
                filtered_sentence.append(w)

        # Replace df row with filtered_sentence
        df.loc[row.Index][1] = filtered_sentence

    # Split dataset according to given parameter
    if(percentTest > 0):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(percentTest), random_state=0)
    else:
        x_train = x
        x_test = x
        y_train = y
        y_test = y

    # Split given training set into spam and ham

    # Get word frequencies for each class
    spam_word_counts, ham_word_counts, spam_df, ham_df = split_and_get_word_freq(x_train, y_train)

    # Compute totals for both spam and ham classes in dataset
    for index, row in y_train.iterrows():
        if row['v1'] == 'spam':
            counts[0] = counts[0] + 1
        else:
            counts[1] = counts[1] + 1

    total = counts[0] + counts[1]

    # Get prior probability for each class
    spamPriorProb = float(counts[0] / total)
    hamPriorProb = float(counts[1] / total)

    # These variables count ALL words (including duplicates) in each word count dictionary
    sum_spam_words = sum(spam_word_counts.values())
    sum_ham_words = sum(ham_word_counts.values())

    # Length of the dictionary shows how many DISTINCT words are in each dictionary
    distinct_spam_words = len(spam_word_counts)
    distinct_ham_words = len(ham_word_counts)


    predictions = []
    # Now let's test the classifier using the training data
    for index, row in x_test.iterrows():
        sentence = word_tokenize(row['v2'])
        spam_prob_list = []
        ham_prob_list = []
        for w in sentence:
            if w in spam_word_counts:
                spam_word_prob = (spam_word_counts[w] + 1) / (sum_spam_words + distinct_spam_words)
            else:
                spam_word_prob = 0.0000000000001
            if w in ham_word_counts:
                ham_word_prob = (ham_word_counts[w] + 1) / (sum_ham_words + distinct_ham_words)
            else:
                ham_word_prob = 0.00000000000001

            spam_prob_list.append(spam_word_prob)
            ham_prob_list.append(ham_word_prob)

        # Now calculate probablity for ham and spam
        spam_prob = spamPriorProb
        for value in spam_prob_list:
            spam_prob = spam_prob * value

        ham_prob = hamPriorProb
        for value in ham_prob_list:
            ham_prob = ham_prob * value

        # Append prediction to list (0 for spam, 1 for ham)
        if spam_prob > ham_prob:   predictions.append(0)
        else:                       predictions.append(1)

    prediction_index = 0
    wrong_predictions = 0
    for index, row in y_test.iterrows():
        if predictions[prediction_index] == 0:  prediction = 'spam'
        else:                                   prediction = 'ham'

        if prediction != row['v1']: wrong_predictions += 1

    # Now try it out on
    print("Number of sentences in training set:")
    print(len(x_train))
    print("Number of sentences in test set:")
    print(len(x_test))
    print("The bayesian classifier gave this many correct classifications:")
    print(len(x_test) - wrong_predictions)
    print("Gave this many wrong predictions")
    print(wrong_predictions)
    print("Here's accuracy percentage")
    print(wrong_predictions / len(x_test))
    print("Here it is formatted")
    print('{:.2}'.format((len(x_test) - wrong_predictions) / len(x_test)))
    return (len(x_test) - wrong_predictions) / len(x_test)

def mean(given_dict):
    total_counts = 0
    sum = 0
    for key in given_dict.keys():
        total_counts += 1
        sum = sum + given_dict.get(key)

    return sum / total_counts

def standard_deviation(given_dict):
    dict_mean = mean(given_dict)
    new_dict = dict()   # Create separate dict to store results
    # For each number, subtract mean then square result
    for key, value in given_dict.items():
        new_value = value - dict_mean
        new_value = new_value * new_value
        new_dict[key] = value

    # Now get mean of new values inside new_dict
    new_mean = mean(new_dict)

    # Finally, take square root of that mean
    return math.sqrt(new_mean)

def data_preprocessing():
    # Load data into DataFrame
    path = "spam.csv"
    df = pandas.read_csv(path, encoding="iso-8859-1")

    # Seperate into data and labels
    data_df = df.iloc[:, 1]  # Data is in 2nd column (1st with 0 indexing)
    target_df = df.iloc[:, 0]  # Target (labels) is located in first (0th) column

    data_df = data_df.to_frame()
    target_df = target_df.to_frame()

    nltk.download('stopwords')  # Download stopwords resource from nltk
    nltk.download('punkt')  # Download nltk package for 'tokenize'
    stop_words = set(stopwords.words('english'))  # Set stopwords dictionary to english

    ps = PorterStemmer()  # Initialize for stemming in for loop
    word_counts = dict()  # Dictionary to store word counts

    for index, row in data_df.iterrows():
        sentence = word_tokenize(row[0])  # Tokenize the current sentence so nltk can remove stop words
        filtered_sentence = []  # Initialize an empty string for appending non stop words
        for w in sentence:
            stemmed_word = ps.stem(w)
            if stemmed_word not in stop_words and stemmed_word.isalnum():  # Only append words that aren't stop words AND is alphanumeric (not puncturation) to our filtered sentence
                filtered_sentence.append(
                    stemmed_word)  # Append word the stem of 'w' to filtered_sentence if it's not a stop word
                if stemmed_word in word_counts:  # Also update word_counts if word isn't a stop word
                    word_counts[stemmed_word] = word_counts[stemmed_word] + 1
                else:
                    word_counts[stemmed_word] = 1

        df.at[index, 'v2'] = filtered_sentence  # Set current row value to new tokenized filtered_sentence

    # Sort the word_counts dictionary
    sorted_word_counts = OrderedDict(sorted(word_counts.items(), key=itemgetter(1), reverse=True))
    print("Here is the sorted counts of every word in the document:")
    print(sorted_word_counts)
    print("\n")

    # Now we need to find 50 top words in spam and ham classes
    # Next two lines separates df into DataFrames including only spam and ham in each
    spam_df = df[df['v1'] == 'spam']
    ham_df = df[df['v1'] == 'ham']

    # Seperate into data and labels
    spam_data_df = spam_df.iloc[:, 1]  # Data is in 2nd column (1st with 0 indexing)
    spam_target_df = spam_df.iloc[:, 0]  # Target (labels) is located in first (0th) column

    # Seperate into data and labels
    ham_data_df = ham_df.iloc[:, 1]  # Data is in 2nd column (1st with 0 indexing)
    ham_target_df = ham_df.iloc[:, 0]  # Target (labels) is located in first (0th) column

    # Spam word frequency
    spam_word_counts = dict()
    for row in spam_data_df:
        sentence = row  # Tokenize the current sentence so nltk can remove stop words
        for w in sentence:
            stemmed_word = ps.stem(w)
            if stemmed_word not in stop_words and stemmed_word.isalnum():
                if stemmed_word in spam_word_counts:  # Also update word_counts if word isn't a stop word
                    spam_word_counts[stemmed_word] = spam_word_counts[stemmed_word] + 1
                else:
                    spam_word_counts[stemmed_word] = 1

    # Ham word frequency
    ham_word_counts = dict()
    for row in ham_data_df:
        sentence = row  # Tokenize the current sentence so nltk can remove stop words
        for w in sentence:
            stemmed_word = ps.stem(w)
            if stemmed_word not in stop_words and stemmed_word.isalnum():
                if stemmed_word in ham_word_counts:  # Also update word_counts if word isn't a stop word
                    ham_word_counts[stemmed_word] = ham_word_counts[stemmed_word] + 1
                else:
                    ham_word_counts[stemmed_word] = 1

    # Sort both word_counts in reverse so that it's ordered from greatest to least
    sorted_spam_word_counts = OrderedDict(sorted(spam_word_counts.items(), key=itemgetter(1), reverse=True))
    sorted_ham_word_counts = OrderedDict(sorted(ham_word_counts.items(), key=itemgetter(1), reverse=True))

    # Now get top 50 of each by slicing the sorted dicts

    top_50_spam_words = Get_Word_Freq(spam_data_df, ps)
    top_50_ham_words = Get_Word_Freq(ham_data_df, ps)

    print("Here are the top 50 spam words:")
    print(top_50_spam_words)
    print("\n")
    print("And here are the top 50 ham words:")
    print(top_50_ham_words)
    print("\n")
    # Now use the dictionaries above to calculate the outliers
    # First, calculate the mean
    count_mean = mean(word_counts)

    # Now calculate standard deviation
    count_standard_deviation = standard_deviation(word_counts)

    # Now calculate z-scores for each data point in counts, store in its own dict
    zscore_dict = dict()
    for key, value in word_counts.items():
        zscore_dict[key] = (value - count_mean) / count_standard_deviation

    # Our threshold for this is -1.5 and 1.5, so iterate through zscore_dict and get outliers
    outlier_dict = dict()
    maximum_threshold = 1.5
    minimum_threshold = -1.5

    for key, value in zscore_dict.items():
        if zscore_dict.get(key) > maximum_threshold or zscore_dict.get(key) < minimum_threshold:
            outlier_dict[key] = zscore_dict.get(key)

    print("These are the outliers in the dataset:")
    print(outlier_dict)
    print("\n")

    return df, data_df, target_df, top_50_ham_words, top_50_spam_words



def Part_Three_NNs(data, targets, spam_data, ham_data, use_training=False, split_ratio=0.0):
  classes = numpy.array(normalize(targets))

  if use_training==True:
    # Split dataset according to given parameter
    x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=float(split_ratio), random_state=0)

    spam_preds = SPAM_NN_With_TrainTestSplit(x_train, y_train, x_test, y_test, spam_data)
    ham_preds = HAM_NN_With_TrainTestSplit(x_train, y_train, x_test, y_test, spam_data, ham_data)
  else:
    spam_preds = SPAM_NN(data, classes, spam_data)
    ham_preds = HAM_NN(data, classes, spam_data, ham_data)

  return spam_preds, ham_preds

def normalize(classes):
  result = []
  for row in classes.values:
    if row == "ham": result.append(0)
    else: result.append(1)
  return result

def NN_GradDesc(iterations, model, criterion, optimizer, x_train, y_train, x_test=None, y_test=None):
  #The Training of the NN
  for epoch in range(iterations):
    y_pred = model(x_train.float())
    loss = criterion(y_pred, y_train.float())
    #print('epoch: ', epoch,' loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  #Testing 
  if not isinstance(x_test, pandas.DataFrame): x_test = x_train               #If not using a train/test split must use the train data for the test data
  x_test_results = model(x_test.float())

  prediction = t.argmax(x_test_results)
  print(prediction)

  if not isinstance(y_test, pandas.DataFrame): y_test = y_train               #If not using a train/test split must use the train data for the test data
  print(y_test)                                          

  # Testing
  n_tests = 0
  n_incorrectPredictions = 0

  for index, row in enumerate(x_test):
      output = model(row.float())
      predicted = t.argmax(output)
      if predicted != y_test[index]:
          n_incorrectPredictions += 1

  # Printing results
  print("Out of", len(x_test), "rows, our NN gave", n_incorrectPredictions, "wrong predictions.")

  return (len(x_test) - n_incorrectPredictions) / len(x_test)

def SPAM_NN(full_data, full_targets, spam):
  input_vector = populate_vector(full_data, spam)
  x = t.tensor(input_vector)
  y = t.tensor(full_targets)

  #Defining some hyper parameters
  batch_size, n_in, n_h, n_out = 10, 50, 25, 1        #Arbitrary Batch size, input layer given as 50, hidden layer commonly half input size, Output is binary

  #Creating the model
  model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())
  criterion = t.nn.MSELoss()
  optimizer = t.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate

  print("1st Version NN")
  correct_preds = NN_GradDesc(50, model, criterion, optimizer, x, y)
  return correct_preds

def SPAM_NN_With_TrainTestSplit(x_train, y_train, x_test, y_test, spam_data): 
  input_vector = populate_vector(x_train, spam_data)
  x = t.tensor(input_vector)
  y = t.from_numpy(y_train)

  test_vector = populate_vector(x_test, spam_data)
  x_test = t.tensor(test_vector)
  y_test = t.from_numpy(y_test)

  #Defining some hyper parameters
  batch_size, n_in, n_h, n_out = 10, 50, 25, 1        #Arbitrary Batch size, input layer given as 50, hidden layer commonly half input size, Output is binary

  #Creating the model
  model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())
  criterion = t.nn.MSELoss()
  optimizer = t.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate

  print("1st Version NN")
  correct_preds = NN_GradDesc(50, model, criterion, optimizer, x, y, x_test, y_test)
  return correct_preds

def populate_vector(data, spam_words):
  arr = []
  for row in data.values:
    d = dict()
    for key in spam_words:
      if key in row: d[key] = 1
      else: d[key] = 0
    arr.append(list(d.values()))

  return arr

def HAM_NN(data, targets, spam, ham):
  input_vector = populate_vectorTwo(data, spam, ham)
  x = t.tensor(input_vector)
  y = t.tensor(targets)

  #Defining some hyper parameters
  batch_size, n_in, n_h, n_out = 10, 50, 25, 1     #Arbitrary Batch size, input layer given as 50, hidden layer commonly half input size, Output is binary

  #Creating the model
  model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())
  criterion = t.nn.MSELoss()
  optimizer = t.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate

  print("2nd Version NN")
  correct_preds = NN_GradDesc(50, model, criterion, optimizer, x, y)
  return correct_preds

def HAM_NN_With_TrainTestSplit(x_train, y_train, x_test, y_test, spam_data, ham_data):
  input_vector = populate_vectorTwo(x_train, spam_data, ham_data)
  x = t.tensor(input_vector)
  y = t.tensor(y_train)

  test_vector = populate_vector(x_test, spam_data)
  x_test = t.tensor(test_vector)
  y_test = t.from_numpy(y_test)

  #Defining some hyper parameters
  batch_size, n_in, n_h, n_out = 10, 50, 25, 1     #Arbitrary Batch size, input layer given as 50, hidden layer commonly half input size, Output is binary

  #Creating the model
  model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())
  criterion = t.nn.MSELoss()
  optimizer = t.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate

  print("2nd Version NN")
  correct_preds = NN_GradDesc(50, model, criterion, optimizer, x, y, x_test, y_test)
  return correct_preds

def populate_vectorTwo(data, spam_words, ham_words):
  arr = []
  for row in data.values:
    d = dict()
    for key in spam_words:
      if key in row and key not in ham_words.keys(): d[key] = 1
      else: d[key] = 0
    arr.append(list(d.values()))

  return arr


# Main driver function for program
def main():
    # Data Preprocessing
    df, data_df, target_df, top_50_ham_word_counts, top_50_spam_word_counts = data_preprocessing()

    ########## Evaluation of BC and NN methods ##########

    ##### Bayesian Classifier section #####

    # First run on entire dataset
    print("First Bayesian Classifier run on entire dataset")
    first_correct_predictions = BayesClassifier(0.0, df, data_df, target_df)

    # Second run with 80% training data
    print("And this is the Bayesian Classifier with a 80/20 training/test data split (please wait a few seconds):")
    third_correct_predictions = BayesClassifier(0.20, df, data_df, target_df)


    # Third run with 70% training data
    print("This is the Bayesian Classifier with 70/30 training/test data split (please wait a few seconds):")
    second_correct_predictions = BayesClassifier(0.30, df, data_df, target_df)


    performance = []
    performance.append(first_correct_predictions)
    performance.append(second_correct_predictions)
    performance.append(third_correct_predictions)

    # Setup for bar chart
    plt.bar(numpy.arange(3), performance, align='center', alpha=0.5)
    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(numpy.arange(3), ["100%", "80%", "70%"])
    plt.ylabel("Predication Accuracy (%)")
    plt.xlabel("(%) of dataset used for training")
    plt.title("Bayesian Classifier Accuracy by Training Data Given")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2%}'))
    plt.ylim(0.82,0.88)
    

    # Uncomment the plt.show() to pop up figure in new window instead of saving
    # Note that using show() and savefig() simultaneously will not work
    #plt.show()
    plt.savefig('Bayes_Results.png')
    plt.close()
    
    ##### Neural Network #####

    # First on whole dataset
    print("\n\nThis is the NN without a Train/Test Split")
    first_spam_preds, first_ham_preds = Part_Three_NNs(data_df, target_df, top_50_spam_word_counts, top_50_ham_word_counts)

    # Then with training data
    #80%/20% Split
    print("\n\nThis is the NN with a 80/20 Train/Test Split")
    second_spam_preds, second_ham_preds = Part_Three_NNs(data_df, target_df, top_50_spam_word_counts, top_50_ham_word_counts, True, 0.20)

    #70%/30% Split
    print("\n\nThis is the NN with a 70/30 Train/Test Split")
    third_spam_preds, third_ham_preds = Part_Three_NNs(data_df, target_df, top_50_spam_word_counts, top_50_ham_word_counts, True, 0.30)


    performance = []
    performance.append(first_spam_preds)
    performance.append(second_spam_preds)
    performance.append(third_spam_preds)

    # Setup for bar chart
    plt.bar(numpy.arange(3), performance, align='center', alpha=0.5)
    plt.xticks(numpy.arange(3), ["100%", "80%", "70%"])
    plt.ylabel("Accuracy (%)")
    plt.xlabel("% Training data given")
    plt.title("Neural Network Accuracy by Training Data Given")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2%}'))
    plt.ylim(0.82,0.88)

    #plt.show()
    plt.savefig('NN_Results.png')

main()  # Call to the main driver function above