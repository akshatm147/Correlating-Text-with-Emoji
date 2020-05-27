from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import emoji

def read_glove_vecs(glove_file):
    words, word_to_vec_map = set(), {}
    with open(glove_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            words.add(line[0])
            word_to_vec_map[line[0]] = np.array(line[1:], dtype=np.float64)
        
        i, words_to_index = 1, {}
        for w in sorted(words):
            words_to_index[w] = i
            i = i + 1
    return words_to_index, word_to_vec_map

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename):
    tweet, emoji = [], []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            tweet.append(row[0])
            emoji.append(row[1])

    return np.asarray(tweet), np.asarray(emoji, dtype=int)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


emoji_dictionary = {"0": ":heart:",
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:",
                    "5": ":smiling_face_with_heart-eyes:",
                    "6": ":face_with_tears_of_joy:",
                    "7": ":two_hearts:",
                    "8": ":fire:",
                    "9": ":smiling_face_with_smiling_eyes:",
                    "10": ":smiling_face_with_sunglasses:",
                    "11": ":sparkles:",
                    "12": ":blue_heart:",
                    "13": ":face_blowing_a_kiss:",
                    "14": ":camera:",
                    "15": ":United_States:",
                    "16": ":sun:",
                    "17": ":purple_heart:",
                    "18": ":winking_face:",
                    "19": ":hundred_points:",
                    "20": ":beaming_face_with_smiling_eyes:",
                    "21": ":Christmas_tree:",
                    "22": ":camera_with_flash:",
                    "23": ":winking_face_with_tongue:"}

def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
              
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    tick_marks = np.arange(len(df_confusion.columns))
    plt.colorbar()
    plt.yticks(tick_marks, df_confusion.index)
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.xlabel(df_confusion.columns.name)
    plt.ylabel(df_confusion.index.name)
    
    
def predict(X, Y, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):
        words, avg = X[j].lower().split(), np.zeros((50,))

        for w in words:
            try:
                avg += word_to_vec_map[w]
            except:
                continue
        avg = avg/len(words)

        Z = W.dot(avg) + b
        pred[j] = np.argmax(softmax(Z))
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred

def predict1(X, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):
        words, avg = X[j].lower().split(), np.zeros((50,))

        for w in words:
            avg += word_to_vec_map[w]
        avg = avg/len(words)

        Z = W.dot(avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
        print(X[j], pred[j])
    
    return pred