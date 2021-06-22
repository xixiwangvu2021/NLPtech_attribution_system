# Based on https://github.com/cltl/ba-text-mining/blob/master/lab_sessions/lab4/Lab4a.4-NERC-CRF-Dutch.ipynb

import csv
import numpy as np # to create 0 vectors for the words which are not in the vocabulary
import sklearn_crfsuite
import sys

from gensim.models import KeyedVectors  # to load pre-trained word embeddings
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


class CRF(object):
    def token2features(self, sentence, i):
        # print(sentence)

        nr_in_sentence = sentence[i][3]
        word = sentence[i][5]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]

        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()
        # else:
        #     print(sentence)

        prev_prev_lemma, prev_lemma, next_next_lemma, next_lemma = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_lemma = sentence[i-2][7]
        if i - 1 >= 0:
            prev_lemma = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_lemma = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_lemma = sentence[i+1][7]

        features = {
            'bias': 1.0,
            'token': word,
        }

#         # Schermafbeelding 2021-06-20 om 00.03.42
#         features = {
#             'bias': 1.0,
#             'token': lemma.lower(),
#             'postag': postag,
#             'dep_label': dep_label,
#             'nr_in_sentence': nr_in_sentence,
#         }

#         features = {
#             'bias': 1.0,
#             'token': word,
#             'word.isupper()': word_is_upper,
#             'postag': postag,
#             'postag[:2]': postag[:2],
#             'prev_prev_lemma': prev_prev_lemma,
#             'prev_lemma': prev_lemma,
#             'next_next_lemma': next_next_lemma,
#             'next_lemma': next_lemma,
#             'dep_label': dep_label,
#             'nr_in_sentence': nr_in_sentence,
#         #    'in_quote': in_quote,  # Adding this one makes no difference
#         #     'after_colon': after_colon,  # Adding this one makes no difference
#             'dependency_distance': dependency_distance,
#             'dependency_path': dependency_path,
#         }

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            # features.update({
            #     '-1:word.lower()': word1.lower(),
            #     '-1:word.isupper()': word1_is_upper,
            #     '-1:postag': postag1,
            #     '-1:postag[:2]': postag1[:2],
            # })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            # features.update({
            #     '+1:word.lower()': word1.lower(),
            #     '+1:word.isupper()': word1_is_upper,
            #     '+1:postag': postag1,
            #     '+1:postag[:2]': postag1[:2],
            # })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.token2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        # if you added features to your input file, make sure to add them here as well.
        # return [label for token, label in sent]
        return [label for article_name, sentence_nr, nr_in_file, nr_in_sentence, fromto, word, lemma, postag, dep_label,
                          token_dep_head, label, in_quote, after_colon, dep_distance, dep_path in sent]

    def extract_sents_from_conll(self, inputfile):

        # csvinput = open(f'CRF/Preprocessed_data/{inputfile}', 'r')
        csvinput = open(f'Preprocessed_data/{inputfile}', 'r', encoding="utf-8")
        csvreader = csv.reader(csvinput, delimiter='\t')
        # First consume header row
        headers = next(csvreader)
        sents = []
        current_sent = []
        for i, row in enumerate(csvreader):
            # note that this is a simplification that works well for this particular data, in other situations,
            # you may need to do more advanced preprocessing to identify sentence boundaries
            if row[3] == "1" and row[2] != "1":
                sents.append(current_sent)
                current_sent = []
            current_sent.append(tuple(row))

        # Add last row of file
        sents.append(current_sent)

        # Close file
        csvinput.close()

        return sents  # header is sliced

    # def extract_sents_from_conll(inputfile):
    #     '''Read the data from tsv file, return sentences as tokens with corresponding labels.'''
    #
    #     rows = csv.reader(open(inputfile, encoding="utf-8"), delimiter='\t')
    #     sents = []
    #     current_sent = []
    #     for row in rows:
    #         current_sent.append(tuple(row))
    #         # After each sentence there is a special token: Sent_end. Its label is O. It was added in the preprocessing step.
    #         if row[0] == "Sent_end":
    #             sents.append(current_sent)
    #             current_sent = []
    #
    #     return sents

    def train_crf_model(self, X_train, y_train):

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        return crf

    def create_crf_model(self, trainingfile):
        
        train_sents = self.extract_sents_from_conll(trainingfile)
        X_train = [self.sent2features(s) for s in train_sents]
        y_train = [self.sent2labels(s) for s in train_sents]

        crf = self.train_crf_model(X_train, y_train)

        return crf

    def run_crf_model(self, crf, evaluationfile):
        
        test_sents = self.extract_sents_from_conll(evaluationfile)
        X_test = [self.sent2features(s) for s in test_sents]
        y_test = [self.sent2labels(s) for s in test_sents]
        y_pred = crf.predict(X_test)

        return y_pred, test_sents, y_test

    def write_out_evaluation(self, eval_data, pred_labels, outputfile):
        
        outfile = open(f'Result/{outputfile}', 'w')
        headers = ["Article_Name", "Sentence_nr", "Nr_in_file", "Nr_in_sentence", "FromTo", "Word", "Lemma", "POS",
                   "Dep_label", "Token_dep_head", "AR_label"]
        # headers = ['Word', 'pred_AR_label']
        header_row = '\t'.join(headers) + '\n'

        outfile.write(header_row)

        for evalsents, predsents in zip(eval_data, pred_labels):
            for data, pred in zip(evalsents, predsents):
                # Data: from tuple to string separated string, except for the gold label from the original file
                data_tsv = '\t'.join(list(data)[:10])
                outfile.write(data_tsv + "\t" + pred + "\n")
                # Why last line not written?

        # Close file
        outfile.close()
 
    def write_out_evaluation(self, eval_data, pred_labels, outputfile):
        
        outfile = open(f'Result/{outputfile}', 'w')
        headers = ["Article_Name", "Sentence_nr", "Nr_in_file", "Nr_in_sentence", "FromTo", "Word", "Lemma", "POS",
                   "Dep_label", "Token_dep_head", "AR_label"]
        # headers = ['Word', 'pred_AR_label']
        header_row = '\t'.join(headers) + '\n'

        outfile.write(header_row)

        for evalsents, predsents in zip(eval_data, pred_labels):
            for data, pred in zip(evalsents, predsents):
                # Data: from tuple to string separated string, except for the gold label from the original file
                data_tsv = '\t'.join(list(data)[:10])
                outfile.write(data_tsv + "\t" + pred + "\n")

        # Close file
        outfile.close()
        
    def write_out_evaluation_diff(self, eval_data, pred_labels, outputfile):
        
        outfile = open(f'Result/Diff/{outputfile}', 'w')
        headers = ["Article_Name", "Sentence_nr", "Nr_in_file", "Nr_in_sentence", "FromTo", "Word", "Lemma", "POS",
                   "Dep_label", "Token_dep_head", "AR_label", "Pred_AR_label"]
        header_row = '\t'.join(headers) + '\n'

        outfile.write(header_row)

        for evalsents, pred_labels in zip(eval_data, pred_labels):
            for data, pred_label in zip(evalsents, pred_labels):
                eval_label = data[10]
                if eval_label != pred_label:
                    # Data: from tuple to string separated string, except for the gold label from the original file
                    data_tsv = '\t'.join(list(data[:11]))
                    outfile.write(data_tsv + "\t" + pred_label + "\n")

        # Close file
        outfile.close()
        
    def report_evaluation(self, labels, y_test, y_pred, outputfile):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        
        outputfile = outputfile.replace('.csv', '.txt')

        with open(f'Result/Evaluation/{outputfile}', 'w') as f:
            sys.stdout = f    
                  
            print('The predictions are written on the output file.')
            print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=4))
            print('Accuracy score for sequence items')
            print(metrics.flat_accuracy_score(y_test, y_pred))
            print('Precision score for sequence items')
            print(metrics.flat_precision_score(y_test, y_pred, average='weighted'))
            print('Recall score for sequence items')
            print(metrics.flat_recall_score(y_test, y_pred, average='weighted'))
            print('F1 score score for sequence items')
            print(metrics.flat_f1_score(y_test, y_pred, average='weighted'))
                  
        sys.stdout = original_stdout  # Reset the standard output to its original value

    def train_and_run_crf_model(self, trainingfile, evaluationfile, outputfile):

        crf = self.create_crf_model(trainingfile)
        labels = list(crf.classes_)
        labels.remove('O')
#         labels.remove('AR_label')
        print(labels)
        y_pred, eval_sents, y_test = self.run_crf_model(crf, evaluationfile)
        self.write_out_evaluation(eval_sents, y_pred, outputfile)
        self.write_out_evaluation_diff(eval_sents, y_pred, outputfile)
        self.report_evaluation(labels, y_test, y_pred, outputfile)

        
class FeaturesCRF(CRF):
    def token2features(self, sentence, i):
        # print(sentence)

        nr_in_sentence = sentence[i][3]
        word = sentence[i][5]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]

        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()
        # else:
        #     print(sentence)

        prev_prev_lemma, prev_lemma, next_next_lemma, next_lemma = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_lemma = sentence[i-2][7]
        if i - 1 >= 0:
            prev_lemma = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_lemma = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_lemma = sentence[i+1][7]

        # # Schermafbeelding 2021-06-20 om 00.03.42
        # features = {
        #     'bias': 1.0,
        #     'token': lemma.lower(),
        #     'postag': postag,
        #     'dep_label': dep_label,
        #     'nr_in_sentence': nr_in_sentence,
        # }

        features = {
            'bias': 1.0,
            'token': word,
            'word.isupper()': word_is_upper,
            'postag': postag,
            'postag[:2]': postag[:2],
            'prev_prev_lemma': prev_prev_lemma,
            'prev_lemma': prev_lemma,
            'next_next_lemma': next_next_lemma,
            'next_lemma': next_lemma,
            'dep_label': dep_label,
            'nr_in_sentence': nr_in_sentence,
            'in_quote': in_quote,
            'after_colon': after_colon,
            'dependency_distance': dependency_distance,
            'dependency_path': dependency_path,
        }

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            # features.update({
            #     '-1:word.lower()': word1.lower(),
            #     '-1:word.isupper()': word1_is_upper,
            #     '-1:postag': postag1,
            #     '-1:postag[:2]': postag1[:2],
            # })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word.isupper()

            # features.update({
            #     '+1:word.lower()': word1.lower(),
            #     '+1:word.isupper()': word1_is_upper,
            #     '+1:postag': postag1,
            #     '+1:postag[:2]': postag1[:2],
            # })
        else:
            features['EOS'] = True

        return features


class Features2CRF(CRF):
    def token2features(self, sentence, i):
        nr_in_sentence = sentence[i][3]
        word = sentence[i][5]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]

        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()

        prev_prev_postag, prev_postag, next_next_postag, next_postag = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_postag = sentence[i-2][7]
        if i - 1 >= 0:
            prev_postag = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_postag = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_postag = sentence[i+1][7]

        features = {
            'bias': 1.0,
            'token': word,
            'word.isupper()': word_is_upper,
            'postag': postag,
            'postag[:2]': postag[:2],
            'prev_prev_postag': prev_prev_postag,
            'prev_postag': prev_postag,
            'next_next_postag': next_next_postag,
            'next_postag': next_postag,
            'dep_label': dep_label,
            'nr_in_sentence': nr_in_sentence,
            'in_quote': in_quote,
            'after_colon': after_colon,
            'dependency_distance': dependency_distance,
            'dependency_path': dependency_path,
        }

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isupper()': word1_is_upper,
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isupper()': word1_is_upper,
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    
class Features3CRF(CRF):
    def token2features(self, sentence, i):

        nr_in_sentence = sentence[i][3]
        word = sentence[i][5]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]

        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()

        prev_prev_postag, prev_postag, next_next_postag, next_postag = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_postag = sentence[i-2][7]
        if i - 1 >= 0:
            prev_postag = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_postag = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_postag = sentence[i+1][7]

        features = {
            'bias': 1.0,
            'token': word,
            'word.isupper()': word_is_upper,
            'postag': postag,
#             'postag[:2]': postag[:2],
#             'prev_prev_postag': prev_prev_postag,
#             'prev_postag': prev_postag,
#             'next_next_postag': next_next_postag,
#             'next_postag': next_postag,
            'dep_label': dep_label,
            'nr_in_sentence': nr_in_sentence,
#             'in_quote': in_quote,
#             'after_colon': after_colon,
            'dependency_distance': dependency_distance,
            'dependency_path': dependency_path,
        }

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()
                
            word2 = sentence[i - 2][5]
            postag2 = sentence[i - 2][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isupper()': word1_is_upper,
                '-1:postag': postag1,
#                 '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i > 1:
            word2 = sentence[i - 2][5]
            postag2 = sentence[i - 2][7]

            word2_is_upper = ''
            if word2:
                word2 = word2.lower()
                word2_is_upper = word2.isupper()

            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.isupper()': word2_is_upper,
                '-2:postag': postag2,
#                 '-2:postag[:2]': postag2[:2],
            })
 
        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isupper()': word1_is_upper,
                '+1:postag': postag1,
#                 '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        if i < len(sentence) - 2:
            word2 = sentence[i + 2][5]
            postag2 = sentence[i + 2][7]

            word2_is_upper = ''
            if word2:
                word2 = word2.lower()
                word2_is_upper = word2.isupper()

            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.isupper()': word2_is_upper,
                '+2:postag': postag2,
#                 '+2:postag[:2]': postag2[:2],
            })

        return features
    

class Features4CRF(CRF):
    cue_lemmas = ['say', 'be', 'to', 'have', 'tell', 'call', 'write', 'accord', 'add', 'ask', 'show', 'support', 'note', 
                  'report', 'suggest', 'argue', 'expect', 'report', 'believe', 'agree', 'think', 'announce', 'cite', 'suggest']
    
    def token2features(self, sentence, i):
        nr_in_sentence = sentence[i][3]
        word = sentence[i][5]
        lemma = sentence[i][6]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]
        is_cue_lemma = (lemma in self.cue_lemmas)
        
        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()

        prev_prev_postag, prev_postag, next_next_postag, next_postag = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_postag = sentence[i-2][7]
        if i - 1 >= 0:
            prev_postag = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_postag = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_postag = sentence[i+1][7]

        features = {
            'bias': 1.0,
#             'token': word,
            'word.isupper()': word_is_upper,
            'postag': postag,
            'postag[:2]': postag[:2],
            'prev_prev_postag': prev_prev_postag,
            'prev_postag': prev_postag,
            'next_next_postag': next_next_postag,
            'next_postag': next_postag,
            'dep_label': dep_label,
            'nr_in_sentence': nr_in_sentence,
            'in_quote': in_quote,
            'after_colon': after_colon,
            'dependency_distance': dependency_distance,
            'dependency_path': dependency_path,
            'is_cue_lemma': is_cue_lemma,
        }

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
#                 '-1:word.lower()': word1.lower(),
                '-1:word.isupper()': word1_is_upper,
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
#                 '+1:word.lower()': word1.lower(),
                '+1:word.isupper()': word1_is_upper,
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    
class EmbeddingCRF(CRF):
    def __init__(self, nr_model_dimensions, **kwargs):
        """
        Args:
            nr_model_dimensions (string): the.
            evaluationfile (string): the.
            outputfile (string): the.
        """
        super(EmbeddingCRF, self).__init__(**kwargs)
        self.model = KeyedVectors.load_word2vec_format(f'glove.6B.{str(nr_model_dimensions)}d.w2vformat.txt')
        self.nr_model_dimensions = nr_model_dimensions
    
    ### Embedding function 
    def get_features(self, token):
        '''Get token, return word vector'''

        token = token.lower()
        try:
            vector = self.model[token]
        except:
            # if the word is not in vocabulary,
            # returns zeros array
            vector = np.zeros(self.nr_model_dimensions,)

        return vector   

    def token2features(self, sentence, i):
        '''Get tokens in the sentence, add bias, token and word embeddings as features and return all as a feature dictionary.'''

        word = sentence[i][5]
        wordembedding = self.get_features(word)  ## word embedding vector 
        wordembedding = np.array(wordembedding)  ## vectors
        
        if word:
            word = word.lower()

        features = {
            'bias': 1.0,
            'token': word
        }

        for iv,value in enumerate(wordembedding):
            features['v{}'.format(iv)] = value

        if i == 0:
            features['BOS'] = True

        elif i == len(sentence) -1:
            features['EOS'] = True
            
        return features


class FeaturesEmbeddingCRF(EmbeddingCRF):
    def token2features(self, sentence, i):
        '''Get tokens in the sentence, add bias, token and word embeddings as features and return all as a feature dictionary.'''

        nr_in_sentence = sentence[i][3]  # Normalize
        word = sentence[i][5]
        postag = sentence[i][7]
        dep_label = sentence[i][8]
        in_quote = sentence[i][11]
        after_colon = sentence[i][12]
        # constituent_path
        dependency_distance = sentence[i][13]
        dependency_path = sentence[i][14]

        word_is_upper = ''
        if word:
            word = word.lower()
            word_is_upper = word.isupper()

        prev_prev_lemma, prev_lemma, next_next_lemma, next_lemma = '', '', '', ''
        if i - 2 >= 0:
            prev_prev_lemma = sentence[i-2][7]
        if i - 1 >= 0:
            prev_lemma = sentence[i-1][7]
        if i + 2 < len(sentence):
            next_next_lemma = sentence[i+2][7]
        if i + 1 < len(sentence):
            next_lemma = sentence[i+1][7]

        features = {
            'bias': 1.0,
            'token': word,
            'word.isupper()': word_is_upper,
            'postag': postag,
            'postag[:2]': postag[:2],
            'prev_prev_lemma': prev_prev_lemma,
            'prev_lemma': prev_lemma,
            'next_next_lemma': next_next_lemma,
            'next_lemma': next_lemma,
            'dep_label': dep_label,
            'nr_in_sentence': nr_in_sentence,
#             'in_quote': in_quote,  # Made no difference
#             'after_colon': after_colon,  # Made no difference
            'dependency_distance': dependency_distance,
            'dependency_path': dependency_path,
        }

        # Add word embeddings
        wordembedding = self.get_features(word)  ## word embedding vector 
        wordembedding = np.array(wordembedding)  ## vectors

        for iv,value in enumerate(wordembedding):
            features['v{}'.format(iv)] = value

        if i > 0:
            word1 = sentence[i - 1][5]
            postag1 = sentence[i - 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isupper()': word1_is_upper,
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][5]
            postag1 = sentence[i + 1][7]

            word1_is_upper = ''
            if word1:
                word1 = word1.lower()
                word1_is_upper = word1.isupper()

            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isupper()': word1_is_upper,
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features