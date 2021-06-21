# Based on https://github.com/cltl/ba-text-mining/blob/master/lab_sessions/lab4/Lab4a.4-NERC-CRF-Dutch.ipynb

import csv
import sklearn_crfsuite
import sys
# from ..CRF import train_and_run_crf_model


def token2features(sentence, i):

    nr_in_sentence = sentence[i][3]
    lemma = sentence[i][6]
    postag = sentence[i][7]
    dep_label = sentence[i][8]

    prev_prev_lemma, prev_lemma, next_next_lemma, next_lemma = '', '', '', ''
    if i - 2 >= 0:
        prev_prev_lemma = sentence[i-2][7]
    if i - 1 >= 0:
        prev_lemma = sentence[i-1][7]
    if i + 2 < len(sentence):
        next_next_lemma = sentence[i+2][7]
    if i + 1 < len(sentence):
        next_lemma = sentence[i+1][7]

    # in_quote =
    # after_colon =
    # constituent path
    # dependency path =

    # # Schermafbeelding 2021-06-20 om 00.03.42
    # features = {
    #     'bias': 1.0,
    #     'token': lemma.lower(),
    #     'postag': postag,
    #     'dep_label': dep_label,
    #     'nr_in_sentence': nr_in_sentence,
    # }

    # Schermafbeelding 2021-06-20 om 00.08.42
    features = {
        'bias': 1.0,
        'token': lemma.lower(),
        'prev_prev_lemma': prev_prev_lemma,
        'prev_lemma': prev_lemma,
        'next_next_lemma': next_next_lemma,
        'next_lemma': next_lemma,
        'postag': postag,
        'dep_label': dep_label,
        'nr_in_sentence': nr_in_sentence,
    }
    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) - 1:
        features['EOS'] = True

    return features
# def sent2features_range(sents, i):
#     start_i = max(i - 1, 0)
#     end_i = min(i + 1, len(sents) - 1)
#     sents_range = [token for sent in sents[start_i:end_i] for token in sent]
#     return [token2features(sents, j) for j in range(len(sents_range))]


def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    # if you added features to your input file, make sure to add them here as well.
    # return [label for token, label in sent]
    return [label for article_name, sentence_nr, nr_in_file, nr_in_sentence, fromto,
            word, lemma, postag, dep_label, token_dep_head, label in sent]


# def sent2tokens(sent):
#     # return [token for token, label in sent]
#     return [token for article_name, sentence_nr, nr_in_file, nr_in_sentence, fromto,
#             token, lemma, postag, dep_label, token_dep_head, ar_label in sent]


def extract_sents_from_conll(inputfile):

    csvinput = open(f'CRF/Preprocessed_data/{inputfile}', 'r')
    csvreader = csv.reader(csvinput, delimiter='\t')
    # First consume header row
    headers = next(csvreader)
    sents = []
    current_sent = []
    for row in csvreader:
        current_sent.append(tuple(row))
        # note that this is a simplification that works well for this particular data, in other situations,
        # you may need to do more advanced preprocessing to identify sentence boundaries
        if row[0] == "":
            sents.append(current_sent)
            current_sent = []

    # Close file
    csvinput.close()

    return sents  # header is sliced


def train_crf_model(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    return crf


def create_crf_model(trainingfile):

    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    # X_train = [sent2features_range(train_sents, i) for i in range(len(train_sents))]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)

    return crf


def run_crf_model(crf, evaluationfile):

    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)

    return y_pred, test_sents


def write_out_evaluation(eval_data, pred_labels, outputfile):

    outfile = open(f'CRF/Features_system/Result/{outputfile}', 'w')
    headers = ["Article_Name", "Sentence_nr", "Nr_in_file", "Nr_in_sentence", "FromTo", "Word", "Lemma", "POS",
               "Dep_label", "Token_dep_head", "AR_label"]
    # headers = ['Word', 'pred_AR_label']
    header_row = '\t'.join(headers) + '\n'

    outfile.write(header_row)

    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            # Data: from tuple to string separated string, except for the gold label from the original file
            data_tsv = '\t'.join(list(data)[:-1])
            outfile.write(data_tsv + "\t" + pred + "\n")

    # Close file
    outfile.close()


def train_and_run_crf_model(trainingfile, evaluationfile, outputfile):

    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)


def main():

    args = sys.argv
    trainingfile = args[1]
    evaluationfile = args[2]
    outputfile = args[3]

    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)


if __name__ == '__main__':
    main()
