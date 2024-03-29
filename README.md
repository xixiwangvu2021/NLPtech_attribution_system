# NLP_Technology(RM)_tutorial_report_Processing_Attribution

This repository contains code and auxiliary materials for NLP Technology(RM) tutorial report **Processing Attribution** by Betty van Dongen (e.c.van.dongen@student.vu.nl), Fina Polat (f.e.yilmazpolat@student.vu.nl) and Xixi Wang (x21.wang@student.vu.nl) in June and July 2021.

The report **Processing Attribution** introduces a complete pipeline of processing attribution relations (ARs). The pipeline has integrated the Baseline_CRF,  Embedding_CRF and Features2_CRF. For the first two models, the pipeline consists of Pre-processing, Component extraction, AR linking, Post-processing and Evaluation; for the third model, the pipeline includes a Feature extraction before Component extraction.

Through Pre-processing, we can the data set from conll files to 'neat' csv files. The Feature extraction allows more features to be extracted for the use of the feature model. The component extraction is to parallel extract the cues, sources and contents with BIO format. Through AR linking, we have linked the three components together. Post-processing is to transfer the output of AR linking into conll files to be prepared for the Evaluation. In the Evaluation process, evaluation.attribution.v5.pl need to be used for the metrics for the full match of sources and contents, and the partial match of sources, contents, cues and ARs.

<img src = "images/pipeline_NLPtutorial_V1.png">
</p>
<p align = "center">
Figure 1: The system pipeline
</p>

Although our results aren't comparable with the state-of-the-art performance, yet we've successfully addressed a complete integrated AR pipeline which brings about the possibility of parallel extracting components, and on the other hand, serves as an illustrative reference for similar future work.

## Notebooks

The notebooks in CRF folder are the modules of the integrated pipeline.

The notebooks in the folder of CRF with Embeddings are the experimenting notebooks with word embeddings.

## Running the code

1. All code is in Python 3.

Install requirements are included in the notebooks.

2. Usage

In order to run the pipeline notebooks, we need to 1) add unzipped data folders to the main directory, 2) under the CRF folder, add empty folders 'Preprocessed_data' and 'CONLL' and add word embedding pretrained model 'glove.6B.50d.txt', then 3) under the 'CONLL' folder, add the evaluation scripts (evaluation.attribution.v5.pl) and two folders each contain the development data from each data source, and add an empty folder for the evaluation output files.

How to run the pipeline from the beginning to the end is explained in the following videos:

video1 https://drive.google.com/file/d/1SDgBw7m0SQnSx3prNUS0KGb_cHJuRU3u/view?usp=sharing

video2 https://drive.google.com/file/d/1q2ET9FFW16Bq2j99r8U0OKTzI7mxiqpf/view?usp=sharing

video3 https://drive.google.com/file/d/1jUjo6G4SVKb3uLwnVvWPqZaqg7TlAzQh/view?usp=sharing

3. After running through all the notebooks and evaluation scripts, the following results should be given:

![](images/Baseline_1_3.png)
![](images/Baseline_4_6.png)




![](images/Embedding_1_3.png)
![](images/Embedding_4_6.png)




![](images/Features2_1_3.png)
![](images/Features2_4_6.png)


