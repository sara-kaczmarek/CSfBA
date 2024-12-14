# CSfBA: Scalable Product Duplicate Detection: The Turbo MSMP+ Approach

This github repository contains the python code for the individual paper for the Computer Science for Business Analytics (FEM21037) course at Erasmus School of Economics. The main algorithm infrastructure was built in collaboraton with **Marten Jager**, but differ in data cleaning and key-value usage. 

---

## Overview
Below, find the descriptions of the scripts for data cleaning, improved data cleaning, Locality Sensitive Hashing, Multi-component Similarity Method, Model evaluation, and Result Plotting. All the scripts work together to process the data, and run the algorithm to get the results for the paper. 

---

## Components

### *Data Cleaning*
- The ''data_cleaning.py'' script contains functions to clean the data as proposed in previous work on MSMP+ that is used as baseline.
- The 'data_cleaning_improved.py' file contains an improved version of the cleaning procedure with additional steps and data manipulation. 

### *Main Algorithm*
- The 'main_algorithm.py' script contains all the main functions for the algorithm (generating binary vectors, generating the signature matrix (MinHashing), generating candidate duplicate (LSH), generating the dissimilarity matrix (MSM), and clustering the duplicate pairs). All supporting functions for the MSM algorithm can be found in 'msm_functions.py'.

### *Algorithm Evaluation*
- The 'evaluation.py' contains the functions necessary to get the performance results for the algorithm. These functions are then used in the plotting.ipynb notebook to plot the evaluation metrics for LSH (pair completeness, pair quality, and F1* measure), and MSM (F1 measure) with helper functions in 'functions.py' to open the JSON file 'TVs-all-merged.json' and bootstrap the dataset. 

