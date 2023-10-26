# Projet_kaggle

Authors:

**BOULAYAT Meryam** - **GOU Melanie** - **MECHOUEK Lounes** - **COHEN Chlomite** - **MOHAMED Shahina**

Master 2 Bio-informatics at *Univerité de Paris*.

##  Path to documents 
- Report: ``a
- Oral presentation: ``

## Context

> This project is actually trying to answer to this Kaggle project: [**Child Mind Institute - Detect Sleep States**](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states).

Sleep is a fundamental cornerstone of human existence, deeply influencing a range of functions from cognitive processing to emotional well-being. Yet, unraveling the intricate patterns of sleep remains a daunting task, primarily due to a scarcity of naturally occurring, precisely annotated data.

In our project, we harness a dataset composed of nearly 500 multi-day accelerometer recordings from wrist-worn devices to decipher an individual's sleep state. Our approach has bifurcated into two main paths. On one end, we've leveraged a traditional machine learning technique known as the Random Forest. Conversely, we've ventured into neural network architectures, experimenting with models like LSTM, hybrid CNN-LSTM, and GRU.

For those interested in the mechanics, the "src" directory houses scripts centered on data preprocessing and other related utilities that were instrumental in curating our datasets.

Our finished models, representing the culmination of our research efforts, are neatly archived in the "model" directory.


## Implemented Methods for Kaggle Project

In our group's effort to address the challenges of the Kaggle project, we've thoroughly explored and implemented a range of deep learning techniques. Here's a brief overview of the methods we've employed:

- LSTM (Long Short-Term Memory): A type of recurrent neural network architecture, LSTM has the ability to remember patterns over long durations, making it particularly suitable for time-series and sequence prediction tasks.

- GRU (Gated Recurrent Units): A variant of the traditional RNN, GRU's are designed to capture short-term dependencies in data, offering computational efficiency and mitigating the vanishing gradient problem.

- LSTM+CNN: In this hybrid approach, we've combined the sequence learning capabilities of LSTM with the spatial feature extraction prowess of Convolutional Neural Networks (CNN). This amalgamation aims to capture both temporal and spatial patterns in data effectively.

- Bi-LSTM (Bidirectional LSTM): By processing data from both past-to-future and future-to-past directions, Bi-LSTM provides a richer representation and often yields improved performance, especially when the context from both directions is crucial.



