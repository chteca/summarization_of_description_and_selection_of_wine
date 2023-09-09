This project solves the problem of recommending a wine according to its description and consists of 2 parts: 
1. Summarization of the wine description.
2. Selection of the top 5 most delicious wines based on a summary.

To summarize the description, an extractive approach is used using the textrank algorithm.
A Doc2Vec model was trained to find the most similar wines.


**To get a summary, run the file summary_main.py.**

**To get the top 5 wines by description, run the file top_5_main.py**
