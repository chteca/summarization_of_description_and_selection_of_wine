# Wine recommendation system by description

This project solves the problem of recommending a wine according to its description and consists of 2 parts: 
1. Summarization of the wine description.
2. Selection of the top 5 most similar wines based on a summary.

To summarize the description, an extractive approach with textrank algorithm is used.
A Doc2Vec model was trained to find the most similar wines.

### Use of the program
To run the program, you can clone the project to your local computer. Next, on the command line, go to the directory where the project is located and run:
1. to get a summary
  ```
   python summary_main.py
  ```
2. to obtain the most suitable wines for the description
  ```
   python top_5_main.py
   ```
   
