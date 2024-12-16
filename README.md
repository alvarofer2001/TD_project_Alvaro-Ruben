# TD_project_Alvaro-Ruben
Data Processing final project of Álvaro Fernández Gómez and Rubén Fernández Rodrigo

The project objective is to analyze a datasetthat consists of 20130 entries corresponding to recipes from the www.epicurious.com website.
The project will be divided in 4 steps:

# 1) Analysis of input variables and text preprocessing.
First we will plot some graphs that show the relationship between the numerical variables and some of their characteristics.
Some of these graphs are:
  -Correlation between variables
  -Rating relation with the variables
  -Top ten more common cattegories
  -Best rated cattegories

We will also concatenate the text variables into one. We will call it 'text' 

# 2) Vectorization of the variables
We will use three different type of vectorizations:
  -TF-IDF (Term Frequency-Inverse Document Frequency): It does not consider the semantics of the words
  -word2vec: uses word embbeding
  -BERT (Bidirectional Encoder Representations from Transformers): uses contextual embbeding
  
# 3) Training and evaluation of the model
In this project, we will use two different neural networks. In both we will define an LSTM-based model. In the first neural network the input will be the concatenated text of the five varibles and in the second one the five text variables will be separated (five text columns).
The model is trained using an Adam optimizer and Mean Squared Error (MSE) loss. We define the following parameters for training:
  -num_epochs: Number of epochs for training.
  -learning_rate: The learning rate for optimization.
  -batch_size: The batch size for training and evaluation.
  
To evaluate the model's performance, we calculate the Mean Squared Error (MSE) on both training and test datasets. We will also calculate the R^2 score.

# 4) Comparison of the results
In this final section we will compare the measures and results obtained using the different techniques and strategies




