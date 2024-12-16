# Final Project - Data Processing: Recipe Dataset Analysis
This repository contains the final project for the Data Processing course by Álvaro Fernández Gómez and Rubén Fernández Rodrigo. The project aims to analyze a dataset consisting of 20,130 entries corresponding to recipes from the Epicurious website (www.epicurious.com). The project is divided into four key steps:

# 1. Analysis of Input Variables and Text Preprocessing
The first step involves performing an exploratory analysis of the dataset's input variables, along with preprocessing of the text data. We will generate visualizations to examine the relationships between the numerical variables and their characteristics. These include:
•	Correlation between numerical variables
•	The relationship between ratings and other variables
•	Top ten most common categories
•	Best-rated categories
Additionally, all text variables will be concatenated into a single column, referred to as 'text,' to facilitate further processing.

# 2. Vectorization of Text Data
In this step, we will apply three different text vectorization techniques to the 'text' column:
•	TF-IDF (Term Frequency-Inverse Document Frequency): This method does not account for the semantics of words but focuses on word frequency and inverse document frequency.
•	Word2Vec: This technique utilizes word embeddings to represent words in a continuous vector space, capturing semantic relationships between words.
•	BERT (Bidirectional Encoder Representations from Transformers): This method uses contextual embeddings, allowing the model to account for the surrounding context of words.

# 3. Model Training and Evaluation
We train two machine learning models using the scikit-learn library: Random Forest and Support Vector Machine (SVM), applying the different vectorization techniques to evaluate their performance.
Furthermore, we define and train two neural network models based on Long Short-Term Memory (LSTM) architecture. The first model uses the concatenated text of all five variables, while the second model treats each text variable separately as distinct inputs. Both models are trained using the Adam optimizer and Mean Squared Error (MSE) loss function. Hyperparameters such as the number of epochs, learning rate, and batch size have been tuned to optimize performance.
The neural networks are trained using word2vec and BERT embeddings, with the model using separated text columns being trained only with word2vec data.
To assess the performance of the models, we calculate the Mean Squared Error (MSE) and R² score for both training and test datasets. Additionally, we fine-tune a Hugging Face model (bert-base-uncased) using the default configuration and adapt it for our specific task.

# 4. Comparison of Results
In this section, we compare the performance of the different techniques and models. The comparison is based on the performance metrics (MSE and R²) obtained using various vectorization methods, machine learning algorithms, and neural network architectures.

# 5. Extension
As an extension of the project, we summarize the data in the 'directions' column of the dataset using the T5-small model. This step ensures the text data remains manageable and avoids excessively long execution times.



