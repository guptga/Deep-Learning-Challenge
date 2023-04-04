# Deep-Learning-Challenge 
## Alphabet Soup Charity Case Study

![image](https://miro.medium.com/v2/resize:fit:786/format:webp/1*pj58UJEa3eRyf-3c502N6w.jpeg)


## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


### Overview of the analysis: Explain the purpose of this analysis.

**Step 1:** 

* First, loaded the charity_data.csv file into a Pandas DataFrame and removed any unnecessary data that would not contribute to our analysis. 
* Next,  determined the target variable for the model and identified the features that would work best for the model. To do this, look at the number of unique values and the number of data points for each unique value, and used the latter to establish a cutoff point for binning rare categorical variables together into a new value called "Other." 
* Then confirmed the success of this binning using a check.
* To encode the categorical variables,  utilized the pd.get_dummies() function. 
* Split the preprocessed data into a features array, X, and a target array, y, and used the train_test_split function to divide the data into training and testing datasets.
* Finally, scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, and then applying the transform function.

**Step 2:**

* a. With my knowledge in Google Colab and TensorFlow, developed a neural network or deep learning model capable of binary classification. The purpose of the model was to predict the success of an Alphabet Soup-funded organization based on the features in the dataset. To ensure optimal performance, I considered the number of inputs before determining the appropriate number of neurons and layers for the model.

* b. Next, I compiled, trained, and evaluated the binary classification model, taking into account the model's loss and accuracy. This involved creating a neural network model using TensorFlow and Keras and assigning the number of input features and nodes for each layer. Additionally, I designed the first hidden layer, selecting an appropriate activation function, and determining if a second hidden layer was necessary. Finally, I created an output layer with an appropriate activation function.


**Step 3:**, I optimized the model's performance, aiming to achieve a predictive accuracy target of more than 75%.

To achieve this goal, I made several adjustments to the input data to ensure that no variables or outliers were causing confusion in the model. These included dropping more or fewer columns, creating additional bins for rare occurrences in columns, adjusting the number of values for each bin, adding more neurons to a hidden layer, incorporating additional hidden layers, experimenting with different activation functions for the hidden layers, and increasing or decreasing the number of epochs in the training regimen.


### Results

#### Data Preprocessing

After preprocessing the dataset by dropping unnecessary columns and encoding categorical variables:

*__Question: What variable(s) are the target(s) for your model?__* 
* The IS_SUCCESSFUL variable serves as a binary classifier with two potential outcomes: either the decision to provide funding for an organization is successful or not.


*__Question: What variable(s) are the features for your model?__*
* After examining the unique value counts for each column, we decided to narrow down the available features for our model. We focused on those columns that had more than 10 unique values, namely NAME, APPLICATION_TYPE, and CLASSIFICATION. These features possess ideal qualitative properties for training the model.


*__Question: What variable(s) should be removed from the input data because they are neither targets nor features?__*
* During the preprocessing phase of our analysis, we excluded the column labelled EIN since it only contained organizational IDs and had little value for predictive analysis.
* ![image](https://user-images.githubusercontent.com/116124534/229928292-64f9a627-fcbe-4b24-a2c8-7987cc74eaf2.png)

#### Compiling, Training, and Evaluating the Model

*__Question: How many neurons, layers, and activation functions did you select for your neural network model, and why?__*
* Features: APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: Two (with 10 and 5 neurons, respectively)
* Activation Functions:The Rectified Linear Unit (ReLU) activation function was used in the first and second hidden layers of the neural network, which has become the preferred choice for many neural network models due to its ease of training and better performance. The Sigmoid activation function was chosen for the output layer, allowing the neural network to learn non-linearly separable problems by adding just one hidden layer with a Sigmoid activation function. Using a non-linear function produces non-linear boundaries, making Sigmoid well-suited for learning complex decision functions and binary classification tasks in neural networks.
* Trainable Params: 501

*__Question: Were you able to achieve the target model performance?__*
In neural networks, the loss value indicates the model's performance after each optimization iteration, with the aim of minimizing the error. Accuracy, on the other hand, measures how well the model predicts the true values in terms of percentage. After running 20 epochs, the accuracy was went from 55% to 73%, falling short of the target performance.

![image](https://user-images.githubusercontent.com/116124534/229932438-187d7e9b-9009-4a17-b73b-40bc93c5f6d3.png)

![image](https://user-images.githubusercontent.com/116124534/229932647-f8ac70ad-de0b-4817-b61b-44608fe9d6eb.png)


*__Question: What steps did you take in your attempts to increase model performance?__*

Optimizing Model

* Features used: NAME, APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: 3 with 10, 8 and 6 neurons
* Activation Functions: ReLU - first hidden layer, Sigmoid- 2nd, 3rd and outer layer
* Trainable Parameters: 2,899
* Epochs: 20
* Loss: 45%
* Accuracy: 78%
* Target reached: Yes

![image](https://user-images.githubusercontent.com/116124534/229933425-9a6aa744-6552-4288-a682-c5ac043a39e6.png)

![image](https://user-images.githubusercontent.com/116124534/229933552-749ee34c-d37b-49fb-b393-d9095dd83c2e.png)

We achieved a 78% predictive accuracy in classifying the success of organizations funded by Alphabet Soup Charity by iterating through different models and making various optimization attempts. They learned that choosing the right weights and optimization algorithm is very important.
