# Housing-Price-Prediction--Machine-Learning
Different Machine Learning techniques which are already exists in the market for predicting Housing Prices. I have studied multiple papers and research work done in this field and then tried and implemented various algorithms to achieve maximum accuracy in predicting the Housing Prices. I used MSE(Mean Square Error) to predict the accuracy of different models used and drew a conclusion that applying Random Forest algorithm works best for predicting Housing Prices.
You can import the dataset from sklearn or even upload from the same dataset that I have provided in the Dataset Folder.


## Objective
I am doing an empirical valuation, considering California as the primary location, and trying to predict property prices for different locations based on latitude and longitude. As described in a data set obtained from a website called scikit learning, using various parameters such as average number of rooms, population, average occupancy, and average house price. I tried to use different algorithms, each weighted by a percentage of accuracy.
I used three models Linear Regression, Decision Tree, Random Forest. To evaluate the accuracy of each model I use Mean Absolute Error and Mean square Error (MSE).
- Linear regression: linear regression is a supervised machine learning model where the model finds a linear line of best fit between the independent and dependent variable. It finds the relationship between the dependent and independent variable. This model makes approximation process simple.
- Decision Tree: A decision tree is a supervised learning algorithm used for both classification and regression tasks. The problem is also a type of regression and supervised task. This model required less data preparations for implementation.
- Random forest: random forest is also used for classification and regression problems. It builds decision tree for different samples and take the average for regression problem. This model solves the problem of overfitting as output is based on majority.
- Mean absolute error: absolute error means difference between predicted and original value. MAE refers to the average value of all the absolute error.
- Mean square error: MSE is the average of the square of the difference between prediction and original value.
I am going to implement these three models and by using two error measure I can decide which method is more accurate and useful. Then assert my predictions based on the calculations and accuracy implemented.

## Data Description
The dataset utilized in the current study comes from a competition that was organized by the website statlib and uses data from that website. This work utilizes feature selection techniques such as loading the dataset from the sklearn library, searching for null values or outliers and I have also used Variable Influence Factor (VIF). 

A measure of a variable's association with other variables is called the Variance Inflation Factor (VIF). We try to maintain a set of variables such that the Variance Inflation Factors (VIFs) of all the variables are less than those that the classifier predicts 1 when the target value is 1, true negatives are those values in which the classifier predicts 0 when the target value is 0, and false positives are those values in which the classifier predicts 1 when the target value is 1. As a general rule, if the correlation between the variables is high, the Variance Inflation Factor is 1.

For the project I am using dataset “California housing”. Dataset is obtained from statlib repository. The task is to predict the housing price based on many features. This dataset has 20640 instances and 8 features and a target attribute. The target column is MedHouseVal which is median value of house for each district. All the features have float datatype and there are no null values in the dataset.
There is total 8 features:
- MedInc: It is the median income in the group 
- HouseAge: Median house age of the district 
- AveRooms: Average number of rooms 
- aveBedrms: Average number of bedrooms 
- Population: Populations of the district 
- AveOccup: Average number of members in the household
- Latitude and Longitude: Latitude and longitude of the district
This is the target attribute of the dataset. Which is median house value of the district. Here we have to predict the numeric value, so this is a regression problem. This dataset has labels, so this is a classification problem.

![image](https://user-images.githubusercontent.com/61236166/216196475-466e73c3-ed48-4511-9584-6c78d4bd9bc9.png)


## Proposed Framework
The entire work process can be broken into the three sections below. Data preprocessing, data analysis, the use of machine learning techniques, and performance monitoring processes are some aspects.

#### Data Preprocessing : 
I saw that the data contains 21000 instances and almost 8 attributes. We converted the data into pandas for smooth import and checked out each of the fields. Then I tried to get an overview of how our data looked like with the help of various pandas functions.
I also checked if our data had any null or disparaging values which could affect our models or predictions.

![image](https://user-images.githubusercontent.com/61236166/216191198-a0302772-89bd-4289-aaac-232386349af4.png)

#### Data Analysis : 
Prior to creating a model, exploratory data analysis is crucial. We can identify implicit patterns in the data in this way, which will helps us select the best machine learning techniques. We can use various inbuilt libraries to perform data analysis and check the distribution of data. Some of the analysis libraries we used was :
- Heatmap : It help to show the correlation between the attributes in our data. The correlation is between -1 and +1. There is no linear trend between the two variables if the values are closer to zero. The closer the correlation is to 1, the more positively associated they are; that is, as one rises, the other does as well, and the stronger this relationship is, the closer to 1 the correlation is. Through this analysis, I could see a significant relation between median income and the median house value. The heatmap gave a strong and compelling insight on what are the attributes we could use to our advantage in selecting the machine learning models.

![image](https://user-images.githubusercontent.com/61236166/216191479-a8f70380-08b6-410b-a482-c799f40426f9.png)

From the heatmap we can see that median income and median house value are highly corelated. Average bedrooms and median house value are the least corelated. 
Latitude and longitude directly less corelated to median house value but I implemented scatterplot with latitude and longitude with house value. So, I can know the prices with the location of the house.

- Scatterplot : a scatter plot matrix is used to show bivariate correlations between sets of variables. Numerous associations can be investigated in a single chart thanks to the scatter plots in the matrix, which each show the link between a pair of variables. Scatterplot was one of the most important exploratory analysis for the project as I could plot the median house value based on the latitude and longitude. This formed a map of California giving us a clear picture of the accuracy of the data and how the housing prices differ with respect to terrain, county etc.

![image](https://user-images.githubusercontent.com/61236166/216191700-6eff8360-b68d-42ec-a60f-9a45372afcbe.png)

Above scatterplot has been implemented with latitude as Y axis and longitude as X axis and the datapoints are house values. we can see that all datapoints represent the map of California. From this scatterplot we can see that high value houses are mostly in the cities like San Francisco, Los Angeles, San Diego, San Jose. 

- Pair Plot :  Pair plots are used to determine the most distinct clusters or the best combination of features to describe a connection between two variables. By creating some straightforward linear separations or basic lines in our data set, it also helped to create some straightforward classification models. In the pair plot I dropped the attribute latitude and longitude because they separately did not affect the target attribute. 

![image](https://user-images.githubusercontent.com/61236166/216192287-d601170c-2bc5-4e6d-8c88-d12601f6bae6.png)

In above pair plot first dropped the attribute latitude and longitude because they separately do not affect the target attribute.  From the above pair plot, we can see that as the avgrooms and avgbedrooms increases price of the house also increases. Also, aveOccup is the least corelated with the other attributes.

#### Model Selection
For implementation, I used three machine learning models viz,

- Linear Regression: The simplest prediction technique is linear regression. The predictor variable and the variable that comes first in importance, whether the predictor variable and su, are the two items it employs as variables. The link between one dependent variable and one or more independent variables is explained using these estimations To train the model for Linear Regression, first I imported the required libraries from sklearn and then then created the model using Linear Regression(). I used X_train and Y_train to train the model. To test the model, I used X_valid and assign the predicted output to LR_Pred.

![image](https://user-images.githubusercontent.com/61236166/216192933-6c9304ab-dd9a-429d-8957-ef949108c42f.png)

To test the model, I used X_valid and assign the predicted output to LR_Pred. Also, I calculated the MAE, MSE and accuracy score. The result of linear regression is in the screenshot above.![image](https://user-images.githubusercontent.com/61236166/216193016-2e272d56-54c9-4b11-a178-91b0f9ecb407.png)

- Random Forest: A large number of decision trees are built during the training phase of the random forests or random decision forests ensemble learning approach, which is used for classification, regression, and other tasks. The class that the majority of the trees choose is the output of the random forest for classification problems.
Representation of it is as follows:

 ![image](https://user-images.githubusercontent.com/61236166/216193307-baec7ce7-5e9a-4c63-a4f0-38d715bf3952.png)
 
It builds decision tree for different samples and take the average for regression problem. This model solves the problem of overfitting as output is based on majority. I have used Random Forest algorithm in my project and helped me train my model and predict the Housing prices. This one gave one of the most promising results as compared to other models with an accuracy of almost 80%.

![image](https://user-images.githubusercontent.com/61236166/216193502-2f36fabf-e93d-4131-b8ea-f0542299d39e.png)

To train the model for Random Forest, first I created the model. We use X_train and Y_train to train the model. To test the model, I used X_valid and assign the predicted output to RF_Pred. Also, I calculated the MAE, MSE and accuracy score. The result of linear regression is in the screenshot above. From all three model random forest has the maximum accuracy and minimum error rate.

- Decision Tree: The non-parametric supervised learning approach used for classification and regression applications is the decision tree. It is organized hierarchically and has a root node, branches, internal nodes, and leaf nodes. In decision tree, we select attributes based upon which attribute has highest information gain.

 ![image](https://user-images.githubusercontent.com/61236166/216193704-e0324f93-c3da-4c14-9deb-c35d9b52b8f4.png)

When do we end growing our tree must be a question you are asking yourself. Real-world datasets typically contain a lot of features, which leads to a lot of splits, which produces a massive tree. These trees take long to construct and may result in overfitting. In other words, the tree will provide extremely accurate results on the training dataset but inaccurate results on the test dataset.

Since the Random Forest performed really well with the data, it was a plausible step for me to evaluate our model using Decision Tree.

![image](https://user-images.githubusercontent.com/61236166/216194787-1fe9d86f-1b49-421d-9569-32ffa851a706.png)

To train the model for Decision Tree, first I created the model. I used X_train and Y_train to train the model. To test the model, I used X_valid and assign the predicted output to DT_Pred. Also, I calculated the MAE, MSE and accuracy score. 


## Results and Comparison Analysis
To compare each model’s performance and accuracy for the given dataset MAE, MSE and accuracy score is used. The comparison between these values is shown in the given table.

![image](https://user-images.githubusercontent.com/61236166/216433551-49055023-6799-42d6-b766-5ac780b5fe3d.png)

From the above table we can see the comparison between all the three models. Linear regression has the maximum of mean square error and mean absolute error and have the least accuracy for the given dataset which is 60.47%. while random forest has the minimum mean square error and mean absolute error and have the maximum accuracy which is 79.09%. So, for this study random forest is the best choice for the prediction of the house price and linear regression is not a good choice for the given dataset. Decision tree model perform better than Linear regression but not better than Random Forest.

### I hope you enjoyed a brief explanation of the project. You can find the entire code in housingpriceprediction.ipynb file and all you need is jupyter notebook installed on your laptop/pc to run the project.Do share your feedbacks on the project and if you have any questions, you can always reach out to me.
