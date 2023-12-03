# What is Food Insecurity 
According to the USDA, food insecurity is defined as the “lack of consistent access to enough food for every person in a household to live an active, healthy life.” Food insecurity can be caused by several different life circumstances including poverty, illness, and systemic racism [1]. 

# Why is it Important?
Food insecurity is widespread across the United States, affecting more than 44 million people, 13 million of those being children [1]. In Georgia alone, 1 in 9 people face food insecurity [2]. Food insecurity deeply affects people’s daily lives, and prohibits the ability to live a full, healthy life. Because it is such a vast issue, it is also a difficult one to solve.

<img src="/images/num_insecure.png" width="40%" height="40%">

# How Can We Help?
A key step to addressing the issue of food insecurity is raising awareness. This project aims to explore the issue of food insecurity here in Georgia, highlighting where there may be food deserts within the state. We aim to explore food insecurity in each county by different variables, demonstrating possible predictors of food insecurity. Our goal in doing so is to provide you with knowledge on this issue, so that we as a community can better advocate for reform and inform others about food insecurity in our own communities. 

# Terminology to Know
**Low-Access:** In this dataset, food access can be defined by  the distance to the nearest supermarket, supercenter, or large grocery store. For urban counties, we consider individuals who live more than 1 mile away from the nearest supermarket or grocery store as food insecure. For rural counties, where population density is intrinsically lower and grocery stores are fewer, individuals are considered to be food insecure if they live greater than 10 miles from the nearest food source.  

**Low-Income:** Low-income counties, as defined by the Department of Treasury’s New Markets Tax Credit (NMTC) program, are those where the poverty rate is 20% or greater, the median family income of the county is less than 80% of the state’s median family income, or the tract is in a metropolitan area and the median family income is less than 80% of the greater metropolitan median family income. 

**Food Desert:** Regions of the country [that] often feature large proportions of households with low incomes, inadequate access to transportation, and a limited number of food retailers providing fresh produce and healthy groceries for affordable prices. [6]

**Poverty Rate:** The proportion of the population in the county that lives below the Federal poverty threshold. 

**Urban:** According to the US Census, to qualify as an urban area, the territory identified according to criteria must encompass at least 2,000 housing units or have a population of at least 5,000.

**Rural:** According to the US Census, all people, housing, and territory that are not within an urban area.

**SNAP:** Supplemental Nutrition Assistance Program (SNAP) is the largest federal nutrition assistance program. SNAP provides benefits to eligible low-income individuals and families via an Electronic Benefits Transfer card. This card can be used like a debit card to purchase eligible food in authorized retail food stores. [5]

**Tract:** small, statistical subdivisions of a county used by the US Census to gather data. For our project, tract data was aggregated to be cumulative data sets of the entire county, rather than individual tracts.

# Food Insecurity by County
Georgia is made up of 159 counties. The counties can be differentiated as rural and urban. 
The graph below outlines the counties in Georgia that can be identified as rural and urban. As you can see…

[INSERT GRAPH]

**Does living in a rural vs. urban area affect food insecurity in each county?**

<img src="/images/num_rural_urban.png" width="50%" height="50%">

The graph below shows the percent of county populations that are low income and low access, two factors that determine food insecurity. Low access and low income are encoded by saturation of color (darker being higher percentages of low income/access, lighter being lower percentages). Rural and Urban are determined by specific colors (blue for rural and brown for urban).

Taliaferro County, a rural county east of Atlanta, has the highest percentage of food insecurity. Among its 1,717 residents, 47.43% are low income and have low access to food. Chattahoochee County, an urban county southeast of Columbus, has the second highest percentage of food insecurity, as 39.34% of its 11,267 residents are low income and have low access to food. In Georgia, urban residents are more likely to face food insecurity. 11.11% of the urban population is low income and has low access to food, compared to 7.15% of the rural population.

[INSERT GRAPH]

# Food Insecurity by Distance

# Food Insecurity by Population Density
Living in a rural or urban area can greatly affect your proximity to a viable food source. The Pareto chart below explores the question: **How does a county’s population/population density influence the likelihood that the county has food insecurity issues?** The histogram bins counties by their population density. The color saturation/heat map encodes the number of food insecurity in the county at 1 and 10 miles (these are Census identified mile markers for food deserts). The orange trend line shows the cumulative percentage of food insecurity cases across the state , and allows you to see which levels of  population density account for what percentage of food insecurity instances in the state. Looking at the graphs, we can see that there is a large distinction between population density in rural and urban areas. This gap indicates that areas in Georgia are either very sparsely populated or very heavily populated, with little existence of a middle ground. While you might expect that food insecurity is more present in sparsely populated areas, where people would theoretically be living further from grocery stores, the opposite is actually true. In fact ~38% of the food insecurity cases in Georgia exist in the 5 most densely populated counties: Fulton, Clayton, Gwinnett, Cobb, and DeKalb. When considering just counties that contain more urban census tracts than rural census tracts, their responsibility for food insecurity cases jumps to around 45%. 

[INSERT GRAPH]

# Food Insecurity by Vehicle Access

# Food Insecurity by Race
Systemic racism is a big factor in the creation of communities with high levels of food insecurity. The following graph demonstrates food insecurity by race. Each dot is a county, and the x-axis shows the low access population of each race for a particular county. The size of the dot corresponds to the size of the total low access population of a county. As we expect, larger counties tend to have larger low access populations for each race; however, some counties, such as Cherokee county, have a large low access population for a particular race despite not having a large total low access population. This suggests that there is a disparity in food insecurity for certain races in certain counties.To compare the low access populations by race for a particular county, click on one of the dots, and the visualization will highlight all of the dots for that county.

<div style="margin:1em calc(70% - 50vw);">
  <div class='tableauPlaceholder' id='viz1701475078774' style='position: relative'><noscript><a href='#'><img alt='Low Access Population by Race '
                    src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book2_17014750715760&#47;LowAccessPopulationbyRace&#47;1_rss.png'
                    style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' /> <param name='site_root' value='' />
            <param name='name' value='Book2_17014750715760&#47;LowAccessPopulationbyRace' />
            <param name='tabs' value='no' /><param name='toolbar' value='yes' />
            <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book2_17014750715760&#47;LowAccessPopulationbyRace&#47;1.png' />
            <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' />
  </object></div>                
  <script type='text/javascript'>                    var divElement = document.getElementById('viz1701475078774');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
</div>

# Food Insecurity in Seniors vs Children
In addition to race, seniors and children experience high levels of food insecurity. Seniors in particular generally have limited finances and resources [3]. Children experience food insecurity at high rates, especially those situated in single mother households, which has one of the highest food insecurity rates [4]. The unit visualizations below show the number of kids and seniors in each county that face food insecurity issues. Each symbol represents 1000 people, and the counties are represented by color. The biggest takeaway from this graph is the sheer number of these more vulnerable groups, and especially children, that experience food insecurity. Again, we see the most densely populated counties experiencing extreme amounts of food insecurity, with Gwinnett alone having 93,063 children experiencing food insecurity issues. Generally speaking, the five most heavily populated counties again top the charts with cases for vulnerable groups, though not necessarily in the same order.

[INSERT GRAPH]

# Food Insecurity by Governmental Benefits
**Does a county’s number of SNAP recipients and the poverty rate have any correlation?**

Governmental assistance programs such as SNAP are a viable resource to provide people with greater access to food. The following scatterplot demonstrates poverty rate on the x axis, and allows for you to change the y axis to SNAP households in general, and SNAP households that are either 1 mile or 10 miles from a grocery store (these are two distance standards for food deserts for urban and rural areas respectively, as determined by the US Census). The scatterplots reveal that though there are many households that utilize SNAP in Georgia, there is not much correlation between poverty rate and SNAP usage or lack thereof, even for those living within the furthest measured distance from a grocery store (10 miles). 

[INSERT GRAPH]

# Close to Home: Metro ATL Counties Food Insecurity Ranking
Now that we know more about the prevalence of food insecurity in Georgia, you may be overwhelmed by the question “There’s so much need across the state, how can we help?”. In order to put the need into a more tangible context, the bump chart below shows the counties where we are, in the Metro Atlanta area, ranked from 1-11, 1 being the most food insecure of the group, and 11 being the least, across 4 years from 2018-2021 [7]. Hover over each point to see the percentage of food insecurity in each county over the years. As we can see, food insecurity is an issue closer to us than it seems. So, what can we do about it?

[INSERT GRAPH]

There are many things we can do locally to take action towards solving food insecurity:

-[Volunteer](https://www.acfb.org/volunteer/) at your local food bank. The Atlanta Community Food Bank serves these counties and more, and has several volunteer opportunities for you to get involved and make a difference in your local community.

-Donate to organizations who are fighting hunger and food insecurity. 

-Continue to raise awareness of food insecurity! Food insecurity affects people all across the country. The more informed we are about the issue, the better equipped we will be to solve it.



## 3.1 Preprocessing for User-based Analysis
The user dataset we extracted contained numeric data (“rating”, “age”) and categorical data (“country”). This made clustering the data more difficult because most unsupervised learning techniques handle only numeric data. The two methods we used to handle our mixed user dataset are Gower distance and one hot encoding.

Some additional preprocessing was performed on the user dataset before clustering. Each “user_id” was assigned a single rating which was the computed average of all the “user_id” ratings. The country column was dropped and processed to produce a new continent column; this was done to reduce the number of unique entries in the categorical data.
 
The first method used to cluster the mixed data is one hot encoding. One hot encoding is an encoding method used on categorical features that assign each categorical feature a binary encoding. After one hot encoding was performed on the “continent” feature, we used the unsupervised methods K-Means and GMM to cluster users.

The second method used to cluster the mixed data is Gower distance. Gower distance measures the similarity between two mixed data points by handling numeric and categorical features separately and then combining the scores to output an overall similarity score. Manhattan distance is used to measure similarity between numeric features, and one hot encoding is used with Jaccard distance to measure similarity between categorical features. 

![Figure 1 - Data after one-hot encoding](/images/fig1.png)

## 3.2 Preprocessing for Item-based Analysis
For clustering users, we use three features: ‘age’, ‘country’, and the calculated average rating of each user.

In this section, we firstly only considered the feature ‘rating’ of the books by all users. As shown in the histogram below, there are ~60% values of 0, possibly indicating missing values, which are meaningless. Therefore, we dropped all the data points with zero ratings before doing any analysis.

![Figure 2 - Histogram of rating](/images/fig2.png)

We then created a spreadsheet-style pivot table with ‘user_id’ as index, ‘book_title’ as column and ‘rating’ as the values of the cells.

![Figure 3 - Part of pivot table](/images/fig3.png)  
As shown in the pivot table, the majority of the cells are NaNs. That’s because not all users rated all books (in fact, there is a very limited number of books that each user can rate/read). It’s difficult to compute either Euclidean distance or cosine distance for data points with a large number of NaNs. Whereas it would be much easier if we cluster the users by the features related to users (e.g., age, location, etc.), then use the average rating of the book by each cluster of the users to calculate book similarity.
 Since there are more than 100 countries in the user information, we converted the country to continent using the python package pycountry_convert and used one-hot encoding to represent the continent feature.
 
## 3.3 Preprocessing for Content-based Analysis Using Book Features
We also want to cluster books based on four book features: book title, author, category, and publisher. We joined these four features together into one column. With this combined column, we used a count vectorizer to create a document-term matrix of the count of each unique word per book.

## 3.4 Preprocessing for Content-based Analysis Using Summary
For the book summary analysis, we created the numeric dataset from the original sentence, creating the dimension with the unique words in the total dataset. In order to obtain the numeric dataset, we went through the process: 1) we sorted out the 100 ‘common books’ that are rated by diverse users to avoid the possible excessive dimension problem, 2) we sorted out the words that are not in the stopwords [7] to achieve the unique set of words from the summary of the books, 3) created vectorized numeric output as described from the content-based feature analysis.

![Figure 4 - Sample of summary processing](/images/fig4.png) 

## 3.5 Preprocessing for Category Prediction:
Because there were thousands of different unique categories in our dataset, we chose to only use the 10 most popular categories shown below.

![Figure 5 - Count of books across 10 categories](/images/fig5.png)  
The distribution of these categories is considerably uneven, so we decided to randomly sample 1400 books from each of the 10 categories for a total of 14,000 books. Then, we obtained the book cover images for each of these books, resized them to be 32x32, and saved the image data in a file.

## 3.6 Preprocessing for Book Cover Images:
From the complete dataset, the book cover image were provided with the url link with different sizes: small, medium and large. To minimize the loss from the small pixel sizes, we filtered out the image dataset based on the large size images. The filtering process includes three different steps: 1) remove invalid image url, 2) filter out black and white images, 3) drop the duplicated images. First, 830,968 images that have (width,height,3) dimension were filtered to ensure the dataset only have RGB colored arrays, and only unique image urls were selected, which were finalized with 173,515 image url dataset.

![Figure 6 - Image dataset samples](/images/fig6.png)  
The training and test set were randomly selected without replacing duplicate indices with dimension (N,width,height,3), based on the features that were labeled into the dataset.


# 4. Unsupervised Learning - Methods, Results, and Discussion
## 4.1 User-based analysis
We clustered users based on the following features in the dataset: “country”, “rating” and “age.” Details of our data preprocessing for this section can be found in section 3.1. Two methods were used to process the mixed numeric and categorical data: one-hot encoding and gower distance. After preprocessing the data, we used k-means and GMM to cluster similar users together.

### 1) One-hot encoding Results
![Figure 7 - K-means loss graph](/images/fig7.png)  
From the K-Means elbow analysis, we can see that the optimal number of clusters is 4.

![Figure 8 - K-means Davies-Bouldin graph](/images/fig8.png)  
From the K-Means DB analysis, we can see that the optimal number of clusters is 3.

![Figure 9 - GMM silhouette score graph](/images/fig9.png)  
From the GMM silhouette analysis, we can see that the optimal number of clusters is 8.

![Figure 10 - GMM Davies-Bouldin graph](/images/fig10.png)  
From the GMM DB analysis, we can see that the optimal number of clusters is 2, 5, or 9.

### 2) Gower Distance Results

![Figure 11 - Gower Elbow Curve](/images/fig11.png)  
From the Gower elbow analysis, we can see that the optimal number of clusters is 4.

![Figure 12 - Gower Davies-Bouldin graph](/images/fig12.png)  
From the Gower DB analysis, we can see that the optimal number of clusters is 4 and 7.

### Discussion (4.1 User-based analysis)
Results from the Gower clustering method agreed that the optimal number of clusters for the user data set is 4, while the one hot encoding method results were split between 4 and 8 for the optimal number of clusters. Therefore we decided that the optimal number of clusters for clustering our mixed user dataset was 4, because both methods agreed on it.

## 4.2 Item-based analysis
Item-based recommendation system, also known as item-item collaborative filtering or IBCF, is a type of recommendation system that is based on the similarity between items calculated using the rating users have given to items. The core IBCF is all about finding items similar to the ones the user has already liked. For example, take two books, A and B, and check the ratings of all users who rated both books. If these ratings are similar, then these two books are likely to be similar, so someone who likes book A might also like book B. Therefore, we need to calculate the pairwise similarity of the books by the ratings from all users.

### Methods (4.2 Item-based analysis)
Details of our data preprocessing for this section can be found in section 3.2. For clustering users, we use three features: ‘age’, ‘country’, and the calculated average rating of each user (might indicate a user’s rating tendency).
We tried three unsupervised clustering methods: KMeans, GMM, and DBSCAN. 

### Results and Discussion (4.2 Item-based analysis)
**K-means:**  
![Figure 13 - Elbow curve for K-means](/images/fig13.png)  
Based on the elbow curve shown above, 8 or 9 clusters might be reasonable for KMeans clustering.

**GMM:**  
![Figure 14 - Silhouette scores for GMM](/images/fig14.png)  
For GMM clustering, no matter how the number of clusters changes, the Silhouette scores are always much less than 1, indicating a poor cluster of the data points.

**DBSCAN:**  
![Figure 15 - 10th nearest neighbor plot](/images/fig15.png)  
For DBSCAN, We determined the appropriate epsilon level using the elbow method. We used the 10th nearest neighbors to plot the sorted distances of every point to its 10th nearest neighbor. Based on the elbow curve, an epsilon value around 0.2 may be appropriate. However, when using epsilon value of 0.2 and minimum samples of 10, there are 354 clusters, which is too many to eliminate NaN values in the pivot table.

**Item-based Recommendation:**  
Therefore, we used the Gower method discussed in the user-based recommendation section to cluster users. Number of clusters = 4 was used for the Gower clustering method and the average rating of books by users in each cluster was calculated as the pivot table value. Finally, we calculated the pair-wise cosine distance of each pair of books and recommended the top five books that are similar to the target book.  
For example, a user who likes '1st to Die: A Novel' might also like: 'The Pearl', 'One for the Money (Stephanie Plum Novels (Paperback))', 'The Cabinet of Curiosities’, 'Interview with the Vampire', "The Devil's Teardrop".

## 4.3 Content-based Analysis Using Book Features
### Methods
 We also clustered books based on four book features: book title, author, category, and publisher. After preprocessing the data to get our document-term matrix (details in section 3.3), we used k-means, GMM, and DBSCAN on this matrix to cluster similar books before recommending 5 random books that are in the same cluster as the input book. Below, we show some example recommendations for the book The Testament, and metrics such as loss, Davie-Bouldin, and silhouette score for the three clustering methods.

### Results and Discussion (4.3 Analysis using book features)
**K-means:**  
One set of recommendations obtained using k=30: ['The Street Lawyer', 'Cold Sassy Tree', 'Writ of Execution', 'Wuthering Heights (Wordsworth Classics)', 'Breakfast of Champions'].  
We plotted the loss, Davies-Bouldin scores, and Silhouette scores for K-means below.

![Figure 16 - Loss graph for K-means](/images/fig16.png)  
![Figure 17 - Davies-Bouldin graph for K-means](/images/fig17.png)  
![Figure 18 - Silhouette scores graph for K-means](/images/fig18.PNG)  
Using the elbow method on the graphs, a good value of k may be 30. Unfortunately, the silhouette scores for K-means were always close to 0, meaning that clustering using K-means may not be effective.

**GMM:**  
One set of recommendations obtained using 6 components: ['Master of the Game', 'The Apprentice', 'Catch Me If You Can: The True Story of a Real Fake', 'Standing in the Rainbow', 'The Grapes of Wrath'].  
We plotted the Davies-Bouldin scores and Silhouette scores for GMM below.

![Figure 19 - Davies-Bouldin graph for GMM](/images/fig19.PNG)    
![Figure 20 - Silhouette scores graph for GMM](/images/fig20.PNG)  
Based on the graphs, there doesn’t seem to be a very good number of components for GMM, with the silhouette scores always being close to 0. This means that GMM may not be an effective clustering algorithm, regardless of the number of clusters.  We chose 6 components to use.

**DBSCAN:**  
One set of recommendations obtained using epsilon=2.5 and min_samples=10: ['The Client', 'Cry to Heaven', 'The Carousel', 'Secrets', 'Hush'].  

![Figure 21 - 4th nearest neighbors graph](/images/fig21.PNG)  
We used 4th nearest neighbors to plot the sorted distances of every point to its 4th nearest neighbor. The elbow method suggests that an epsilon value between 2.5-3 may be appropriate.

## 4.4 Content-based Analysis Using Summary  
### Methods and Algorithm selection
The book summary was three different learning algorithms utilized for the recommender from the summary dataset. For the summary-based recommender, the K-means, DBSCAN, and GMM were used on the document-term matrix from section 3.4 for the implementation.

### Results (4.4 Analysis Using Summary)  
The recommendation was generated from the book ‘Harry Potter and the order of Phoenix’, categorized as Juvenile Fiction. 5 random books were selected from the same cluster after the clustering, printed with category on the bottom of the book cover image. 

**K-means:**  
![Figure 22 - Summary-based K-means book recommendations](/images/fig22.PNG)  

![Figure 23 - Summary-based K-means elbow curve](/images/fig23.PNG)  
For the K-means, the recommendations were mostly in the Fiction category. Based on the elbow curve, the 11 clusters were selected to be processed.

**DBSCAN:**  
![Figure 24 - Summary-based DBSCAN book recommendations](/images/fig24.PNG)  

![Figure 25 - Summary-based DBSCAN elbow curve](/images/fig25.png)  
The recommendation from DBSCAN was not effective for summary-based clustering. The Davies-Bouldin score remained high with Epsilon value and no cases returned more than a single cluster from the classification.

**GMM:**  
![Figure 26 - Summary-based GMM book recommendations](/images/fig26.png)  

![Figure 27 - Summary-based GMM elbow curve](/images/fig27.png)  
The recommender from GMM returned a similar recommendation from the K-means result. Choosing from the Elbow curve, 7 clusters were selected for GMM clustering.


# 5. Supervised Learning - Methods, Results, and Discussion  
## 5.1 Book Category Prediction  
Our goal is to predict the category of a book from 10 possible categories using the image of the book’s cover.

### Methods (5.1)  
We chose to use a convolutional neural network as our model because our data is in the form of images. Using the image data for 14,000 book covers from section 3.5, we split the data into 80% training data and 20% testing data. 

The CNN’s training variables are as follows:  
&emsp; Batch Size: 64  
&emsp; Epochs: 30  
&emsp; Learning Rate: 0.001  

The model architecture is as follows:  
![Figure 28 - Category CNN architecture](/images/fig28.png)  
This model has a total of 1,148,586 parameters.

Activation functions: Each of the convolution and dense layers uses Leaky ReLU with a slope of 0.1 except for the final dense layer which uses softmax since we are dealing with a classification problem.

We chose to use Adam for the optimizer and categorical cross entropy for the loss.

### Results and Discussion (5.1 Category Prediction)  
Using the above model and training variables, we achieved a test loss of 2.045018434524536
and a test accuracy of 26.04%. 

![Figure 29 - Category prediction accuracy plot](/images/fig29.png) 

![Figure 30 - Category prediction loss plot](/images/fig30.png)  

![Figure 31 - Category prediction confusion matrix](/images/fig31.png) 

Unfortunately our test accuracy of 26% is not desirable. However, given the difficult classification task and the size of our training data, the accuracy is not as low as one might expect. If we had a larger dataset, it is possible that we could have achieved a significantly higher test accuracy. It is likely that 1400 books in each category is not large enough to train a model for a task as difficult as classifying the category based only on the book cover image, especially when considering that some of the categories are quite similar to one another. In addition, we chose to resize the images to 32x32 to reduce computation time. It is possible that with higher resolution images, the test accuracy may be higher. Ultimately, we are not disappointed by the low accuracy of the model considering the size of our dataset, the difficult classification task, and the size and quality of the images.

If we look at the confusion matrix, the model seems to behave as one might expect: it often confuses juvenile fiction for juvenile nonfiction and vice versa. It also has a hard time properly classifying social science books which makes sense because social science has similarities to history, religion, and biography/autobiography. The model seems to make mistakes similar to those a human may make if given the same classification task, although this is certainly subjective.

## 5.2 Book Rating Prediction  
The goal of the book rating prediction model is to predict the average user given rating for a book using the book’s cover image.

### Methods (5.2)  
The original data set was preprocessed to compute the average rating for each book. Data points with the same title were first averaged then rounded to the nearest digit. This ensured each book was classified with a label rating in the range of 1-10.  To produce the final dataset an equal number of datapoints were sampled from each classification. We took 1000 samples for each rating in the range of 1-10, giving us a final data set of size 10000 samples. The dataset was then split to produce the training dataset and testing dataset. We used 90% of the samples to train our model and 10% to test. Giving us exactly 9000 training points and 1000 testing points.

Initially the model used in the category prediction process was also used to predict rating.  When performing preliminary runs on small datasets we noticed that the model training accuracy was increasing with every epoch but its test accuracy was decreasing. This meant that the model was overfitting. To prevent overfitting we increased the drop percentage after every maxpool layer from 0.4 to 0.6. The model's hyperparameters were also tuned, the learning rate was slightly increased and the batch size was decreased.

The rating CNN’s parameters:  
&emsp; Batch Size: 32  
&emsp; Epochs: 20  
&emsp; Learning Rate: 0.01  

![Figure 32 - Rating CNN Summary](/images/fig32.png) 

### Results and Discussion (5.2 Rating Prediction)  
![Figure 33 - Rating CNN model accuracy](/images/fig33.png) 

![Figure 34 - Rating CNN model loss](/images/fig34.png)  

![Figure 35 - Rating CNN confusion matrix](/images/fig35.png) 

Using the customized rating model we were able to obtain a test accuracy of 12.3%. The model's accuracy is very low, and at first glance this model may seem ineffective or impractical, but realistically we know from the real world there is almost no relationship between the cover image of a book and its rating. This is because the rating of a book is tied to the book's content rather than its cover image.

It must be noted that the uniform random probability of  predicting the rating is 10% (because there are 10 different classifications),  and that the rating model accuracy is 12.3%. This means for two almost unrelated variables, book cover image and rating, our model is able to classify book ratings better than the uniform random probability, which in all is a relatively decent result.

Perhaps with a larger dataset and less loss due to image compression we may have been able to obtain a larger test data accuracy, but overall the accuracy would not have changed much due to the fact that the variables are inherently unrelated.

## 5.3 Year of Publication Prediction  
To predict the year of publication from the book cover image, we firstly separate all the images into three different categories: earlier than 1990, 1990-2000, later than 2000. The categories are separated by the distribution of the year of publication of the books. 

![Figure 36 - Distribution of year of publication](/images/fig36.png)  
Then we randomly selected 1200 images from each category, respectively (3600 images in total) and split them into three different sets: training set (containing 1000 images from each category), validation set (containing 100 images from each category) and test set (containing 100 images from each category).

All the images have been reshaped to 224*224 in this section and the training variables are as follows:  
&emsp; Batch Size: 32  
&emsp; Epochs: 20  
&emsp; Learning Rate: 0.0001  

The loss function used to evaluate the models is “categorical_crossentropy”. The optimizer used is Adam (Adaptive Moment Estimation).

### 5.3.1 Custom CNN  
For the custom CNN, we use 6 convolutional layers and 3 fully connected layers. There are 1,935,363 trainable parameters in total. The architecture of the custom CNN is shown below:  

![Figure 37 - Architecture of the custom CNN](/images/fig37.png)  

We train the model for 20 epochs and the the performance metrics are listed below:  
&emsp; Training Set: Accuracy = 39.0%  
&emsp; Testing Set: Precision = 39.3%, Recall = 38.0%, F1-score = 34.2%  

![Figure 38 - Training and validation accuracy and loss curves](/images/fig38.png)  

![Figure 39 - Confusion matrix of the test set](/images/fig39.png)  

The training accuracy is 39.0%, which is only slightly higher than random (33.3%).

### 5.3.2 Pretrained VGG16  
VGG is a deep convolutional neural network that was proposed by Karen Simonyan and Andrew Zisserman [8]. VGG16 is composed of 13 convolutional layers, 5 max-pooling layers, and 3 fully connected layers. Therefore, the number of layers having tunable parameters is 16 (13 convolutional layers and 3 fully connected layers). The architecture of VGG16 is shown below:

![Figure 40 - Architecture of the VGG16](/images/fig40.png)  

In this study, we apply the technique of transfer learning by loading the VGG16 model with pretrained parameters and adding an extra fully connected layer as the last layer. The last added layer generates a tensor output of 3 channels, representing the probability distribution over the 3 catogeries (earlier than 1990, 1990-2000, later than 2000). We train the pretrained VGG16 in two ways: 
1. VGG16-1: Fine tune the last dense layer, which contains 12,291 trainable parameters.
2. VGG16-2: Freeze the first 10 layers of VGG16 and train the last 6 layers, which contains 132,537,347 trainable parameters.

**1) VGG16-1**  
The architecture of the fine-tuned VGG16 is shown below:  

![Figure 41 - Architecture of fine-tuned VGG16](/images/fig41.png)  

We train the model for 20 epochs and the performance metrics are listed below:  
&emsp; Training Set: Accuracy = 47.6%  
&emsp; Testing Set: Precision = 48.5%, Recall = 47.7%, F1-score = 47.7%

![Figure 42 - Training and validation accuracy and loss curves](/images/fig42.png)  

![Figure 43 - Confusion matrix of the test set](/images/fig43.png)  

**2) VGG16-2**  
The architecture of the VGG16  after freezing the first 10 layers is shown below:

![Figure 44 - Architecture of VGG16 after freezing the first ten layers](/images/fig44.png)  

We train the model for 20 epochs and the performance metrics are listed below:  
&emsp; Training Set: Accuracy = 51.3%  
&emsp; Testing Set: Precision = 53.0%, Recall = 52.3%, F1-score = 50.2%  

![Figure 45 - Training and validation accuracy and loss curves](/images/fig45.png)  

![Figure 46 - Confusion matrix of the test set](/images/fig46.png)  

### 5.3.3 Pretrained ResNet50  
ResNet-50 is a convolutional neural network that is 50 layers deep. It uses the batch normalization (BN) to train their model, ensuring forward propagated signals to have non-zero variances and backward propagated gradients exhibit healthy norms with BN, addressing the vanishing gradient problem.  The architecture of VGG16 is shown below:  

![Figure 47 - Architecture of the ResNet50](/images/fig47.png)  

In this study, we apply the technique of transfer learning by loading the ResNet50 model with pretrained parameters and adding an extra fully connected layer as the last layer. The last added layer generates a tensor output of 3 channels, representing the probability distribution over the 3 catogeries (earlier than 1990, 1990-2000, later than 2000). We train the pretrained ResNet50 in two ways:  
1. ResNet50-1: Fine tune the last dense layer, which contains 6,147 trainable parameters.
2. ResNet50-2: Train the whole model, which contains 23,540,739 trainable parameters. 

**1) ResNet50-1**  
The architecture of the fine-tuned ResNet50 is shown below:

![Figure 48 - Architecture of the fine-tuned ResNet50](/images/fig48.png)  

We train the model for 20 epochs and the performance metrics are listed below:  
&emsp; Training Set: Accuracy = 56.6%  
&emsp; Testing Set: Precision = 47.0%, Recall = 44.0%, F1-score = 43.6%  

![Figure 49 - Training and validation accuracy and loss curves](/images/fig49.png)  

![Figure 50 - Confusion matrix of the test set](/images/fig50.png)  

**2) ResNet50-2**  
The architecture of ResNet50 that all the layers are trainable is shown below:

![Figure 51 - Architecture of the ResNet50 that all the layers are trainable](/images/fig51.png)

![Figure 52 - Training and validation accuracy and loss curves](/images/fig52.png)

![Figure 53 - Confusion matrix of the test set](/images/fig53.png)

### 5.3.4 Discussions and Model Comparison
![Table 1 - Comparison of models](/images/table1.png)  

Among all these models trained in this study, the ResNet50 model with all the trainable layers obtained the highest training accuracy, which is 81.5%. while its validation and test accuracy is around 50%, probably indicating overfitting. The modified VGG16 model with first 10 frozen layers performed best for the test set, possibly because it has the most trainable parameters.

![Table 2 - Model parameter numbers](/images/table2.png) 

Overall, the accuracy of all these five models is not high enough, possibly because the book cover images do not contain much information about the feature we select, year of publication. In addition, the feature of the year of publication shows a skewed distribution. It might be relatively easier to distinguish books published before 1990 and later than 2000, but it’s hard to tell the books published between 1990 and 2000 from the other two categories, which is also indicating by the confusion matrixes, since it’s really a narrow time period so that book cover images may not have distinctive features.


# 6. Conclusion
The proposed book recommender was derived from the analysis of correlation with different features in our project. Several computational machine learning methods were applied to 1 million datasets, applying both supervised and unsupervised methods. From the unsupervised machine learning studies, k-means, DBSCAN, and GMM were applied for item-based, content-based, and user-based features in the dataset. The effectiveness of the algorithm varied with different feature analyses, but the k-means were driven to be effective for three different feature groups. We also processed through supervised machine learning with book cover images, using the Convolutional Neural Network(CNN). We analyzed the effectiveness of our model based on three different feature labels: category, year of publication, and user ratings. Correlations were weak between the cover image and the features, but the rational guess was reinforced that category and year of publication have more correlation than the rating to the book cover images.


# References
1. Pazzani, M.J., Billsus, D. (2007). Content-Based Recommendation Systems. In: Brusilovsky, P., Kobsa, A., Nejdl, W. (eds) The Adaptive Web. Lecture Notes in Computer Science, vol 4321. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-72079-9_10
2. Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. Egyptian Informatics Journal, 16(3), 261–273. doi:10.1016/j.eij.2015.06.005
3. Pu, P., Chen, L., & Hu, R. (2011). A User-Centric Evaluation Framework for Recommender Systems. Proceedings of the Fifth ACM Conference on Recommender Systems, 157–164. Παρουσιάστηκε στο Chicago, Illinois, USA. doi:10.1145/2043932.2043962
4. Uko E Okon, B O Eke and P O Asagba. An Improved Online Book Recommender System using Collaborative Filtering Algorithm. International Journal of Computer Applications 179(46):41-48, June 2018.
5. Anwar, Khalid and Siddiqui, Jamshed and Saquib Sohail, Shahab, Machine Learning Techniques for Book Recommendation: An Overview (March 20, 2019). Proceedings of International Conference on Sustainable Computing in Science, Technology and Management (SUSCOM), Amity University Rajasthan, Jaipur - India, February 26-28, 2019, Available at SSRN: https://ssrn.com/abstract=3356349 or http://dx.doi.org/10.2139/ssrn.3356349
6. Dataset: Sercan Yeşilöz. (2021). Book-Crossing: User review ratings, V.37. Retrieved May 27, 2022 from https://www.kaggle.com/code/sercanyesiloz/book-recommendation-system/data.
7. Rajaraman, A., & Ullman, J. D. (2011). Mining of massive datasets. Cambridge University Press.
8. Simonyan, Karen, and Andrew Zisserman. “Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014).
