# Recommender Systems Machine Learning

This notebook is mainly used Azure Databricks for developing Spark based recommenders such as Spark ALS algorithm, in a distributed computing environment.
The notebooks show how to establish an end-to-end recommendation pipeline that consists of data preparation, model building, and model evaluation by using the utility functions

| Notebook | Dataset | Environment | Description |
| --- | --- | --- | --- |
| ALS | Beauty Products | PySpark | Utilizing ALS algorithm to recommend beauty products in a PySpark environment.

# Data and evaluation

Datasets used in retail recommendations usually include user information, item information and interaction data, among others.

To measure the performance of the recommender, it is common to use ranking metrics. In production, the business metrics used are CTR and revenue per order. To evaluate a model's performance in production in an online manner, A/B testing is often applied.

# Evaluate

In this directory, notebooks are provided to illustrate evaluating models using various performance measures which can be found in recommenders.

| Notebook | Description | 
| --- | --- | 
| diversity, novelty etc. | Examples of non accuracy based metrics in PySpark environment.
| evaluation | Examples of various rating and ranking metrics in Python+CPU and PySpark environments.

Several approaches for evaluating model performance are demonstrated along with their respective metrics.
1. Rating Metrics: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items
    * Root Mean Square Error (RMSE) - measure of average error in predicted ratings
    * R Squared (R<sup>2</sup>) - essentially how much of the total variation is explained by the model
    * Mean Absolute Error (MAE) - similar to RMSE but uses absolute value instead of squaring and taking the root of the average
    * Explained Variance - how much of the variance in the data is explained by the model
2. Ranking Metrics: These are used to evaluate how relevant recommendations are for users
    * Precision - this measures the proportion of recommended items that are relevant
    * Recall - this measures the proportion of relevant items that are recommended
    * Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance
    * Mean Average Precision (MAP) - average precision for each user normalized over all users
3. Classification metrics: These are used to evaluate binary labels
    * Arear Under Curver (AUC) - integral area under the receiver operating characteristic curve
    * Logistic loss (Logloss) - the negative log-likelihood of the true labels given the predictions of a classifier
4. Non accuracy based metrics: These do not compare predictions against ground truth but instead evaluate the following properties of the recommendations
    * Novelty - measures of how novel recommendation items are by calculating their recommendation frequency among users 
    * Diversity - measures of how different items in a set are with respect to each other
    * Serendipity - measures of how surprising recommendations are to to a specific user by comparing them to the items that the user has already interacted with
    * Coverage - measures related to the distribution of items recommended by the system. 
    
References:
1. Asela Gunawardana and Guy Shani: [A Survey of Accuracy Evaluation Metrics of Recommendation Tasks
](http://jmlr.csail.mit.edu/papers/volume10/gunawardana09a/gunawardana09a.pdf)
2. Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.
3. Yehuda Koren and Robert Bell, "Advances in Collaborative Filtering", Recommender Systems Handbook, Springer, 2015.
4. Chris Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.

# Scenarios
## Recommender Systems for Retail

Recommender systems have become a key growth and revenue driver for modern retail.  For example, recommendation was estimated to [account for 35% of customer purchases on Amazon](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers#). In addition, recommenders have been applied by retailers to delight and retain customers and improve staff productivity. 

### Personalized recommendation

A major task in applying recommendations in retail is to predict which products or set of products a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This scenario is commonly shown on the personalized home page, feed or newsletter. Most models such as ALS, BPR, LightGBM and NCF can be used for personalization. [Azure Personalizer](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/concept-active-learning) also provides a cloud-based personalization service using reinforcement learning.

### Cold Items and Cold Users

In this scenario, the user is already viewing a product page, and the task is to make recommendations that are relevant to it.  Personalized recommendation techniques are still applicable here, but relevance to the product being viewed is of special importance.  As such, item similarity can be useful here, especially for cold items and cold users that do not have much interaction data.

### Frequently bought together

In this task, the retailer tries to predict product(s) complementary to or bought together with a  product that a user already put into shopping cart. This feature is great for cross-selling and is normally displayed just before checkout.  In many cases, a machine learning solution is not required for this task.

### Similar alternatives

This scenario covers down-selling or out of stock alternatives to avoid losing a sale. Similar alternatives predict other products with similar features, like price, type, brand or visual appearance.

### Other considerations

Retailers use recommendation to achieve a broad range of business objectives, such as attracting new customers through promotions, or clearing products that are at the end of their season. These objectives are often achieved by re-ranking the outputs from recommenders in scenarios above. 

# 🤝🏽 Support
Need help? Report a bug? Want to share a perspective? Ideas for collaborations? Reach out via the following channels:

- [Project Issues](https://github.com/Lai-AI-BI-CI/Product_Channel_Sales_Analysis/issues): bugs, proposals for changes, feature requests
- [LinkedIn](https://www.linkedin.com/in/kwan-lai-yeung/) / Email [work.kwanlai1036@gmail.com](mailto:work.kwanlai1036@gmail.com): ideal for projects discussions, ask questions, collaborations, general chat
