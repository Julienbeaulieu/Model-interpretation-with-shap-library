RECS is a periodic survey sponsored by the U.S. Energy Information Administration (EIA) that provides detailed information about energy usage in U.S. homes. 

Data source: [RECS Survey data](https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata)

Our goal in this analysis is to build a model that predicts energy consumption in `KWH`. 

Our analysis will also go beyond getting the best possible predictions. We will focus on gaining insights that can then be used by analysts and decision makers in order to help them meet future energy demand and improve efficiency as well as building design.

We will answer the following questions using data visualization and the [SHAP library](https://github.com/slundberg/shap): 
1. Which columns in the dataset were the most important for our predictions? 
2. How are they related to the dependent variable? 
3. How do they interact with each other? 
4. Which particular features were most important for a particular observation?

Finally, we will decide which model to use based on the following: 
- Capacity to answer the above questions
- Performance
- Ease of implementation

The models we will use and compare are:
- Random forest
- XGBoost
- Neural network using entity embeddings of categorical variables [Entity Embeddings Paper](https://arxiv.org/abs/1604.06737)