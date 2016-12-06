# Machine Learning Engineer Nanodegree
## Capstone Proposal
Armen Donigian 

November 22nd, 2016

## Proposal

### Domain Background

The following proposal is a machine learning project to build a predictive model in response to the [**House Prices: Advanced Regression Techniques**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

> Besides being home, real estate has been a very popular investment. While most of us realize the number of bedrooms or bathrooms would influence the price of a home, it turns out that at times features such as the size of the garage or proximity to a particular landmark influences the price just as much.

The competition began on August 30, 2016 while the final submission must be made by March 1st, 2017. Only two submissions per day are allowed.
The top 3 of the leaderboard are:

| Competitor        | Score           |  
| ------------- |:-------------:| 
| maboun      | 	0.07186 |  
| UW Data Science Team 4      | 		0.09403 |  
| HungryFools      | 	0.10051 |  
 
 
This project is of personal interest to me since I am interested in investing in real estate and as with any investment, the purchase price can significantly impact returns. Any insights as well as lessons learned would be directly transferrable future investment gains. Additionaly, Kaggle competitions offer a great opportunity to solve real world problems in a collaborative, competitive and open learning envionrment. Doing well on Kaggle competitions helps enhance my personal experience and online brand.

### Problem Statement

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

The task is to guess the price of a house given the 79 features. This is a **supervized learning** problem which will require **a regressor** to be implemented. The predictors (features) are the 79 supplied attributes of a property. The label (target) is the sale price of a property. See example submission below for details.

### Datasets and Inputs

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. For full details, see data specifications below. 

#### Data Fields
 

| File        | Dimensions (rows, columns)          |  Text or Numeric      |  
| ------------- |:-------------:| -----:|
| train     | 	1459, 81 | Alpha Numeric |
| test      | 		1459, 80 | Alpha Numeric |  
 
 
Here's a brief version of what you'll find in the data description file.

```
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale
```

#### Pre-processing
There are several pre-processing steps which will need to be performed, including but not limited to:
+ apply log transformations for feature scaling & target analysis
+ construct derived representations of string year such as a decade feature 
+ identify null & missing data, apply a mnemonic or drop it (depending on size & impact relative to other training data)

#### File Descriptions

**data_description.txt** is a full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here

**train.csv** is the training set 

**test.csv** test set

**sample_submission.csv** shows the correct submission format.

### Solution Statement
For each Id in the test set, you must predict the value of the SalePrice variable. A potential solution to the problem would look something like the following:

```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
etc.
```


### Benchmark Model

Given this is a Kaggle competition, the benchmark model will be the score of my model on the private leaderboard (data I don't have access to). Additionally, I plan to create an initial benchmark model after generating the training data set without doing any feature engineering or hyper-parameter tuning just to have a point of reference for the rest of the model training process.

My goal is to target the top 20% of the public leaderboard.

### Evaluation Metrics
Given this is a Kaggle competition, the evaluation metric has been specified as the [**Root-Mean-Squared-Error (RMSE)**](https://www.kaggle.com/c/outbrain-click-prediction/details/evaluation). 

![Eval Equation](../assets/Evaluation_Ames__Kaggle.png)

The root mean squared error metric intutively is suitable for house price prediction since it's used to measure difference between values in sample vs values predicted by a model. 

### Project Design
 
The first step to any Machine Learning project is to understand the data as well as the problem (what is the target). The task is to predict house prices.

Here's an end-2-end workflow for building a regressor using `scikit-learn`.

1. Problem Definition (Ames house price data) 2. Loading the Dataset3. Exploratory Data Analysis (some skewed distributions and correlated attributes) 4. Evaluate Algorithms 5. Evaluate Algorithms with Standardization  6. Algorithm Tuning  7. Ensemble Methods (Bagging and Boosting, Gradient Boosting looked good).   Tuning Ensemble Methods (getting the most from Gradient Boosting).8. Finalize Model (use all training data and confirm using validation dataset).

#### Exploratory Data Analysis (expanded...)

I will perform Exploratory Data Analysis (EDA) to explore and better undertand the data. EDA will include the following:

+ examine data types, handle incorrect / non-conforming  
+ examine missing values
+ examine outliers
+ create correlation plots between different features
+ examine feature distributions against target (guassian, few unique, heavy-tailed)
+ skew of univariate distributions
+ univariate plots
+ multimodla data visualizations
+ target analysis
+ create visualizations of insights discovered from analysis

#### Feature Transformation & Engineering (expanded...)

+ does data need to be standardized?
+ convert categorical features to numerical
+ engineer new features from existing features & accordingly to domain knowledge

Next, I would do principle component analysis to better understand the correlation of the data. For example, say you have 100 features, is it really 100 dimensions or fewer. Look at your eigenvalues in order (biplot) and see if it decays fast or slowly. If the dimensions drop off pretty quickly then it implies there’s a lot of structure in your data and that you need less dimensions. On the other hand, if it decays more slowly then it’s harder to find structure. 

Since this is a supervised learning problem, decision trees & variations of tree models are worthwhile. The types of algorithms to try are:

+ Decision Trees
+ Random Forrests
+ Gradient Boosted Machines
+ XGBoost
+ Deep Learning Model using TensorFlow

Consider spending more time tuning the hyper-parameters for better results. I would try deep learning method in the event there’s a non-linear structure which decision trees weren’t able to pick up on.

-----------
