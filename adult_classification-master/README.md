# Adult Data Set 
Download: [Data Folder](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/), [Data Set Description](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
https://archive.ics.uci.edu/ml/datasets/adult
Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
**Given**: This dataset performs very well with linear models. 
<table border="1" cellpadding="6">
    
    <tr>
        <td bgcolor="#DDEEFF"><b>Data Set Characteristics:</b></td>
        <td>Multivariate</td>
        <td bgcolor="#DDEEFF"><b>Number of Instances:</b></td>
        <td>48842</td>
        <td bgcolor="#DDEEFF"><b>Area:</b></td>
        <td>Social</td>
    </tr>
    <tr>
        <td bgcolor="#DDEEFF"><b>Attribute Characteristics:</b></td>
        <td>Categorical, Integer</td>
        <td bgcolor="#DDEEFF"><b>Number of Attributes:</b></td>
        <td>14</td>
        <td bgcolor="#DDEEFF"><b>Date Donated</b></td>
        <td>1996-05-01</td>
    </tr>
    <tr>
        <td bgcolor="#DDEEFF"><b>Associated Tasks:</b></td>
        <td>Classification</td>
        <td bgcolor="#DDEEFF"><b>Missing Values?</b></td>
        <td>Yes</td>
        <td bgcolor="#DDEEFF"><b>Number of Web Hits:</b></td>
        <td>1008052</td>
    </tr>
</table>

## Attribute Information:
Listing of attributes: 
- `class`: >50K, <=50K. 
- `age`: continuous. 
- `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
- `fnlwgt`: continuous. 
- `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
- `education-num`: continuous. 
- `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
- `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
- `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
- `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
- `sex`: Female, Male. 
- `capital-gain`: continuous. 
- `capital-loss`: continuous. 
- `hours-per-week`: continuous. 
- `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
## Problem Statement
> Definition: A computer program is said to learn from experience `E` with respect to some class of tasks `T` and performance measure `P`, if its performance at tasks in `T`, as measured by `P`, improves with experience `E`.
From Mitchell's *Machine Learning*
Here we will call the known information about the adults in this data set to be `E`.
We will define our task `T` to be a binary classification of whether or not the individuals income exceeds $50K/yr.
We will define a performance metric `P` to be an F-score. 
We seek a program that takes in the adult data set and is able to assess whether or not a given adult in the set makes more than $50k/yr as measured by the F-score metric. 
## Solution Statement
As noted, this dataset performs very well with linear models. As such we will be developing our suite of linear models to develop our program.
## Benchmark Model
As a benchmark model, we will simply guess the most dominant class.
## Performance Metric
F-score. `#TODO: Why did we pick this?`
## Project Roadmap
1. Gather and store data

    Link to data: http://archive.ics.uci.edu/ml/machine-learning-databases/adult/
    
``` 
├── README.md
├── adult_practice.ipynb
├── data
│   ├── adult.data.txt
│   └── adult.test.txt
├── doc
│   └── adult.names.txt
└── ipynb
```

```
32561 data/adult.data.txt
16281 data/adult.test.txt
48842 total
```

Data takes up 3.9 MB of disk space. Will store and retrieve as csv.

1. Establish sampling procedure

    Use 90 confidence interval, 1 margin of error on 48842 instances means we should use 5911 points as a 
    representative sample.
    
    We might think about using bootstrapping/repetitive sampling in some places. Maybe taking three 
    representative samples and looking at the mean.

1. EDA - data cleaning

    Fill the nan/missing values.

    #TODO# : drop nan values in test set.

    Check the data types.

1. EDA - statistics
    1. EDA - descriptive
    1. EDA - correlation and distribution - features
    1. EDA - correlation and distribution - target
1. EDA - handle categorical features
1. Establish performance metric 
   - confusion matrix
   - accuracy
   - precision
   - f-score
1. Benchmark Model
1. Standardize Data
1. Skew-Normalize Data
1. Investigate outliers
   - box plot
1. Bias-Variance Tradeoff in sample size
1. Principal Component Analysis
1. Examine Scree Plot
1. Segmentation
1. Feature selection
1. Develop model pipelines
1. Gridsearch model pipelines
## Stretch Goals
1. Explore individual feature relevance
1. Reduce the feature set
1. Explore other data projections:
   - Polynomial expansions