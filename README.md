# Decision Trees for Machine Learning

**Disclaimer:** This repository is a sketchbook learning the background of decision tree algorithms. It is neither clean nor readable. Please direct yourself to [**Chefboost**](https://github.com/serengil/chefboost) repository to have clean one. 

This is the repository of **[Decision Trees for Machine Learning](https://www.udemy.com/course/decision-trees-for-machine-learning/?referralCode=FDC9B836EC6DAA1A663A)** online course published on Udemy. In this course, the following algorithms will be covered. All project is going to be developed on Python (3.6.4), and neither out-of-the-box library nor framework will be used to build decision trees.

1- [ID3](https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/)

2- [C4.5](https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/)

3- [CART (Classification And Regression Trees)](https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/)

4- [Regression Trees (CART for regression)](https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/)

5- [Random Forest](https://sefiks.com/2017/11/19/how-random-forests-can-keep-you-from-decision-tree/)

6- [Gradient Boosting Decision Trees for Regression](https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/)

7- [Gradient Boosting Decision Trees for Classification](https://sefiks.com/2018/10/29/a-step-by-step-gradient-boosting-example-for-classification/)

8- [Adaboost](https://sefiks.com/2018/11/02/a-step-by-step-adaboost-example/)

Just call the [decision.py](/python/decision.py) file to run the program. You might want to change the running algorithm. You just need to set algorithm variable.

```
algorithm = "ID3" #Please set this variable to ID3, C4.5, CART or Regression
```

Moreover, you might want to apply random forest. Please set this to True in this case.

```
enableRandomForest = False
```

Furthermore, you can apply gradient boosting regression trees.

```
enableGradientBoosting = True
```

Besides, adaptive boosting is allowed to run

```
enableAdaboost = True
```

Finally, you can change the data set to build different decision trees. Just pass the file name, and its column names if it does not exist.

```
df = pd.read_csv("car.data"
  #column names can either be defined in the source file or names parameter in read_csv command
  ,names=["buying","maint","doors","persons","lug_boot","safety","Decision"] 
)
```

# Prerequisites

Pandas and numpy python libraries are used to load data sets in this repository. You might run the following commands to install these packages if you are going to use them first time.

```
pip install pandas
pip install numpy
```

# Updates

To keep yourself up-to-date you might check posts in my blog about [decision trees](https://sefiks.com/tag/decision-tree/) 

# License

This repository is licensed under the MIT License - see [LICENSE](https://github.com/serengil/decision-trees-for-ml/blob/master/LICENSE) for more details.
