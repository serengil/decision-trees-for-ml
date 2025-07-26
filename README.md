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

# Updates

To keep yourself up-to-date you might check posts in my blog about [decision trees](https://sefiks.com/tag/decision-tree/) 

# License

This repository is licensed under the MIT License - see [LICENSE](https://github.com/serengil/decision-trees-for-ml/blob/master/LICENSE) for more details.

# Example Usage

Here is a step-by-step example of how to run the decision tree script:

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Run the main script:**
   ```
   python python/decision.py
   ```

3. **Sample input:**
   The script uses the dataset in `dataset/golf.txt` by default. You can change the dataset by editing the `df = pd.read_csv(...)` line in `python/decision.py`.

   Example of `golf.txt` (first few lines):
   ```
   Outlook,Temperature,Humidity,Wind,Decision
   Sunny,Hot,High,Weak,No
   Sunny,Hot,High,Strong,No
   Overcast,Hot,High,Weak,Yes
   Rain,Mild,High,Weak,Yes
   Rain,Cool,Normal,Weak,Yes
   Rain,Cool,Normal,Strong,No
   Overcast,Cool,Normal,Strong,Yes
   Sunny,Mild,High,Weak,No
   Sunny,Cool,Normal,Weak,Yes
   Rain,Mild,Normal,Weak,Yes
   Sunny,Mild,Normal,Strong,Yes
   Overcast,Mild,High,Strong,Yes
   Overcast,Hot,Normal,Weak,Yes
   Rain,Mild,High,Strong,No
   ```

4. **Sample output:**
   After running the script, a file named `rules.py` will be generated in the `python/` directory. This file contains the decision rules as a Python function. You will also see console output similar to:
   ```
   C4.5  tree is going to be built...
   finished in  0.02  seconds
   ```

   Example of generated rule (in `python/rules.py`):
   ```python
   def findDecision(obj):
      if obj[0] == 'Overcast':
         return 'Yes'
      if obj[0] == 'Rain':
         if obj[3] == 'Weak':
            return 'Yes'
         if obj[3] == 'Strong':
            return 'No'
      if obj[0] == 'Sunny':
         if obj[2] == 'High':
            return 'No'
         if obj[2] == 'Normal':
            return 'Yes'
   ```

5. **Changing the algorithm or dataset:**
   Edit the variables at the top of `python/decision.py` to select a different algorithm or dataset.

# Output Files and Results

## Generated Files

When you run the decision tree script, it generates Python files containing the decision rules. The exact filename depends on the algorithm and settings used:

- **Standard decision tree:** `rules.py`
- **Random Forest:** `rule_0.py`, `rule_1.py`, `rule_2.py`, etc. (one file per tree)
- **Gradient Boosting:** `rules0.py`, `rules1.py`, `rules2.py`, etc. (one file per iteration)
- **Adaboost:** `rules_0.py`, `rules_1.py`, `rules_2.py`, etc. (one file per round)

## Understanding the Output

### Decision Rules Format

The generated files contain a Python function called `findDecision(obj)` that implements the decision tree as a series of if-else statements. For example:

```python
def findDecision(obj):
   if obj[0] == 'Sunny':
      if obj[2] == 'High':
         return 'No'
      if obj[2] == 'Normal':
         return 'Yes'
   if obj[0] == 'Rain':
      if obj[3] == 'Weak':
         return 'Yes'
      if obj[3] == 'Strong':
         return 'No'
   if obj[0] == 'Overcast':
      return 'Yes'
```

### How to Use the Generated Rules

1. **Import the rules file:**
   ```python
   import rules  # or whatever the generated filename is
   ```

2. **Make predictions:**
   ```python
   # Create a feature vector (in the same order as your dataset columns)
   features = ['Sunny', 'Hot', 'High', 'Weak']  # Example for golf dataset
   
   # Get prediction
   prediction = rules.findDecision(features)
   print(f"Prediction: {prediction}")
   ```

### Feature Vector Format

The `obj` parameter in `findDecision(obj)` is a list where each element corresponds to a feature column in your dataset, in the same order:

- `obj[0]` = First feature column
- `obj[1]` = Second feature column
- `obj[2]` = Third feature column
- And so on...

**Example for the golf dataset:**
- `obj[0]` = Outlook (Sunny, Overcast, Rain)
- `obj[1]` = Temperature (Hot, Mild, Cool)
- `obj[2]` = Humidity (High, Normal)
- `obj[3]` = Wind (Weak, Strong)

### Console Output

When you run the script, you'll see output like:
```
C4.5  tree is going to be built...
finished in  0.022043228149414062  seconds
```

This shows:
- Which algorithm was used
- How long the tree building process took

### Multiple Output Files

For ensemble methods (Random Forest, Gradient Boosting, Adaboost), multiple rule files are generated. Each file represents one tree or iteration in the ensemble. To use these:

1. **Random Forest:** Use all generated files and take a majority vote
2. **Gradient Boosting:** Use files sequentially, each building on the previous
3. **Adaboost:** Use all files with their respective weights

## Troubleshooting

- **No output files generated:** Check that `dump_to_console = False` in the script
- **Wrong predictions:** Ensure your feature vector matches the dataset column order
- **Import errors:** Make sure the generated rules file is in your Python path

# Running with Command-Line Arguments

You can now configure the script without editing the code by using command-line arguments:

```
python python/decision.py [OPTIONS]
```

**Available options:**
- `--algorithm` (`ID3`, `C4.5`, `CART`, `Regression`) — Algorithm to use (default: `C4.5`)
- `--dataset` — Path to dataset file (default: `dataset/golf.txt`)
- `--random-forest` — Enable Random Forest
- `--num-trees` — Number of trees for Random Forest (default: 3)
- `--multitasking` — Enable multitasking for Random Forest
- `--adaboost` — Enable Adaboost
- `--gradient-boosting` — Enable Gradient Boosting
- `--epochs` — Number of epochs for boosting (default: 10)
- `--learning-rate` — Learning rate for boosting (default: 1)
- `--dump-to-console` — Print rules to console instead of file

**Example:**
```
python python/decision.py --algorithm ID3 --dataset dataset/golf.txt --dump-to-console
```

# Getting Started

## Prerequisites
- Python 3.6 or higher

## Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/decision-trees-for-ml.git
   cd decision-trees-for-ml
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

# How to Run

You can run the main script with various options using command-line arguments.

**Basic usage:**
```
python python/decision.py
```

**Specify algorithm and dataset:**
```
python python/decision.py --algorithm ID3 --dataset dataset/golf.txt
```

**Enable Random Forest:**
```
python python/decision.py --random-forest --num-trees 5
```

**Print rules to console:**
```
python python/decision.py --dump-to-console
```

For a full list of options, see the [Running with Command-Line Arguments](#running-with-command-line-arguments) section below.

# Expected Output

- The script will print the algorithm being used and the time taken to build the tree.
- By default, it will generate a Python file (e.g., `rules.py`) containing the decision rules.
- If `--dump-to-console` is used, rules will be printed to the terminal instead of being saved to a file.

See the [Output Files and Results](#output-files-and-results) section for more details.
