# Decision Trees - Complete Simple Guide

## 1. Example 1: Tennis Playing Decision

**Scenario:** Should I play tennis today?

**Features:** Weather, Temperature, Humidity, Wind


| Weather | Temperature | Humidity | Wind | Play Tennis? |
| :-- | :-- | :-- | :-- | :-- |
| Sunny | Hot | High | Weak | No |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | Yes |
| Rain | Cool | Normal | Strong | No |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | No |
| Sunny | Cool | Normal | Weak | Yes |
| Rain | Mild | Normal | Weak | Yes |

**Decision Tree Result:**[^1]

```
Weather = ?
├── Sunny → Humidity = ?
│   ├── High → No
│   └── Normal → Yes
├── Overcast → Yes
└── Rain → Wind = ?
    ├── Weak → Yes
    └── Strong → No
```


## 2. Example 2: Customer Purchase Decision

**Scenario:** Will a customer buy our product?

**Features:** Age, Income, Student Status, Credit Rating


| Age | Income | Student | Credit | Buys Computer? |
| :-- | :-- | :-- | :-- | :-- |
| ≤30 | High | No | Fair | No |
| ≤30 | High | No | Excellent | No |
| 31-40 | High | No | Fair | Yes |
| >40 | Medium | No | Fair | Yes |
| >40 | Low | Yes | Fair | Yes |
| >40 | Low | Yes | Excellent | No |
| 31-40 | Low | Yes | Excellent | Yes |
| ≤30 | Medium | No | Fair | No |
| ≤30 | Low | Yes | Fair | Yes |
| >40 | Medium | Yes | Fair | Yes |

**Decision Tree Result:**[^1]

```
Age = ?
├── ≤30 → Student = ?
│   ├── Yes → Yes
│   └── No → No
├── 31-40 → Yes
└── >40 → Credit = ?
    ├── Excellent → No
    └── Fair → Yes
```


## 3. What If We Have Numerical Data?

When dealing with **continuous numerical features** like age, income, or temperature, decision trees use **binary splits**:[^2]

**Example:** Age feature with values

**Process:**[^2]

1. **Sort values:**
2. **Find midpoints:** [27.5, 32.5, 37.5, 42.5, 47.5]
3. **Try each split:** Age ≤ 27.5? Age ≤ 32.5? etc.
4. **Choose best split** based on information gain or Gini impurity

**Result:** Age ≤ 35?

- Yes → Further splits or prediction
- No → Further splits or prediction


## 4. Geometric Intuition

**Decision trees create rectangular decision boundaries** in feature space:[^3][^4]

**2D Example (Height vs Weight for Gender Classification):**

```
Weight
  ↑
  |     |  Male   |
  | ----+---------|
  |     | Female  |
  |     |         |
  +-----|---------|→ Height
        threshold
```

**Key Properties:**[^3]

- **Axis-parallel splits:** Lines parallel to feature axes
- **Rectangular regions:** Each leaf node = rectangle in feature space
- **Non-overlapping:** Each point belongs to exactly one region
- **Hierarchical:** Complex boundaries built from simple splits


## 5. Pseudo Code

**Decision Tree Learning Algorithm:**[^5][^6]

```python
def DecisionTreeLearner(examples, features):
    # Base Cases
    if all_examples_same_class(examples):
        return class_label
    
    if no_features_left(features):
        return majority_class(examples)
    
    if no_examples_left(examples):
        return majority_class_from_parent()
    
    # Recursive Case
    best_feature = choose_best_feature(examples, features)
    tree = create_node(best_feature)
    
    for each value v in best_feature:
        subset = examples_with_feature_value(examples, best_feature, v)
        remaining_features = features - {best_feature}
        subtree = DecisionTreeLearner(subset, remaining_features)
        add_branch(tree, v, subtree)
    
    return tree

def choose_best_feature(examples, features):
    best_gain = -1
    best_feature = None
    
    for feature in features:
        gain = information_gain(examples, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return best_feature
```


## 6. Conclusion

**Decision Trees are powerful because they:**[^7]

- **Mimic human reasoning:** Use if-then-else logic
- **Handle mixed data:** Both categorical and numerical features
- **Provide interpretability:** Easy to understand and explain
- **Require minimal preprocessing:** No scaling or normalization needed
- **Handle missing values:** Can work with incomplete data

**They're perfect for:** Credit approval, medical diagnosis, customer segmentation, and any domain requiring explainable AI.

## 7. Terminology

| Term | Definition | Example |
| :-- | :-- | :-- |
| **Root Node** | Starting point of the tree | "Weather = ?" |
| **Internal Node** | Decision point with test condition | "Humidity = ?" |
| **Leaf Node** | Final prediction/outcome | "Play Tennis = Yes" |
| **Branch** | Connection between nodes | Weather → Sunny |
| **Depth** | Levels from root to deepest leaf | 3 levels deep |
| **Pruning** | Removing branches to prevent overfitting | Cut complex subtrees |
| **Splitting** | Dividing data based on feature test | Age ≤ 30? |

## 8. Unanswered Questions

**Common challenges that need further research:**[^7]

- **How to prevent overfitting?** → Pruning techniques, max depth limits
- **How to handle imbalanced data?** → Class weights, cost-sensitive learning
- **What about missing values?** → Surrogate splits, imputation methods
- **How to handle large datasets?** → Random forests, gradient boosting
- **How to make trees more stable?** → Ensemble methods
- **How to handle high-dimensional data?** → Feature selection, random subspaces


## 9. Advantages / Disadvantages

### **Advantages ✅**

| Advantage | Explanation | Real Benefit |
| :-- | :-- | :-- |
| **Interpretability** | Easy to understand decision path | Explain predictions to stakeholders |
| **No Assumptions** | Non-parametric, handles any distribution | Works with messy real-world data |
| **Mixed Data Types** | Handles categorical and numerical | No preprocessing needed |
| **Feature Selection** | Automatically selects important features | Reduces dimensionality |
| **Missing Values** | Can handle incomplete data | Robust to data quality issues |

### **Disadvantages ❌**

| Disadvantage | Problem | Impact |
| :-- | :-- | :-- |
| **Overfitting** | Creates overly complex trees | Poor generalization |
| **Instability** | Small data changes → different tree | Unreliable predictions |
| **Bias** | Favors features with more levels | Unfair feature selection |
| **Linear Boundaries** | Only axis-parallel splits | Can't capture diagonal patterns |
| **Class Imbalance** | Biased toward majority class | Poor minority class performance |

## 10. How Do Decision Trees Work? Entropy?

**Core Working Principle:**[^8][^7]

1. **Start with mixed data** (impure root node)
2. **Find best feature** to split on (highest information gain)
3. **Split data** into subsets based on feature values
4. **Repeat recursively** until subsets are pure or stopping criteria met
5. **Make predictions** by following path from root to leaf

**Entropy drives the process** by measuring impurity at each node. The algorithm always chooses splits that **maximize information gain** (reduce entropy most).

## 11. What is Entropy?

**Entropy** measures **disorder** or **uncertainty** in a dataset:[^9][^8]

**Intuitive Understanding:**

- **Low entropy (0):** All examples belong to same class → Pure, organized
- **High entropy (1):** Examples evenly distributed across classes → Mixed, chaotic

**Real-world Analogy:**

- **Clean room (low entropy):** Everything in its place
- **Messy room (high entropy):** Items scattered everywhere

**In Decision Trees:**[^9]

- **Pure leaf node:** Entropy = 0 (all "Yes" or all "No")
- **Mixed node:** Entropy > 0 (50% "Yes", 50% "No" = maximum entropy)


## 12. How to Calculate Entropy?

**Entropy Formula:**[^9]

$$
Entropy(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

Where:

- **S** = dataset
- **p_i** = probability of class i
- **n** = number of classes

**Step-by-Step Example:**[^9]

**Dataset:** 9 "Yes", 6 "No" (total 15 examples)

1. **Calculate probabilities:**
    - P(Yes) = 9/15 = 0.6
    - P(No) = 6/15 = 0.4
2. **Apply formula:**

```
Entropy = -(0.6 × log₂(0.6)) - (0.4 × log₂(0.4))
Entropy = -(0.6 × -0.737) - (0.4 × -1.322)
Entropy = 0.442 + 0.529 = 0.971
```


**Special Cases:**

- All same class: Entropy = 0
- 50-50 split: Entropy = 1 (maximum)


## 13. Observations

**Key Patterns in Decision Tree Behavior:**[^8]

1. **Entropy decreases** as we move from root to leaves
2. **Information gain** is always positive (entropy reduction)
3. **Pure nodes** (entropy = 0) become leaf nodes
4. **Greedy algorithm** makes locally optimal choices
5. **Tree depth** correlates with complexity and overfitting risk
6. **Feature importance** determined by position in tree (higher = more important)

## 14. Entropy vs Probability

| Aspect | Entropy | Probability |
| :-- | :-- | :-- |
| **Measures** | Uncertainty/disorder in dataset | Likelihood of specific outcome |
| **Range** | 0 to log₂(n) | 0 to 1 |
| **Purpose** | Choose best splits | Make predictions |
| **Formula** | -Σ p_i log₂(p_i) | Count(class)/Total |
| **Interpretation** | Lower = more organized | Higher = more likely |

**Example:**[^9]

- **Probability:** 60% chance of "Yes"
- **Entropy:** 0.971 bits of uncertainty in the decision


## 15. Information Gain

**Information Gain** measures how much entropy decreases after a split:[^10][^8]

**Formula:**

$$
IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \times Entropy(S_v)
$$

Where:

- **S** = original dataset
- **A** = attribute/feature
- **S_v** = subset where attribute A has value v

**Step-by-Step Example:**[^8]

**Original dataset:** Entropy = 0.971

**Split on "Weather":**

- Sunny (5 examples): 2 Yes, 3 No → Entropy = 0.971
- Overcast (4 examples): 4 Yes, 0 No → Entropy = 0.0
- Rain (6 examples): 3 Yes, 3 No → Entropy = 1.0

**Information Gain:**

```
IG = 0.971 - [(5/15)×0.971 + (4/15)×0.0 + (6/15)×1.0]
IG = 0.971 - [0.324 + 0.0 + 0.4] = 0.247
```


## 16. Gini Impurity

**Alternative to entropy** for measuring node impurity:[^11][^12]

**Formula:**

$$
Gini(S) = 1 - \sum_{i=1}^{n} p_i^2
$$

**Example:**[^12]
Dataset: 9 "Yes", 6 "No"

- P(Yes) = 0.6, P(No) = 0.4
- Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48

**Gini vs Entropy:**


| Metric | Range | Computation | Sensitivity |
| :-- | :-- | :-- | :-- |
| **Gini** | 0 to 0.5 | Faster (no logarithms) | Less sensitive to changes |
| **Entropy** | 0 to 1 | Slower (logarithmic) | More sensitive to changes |

**Both lead to similar trees** in practice![^13]

## 17. Handling Numerical Data

**For continuous features, decision trees use binary splits:**[^2]

### **Algorithm:**

1. **Sort feature values:**
2. **Generate candidate splits:** [24, 26.5, 29, 32.5, 37.5]
3. **Evaluate each split:** Calculate information gain for X ≤ threshold
4. **Choose best threshold:** Highest information gain
5. **Create binary split:** X ≤ best_threshold?

### **Example: Age Feature**

```
Original: [23(No), 25(No), 28(Yes), 30(Yes), 35(Yes), 40(No)]

Try Age ≤ 26.5:
├── Left: [23(No), 25(No)] → All No, Entropy = 0
└── Right: [28(Yes), 30(Yes), 35(Yes), 40(No)] → Mixed, Entropy = 0.811

Information Gain = Original_Entropy - Weighted_Average_Entropy
```


### **Optimization:**

- **Only consider midpoints** between different class labels
- **Use efficient sorting** algorithms
- **Prune similar thresholds** to reduce computation

**This allows decision trees to handle any mix of categorical and numerical features seamlessly!**[^2]

<div style="text-align: center">⁂</div>

[^1]: https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/

[^2]: https://scikit-learn.org/stable/modules/tree.html

[^3]: https://pub.aimind.so/demystifying-decision-trees-an-intuitive-guide-to-understanding-and-applying-cart-algorithm-for-c778a56924cc

[^4]: https://www.kaggle.com/getting-started/215795

[^5]: https://www.cs.toronto.edu/~axgao/cs486686_f21/lecture_notes/Lecture_07_on_Decision_Trees.pdf

[^6]: https://www.geeksforgeeks.org/machine-learning/decision-tree-implementation-python/

[^7]: https://www.ibm.com/think/topics/decision-trees

[^8]: https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/

[^9]: https://www.geeksforgeeks.org/data-science/how-to-calculate-entropy-in-decision-tree/

[^10]: https://en.wikipedia.org/wiki/Information_gain_(decision_tree)

[^11]: https://www.niser.ac.in/~smishra/teach/cs460/2020/lectures/lec11/

[^12]: https://victorzhou.com/blog/gini-impurity/

[^13]: https://www.geeksforgeeks.org/machine-learning/gini-impurity-and-entropy-in-decision-tree-ml/

[^14]: https://www.xoriant.com/blog/decision-trees-for-classification-a-machine-learning-algorithm

[^15]: https://venngage.com/blog/what-is-a-decision-tree/

[^16]: https://www.wework.com/ideas/professional-development/business-solutions/decision-trees-definition-analysis-and-examples

[^17]: https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(Fleming)/05:_The_Second_Law/5.04:_Calculating_Entropy_Changes

[^18]: https://www.atlassian.com/work-management/project-management/decision-tree

[^19]: https://www.chemguide.co.uk/physical/entropy/entropychange.html

[^20]: https://www.sciencedirect.com/topics/computer-science/information-gain

