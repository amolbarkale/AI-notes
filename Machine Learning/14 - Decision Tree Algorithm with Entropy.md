# Decision Tree Algorithm with Entropy - Step-by-Step Construction

## 1. How Decision Trees Choose the Root Node

The **root node** is selected by finding the feature that **best separates** the data - the one with the **highest information gain**.

### **Root Node Selection Process:**

**Step 1: Calculate Initial Entropy**
Calculate entropy of the entire dataset before any splits.

**Step 2: Test Each Feature**
For every possible feature, calculate what the entropy would be after splitting on that feature.

**Step 3: Calculate Information Gain**
For each feature: Information Gain = Original Entropy - Weighted Average Entropy after split

**Step 4: Select Best Feature**
The feature with **highest information gain** becomes the root node.

## 2. Complete Example: Student Admission Dataset

Let's build a decision tree to predict student admission based on GPA, Test Score, and Activities.

### **Dataset:**

| Student | GPA | Test Score | Activities | Admitted? |
| :-- | :-- | :-- | :-- | :-- |
| 1 | High | High | Yes | Yes |
| 2 | High | High | No | Yes |
| 3 | High | Low | Yes | Yes |
| 4 | High | Low | No | No |
| 5 | Low | High | Yes | Yes |
| 6 | Low | High | No | No |
| 7 | Low | Low | Yes | No |
| 8 | Low | Low | No | No |

**Total:** 4 Yes, 4 No

## 3. Step 1: Calculate Root Entropy

**Original Dataset:** 4 "Yes", 4 "No" out of 8 students

**Entropy Calculation:**

```
P(Yes) = 4/8 = 0.5
P(No) = 4/8 = 0.5

Entropy(Root) = -(0.5 × log₂(0.5)) - (0.5 × log₂(0.5))
              = -(0.5 × -1) - (0.5 × -1)
              = 0.5 + 0.5 = 1.0
```

**Root entropy = 1.0** (maximum uncertainty - perfect 50-50 split)

## 4. Step 2: Test Each Feature for Root Node

### **Option 1: Split on GPA**

**High GPA:** Students 1, 2, 3, 4 → 3 Yes, 1 No

```
P(Yes) = 3/4 = 0.75, P(No) = 1/4 = 0.25
Entropy(High GPA) = -(0.75×log₂(0.75)) - (0.25×log₂(0.25))
                  = -(0.75×-0.415) - (0.25×-2)
                  = 0.311 + 0.5 = 0.811
```

**Low GPA:** Students 5, 6, 7, 8 → 1 Yes, 3 No

```
P(Yes) = 1/4 = 0.25, P(No) = 3/4 = 0.75
Entropy(Low GPA) = -(0.25×log₂(0.25)) - (0.75×log₂(0.75))
                 = 0.5 + 0.311 = 0.811
```

**Weighted Average Entropy:**

```
Entropy_after_GPA = (4/8)×0.811 + (4/8)×0.811 = 0.811
```

**Information Gain for GPA:**

```
IG(GPA) = 1.0 - 0.811 = 0.189
```


### **Option 2: Split on Test Score**

**High Test Score:** Students 1, 2, 5, 6 → 2 Yes, 2 No

```
Entropy(High Test) = -(0.5×log₂(0.5)) - (0.5×log₂(0.5)) = 1.0
```

**Low Test Score:** Students 3, 4, 7, 8 → 1 Yes, 3 No

```
Entropy(Low Test) = -(0.25×log₂(0.25)) - (0.75×log₂(0.75)) = 0.811
```

**Information Gain for Test Score:**

```
IG(Test Score) = 1.0 - [(4/8)×1.0 + (4/8)×0.811] = 1.0 - 0.906 = 0.094
```


### **Option 3: Split on Activities**

**Has Activities:** Students 1, 3, 5, 7 → 3 Yes, 1 No

```
Entropy(Has Activities) = 0.811
```

**No Activities:** Students 2, 4, 6, 8 → 1 Yes, 3 No

```
Entropy(No Activities) = 0.811
```

**Information Gain for Activities:**

```
IG(Activities) = 1.0 - 0.811 = 0.189
```


## 5. Step 3: Select Root Node

**Information Gain Comparison:**

- GPA: 0.189
- Test Score: 0.094
- Activities: 0.189

**Result:** Both GPA and Activities tie with highest information gain (0.189). Let's choose **GPA** as root node.

**Initial Tree:**

```
GPA = ?
├── High (3 Yes, 1 No)
└── Low (1 Yes, 3 No)
```


## 6. Step 4: Recursive Splitting - Left Branch (High GPA)

**Subset:** Students with High GPA = {1, 2, 3, 4} → 3 Yes, 1 No

**Current Entropy:**

```
Entropy(High GPA) = 0.811
```

**Test remaining features: Test Score and Activities**

### **Split High GPA group by Test Score:**

**High GPA + High Test:** Students 1, 2 → 2 Yes, 0 No

```
Entropy = 0 (pure node)
```

**High GPA + Low Test:** Students 3, 4 → 1 Yes, 1 No

```
Entropy = 1.0
```

**Information Gain:**

```
IG = 0.811 - [(2/4)×0 + (2/4)×1.0] = 0.811 - 0.5 = 0.311
```


### **Split High GPA group by Activities:**

**High GPA + Activities:** Students 1, 3 → 2 Yes, 0 No → Entropy = 0
**High GPA + No Activities:** Students 2, 4 → 1 Yes, 1 No → Entropy = 1.0

**Information Gain:** 0.311 (same as Test Score)

**Choose Test Score** (arbitrary choice since they tie).

## 7. Step 5: Continue Recursive Process

**Updated Tree after processing High GPA branch:**

```
GPA = ?
├── High → Test Score = ?
│   ├── High → Yes (Students 1,2: 2 Yes, 0 No)
│   └── Low → Activities = ?
│       ├── Yes → Yes (Student 3: 1 Yes, 0 No)
│       └── No → No (Student 4: 0 Yes, 1 No)
└── Low (Students 5,6,7,8: 1 Yes, 3 No)
```


## 8. Step 6: Process Right Branch (Low GPA)

**Subset:** Students with Low GPA = {5, 6, 7, 8} → 1 Yes, 3 No

**Test remaining features for Low GPA group:**

### **Split by Test Score:**

**Low GPA + High Test:** Students 5, 6 → 1 Yes, 1 No → Entropy = 1.0
**Low GPA + Low Test:** Students 7, 8 → 0 Yes, 2 No → Entropy = 0

**Information Gain:** 0.811 - [(2/4)×1.0 + (2/4)×0] = 0.311

### **Split by Activities:**

**Low GPA + Activities:** Students 5, 7 → 1 Yes, 1 No → Entropy = 1.0
**Low GPA + No Activities:** Students 6, 8 → 0 Yes, 2 No → Entropy = 0

**Information Gain:** 0.311 (tie again)

**Choose Test Score** for consistency.

## 9. Final Decision Tree

```
GPA = ?
├── High → Test Score = ?
│   ├── High → Yes
│   └── Low → Activities = ?
│       ├── Yes → Yes
│       └── No → No
└── Low → Test Score = ?
    ├── High → Activities = ?
    │   ├── Yes → Yes
    │   └── No → No
    └── Low → No
```


## 10. Key Insights from This Process

### **Root Selection Strategy:**

1. **Entropy measures disorder** - we want to reduce it maximally
2. **Information gain quantifies improvement** - higher is better
3. **Greedy approach** - always pick locally best feature
4. **Recursive application** - repeat process for each subset

### **Why This Works:**

- **Maximizes purity** at each step
- **Minimizes uncertainty** progressively
- **Creates interpretable rules** naturally
- **Handles complex interactions** through hierarchy


### **Stopping Criteria:**

- **Pure nodes** (entropy = 0) become leaves
- **No more features** to split on
- **Minimum samples** threshold reached
- **Maximum depth** limit hit

This systematic approach ensures that decision trees always make the **most informative split** at each step, leading to effective classification with clear, interpretable decision paths!

