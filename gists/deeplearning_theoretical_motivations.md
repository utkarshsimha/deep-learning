### Introduction
+ Deep Learning - machine learning algorithms based on learning *multiple levels* of represnetaitons
+ Capture high leevl abstractin -> generalize well

### No free lunch theory
+ Lots & lots of data!
+ Verify *flexible* models -> family of functions -> generic
+ prior knowledge to defeat curse of dimensionality

### Curse of dimensionality
+ We're dealing with high demnsional random variables
+ Each dimension can take a lot of values
+ 1000 input features, 2 output classes - 2^1000; way too much to enumerate
+ Not enough data to consider
+ High dimensional space - generalization looks impossible
+ To defeat, figure out how data was generated; cause of what we are observing

### Why classical non-parametric method doesn't work
+ Discretizing space
+ Function is smooth, no difference between 2.2 and 2.3
+ Low dimension it works
+ High dimension - trying to average around
  * Ends up with nothing or a lot
  * can't average locally
+ not # of dimensions - # of variations in the functions

### Theorem
###### Gaussian kernel machines need *k* examples to learn a function that has *2k* zero crossings along the same line

### Theorem
###### For gaussian kernal machine to learn some some maximally varying functions over *d* inputs requires *O(2^d)* examples

### Using smoothness can't defeat curse of dimensionality 
+ Distributed representations/embeddings & feature learning
+ Deep architedcture <- multiple levels of feature learning
+ Non distributed representations : clustering, n-grams, nearest neighbors, decision trees

### The need for distributed representations
+ Say something about compelx functions without that many examples
+ Make assumptions about world that generated the data
+ Regions can grow exponentially with number of params
+ Not every partition is feasible
+ Will work under some conditions
+ Eg. : Imagine input is image of person. 3 features (person is tall/short, male/female, wears glasses/doesn't) - learn about each feature independently
+ Learning features independently - doesn't grow exponentially! Just enough examples for each feature
+ Grows *linearly* with number of features

### The mirage of convexity (disadvantages of NNets)
+ optimization is non-convex
+ exponential # of local minima
+ saddle points
+ high dimension - local minima not critical point, they are saddle points (might have trouble escaping)
+Genearlize from few examples - transfer knowledge from previous learning : Unsupervised learning

[Video link to talk](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/)
