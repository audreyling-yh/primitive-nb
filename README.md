# primitive-nb
A comparison of two feature engineering methods to predict nationality from name using a Naive Bayes classifier. As a fun challenge, the codes were written with minimal packages (ie. no pandas, scikit-learn, etc). The Naive Bayes classifier is a model based upon counting and conditional probabilities. The codes do this from scratch. What is the point of this? I don't know, but it was a good exercise.

### Data
Predictor: Name
Predicted: Nationality
- Chinese
- Korean
- Japanese

### Features
1. 1st Approximation: Create a new dummy variable to indicate the presence of each alphabet in a name.
2. 2 Consecutive letters: Create 27*27 dummy variables to indicate the presence of every combination of 2 alphabets (eg. 'aa', 'ab', ... 'zy', 'zz') inclusive of '\_a'...'\_z' to indicate first letter of the name, and 'a\_'...'z\_' to indicate last letter of the name.

The scripts for each method run independently of each other. Output metrics can be compared to determine the method with better validation accuracy.


