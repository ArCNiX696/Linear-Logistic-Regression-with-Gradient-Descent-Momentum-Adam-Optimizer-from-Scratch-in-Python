📊 Linear and Logistic Regression with Gradient Descent, Momentum, and Adam Optimizer from Scratch in Python

Welcome to the repository for our Linear and Logistic Regression project! This project involves implementing Linear and Logistic Regression models from scratch and optimizing them using Gradient Descent, Gradient Descent with Momentum, and the Adam Optimizer. Below, you'll find an overview of the project, key insights, and instructions to get started.

📘 Project Overview
This project explores the implementation and optimization of Linear and Logistic Regression models using different optimization techniques. The primary focus is on:

Gradient Descent
Gradient Descent with Momentum
Adam Optimizer
The datasets used include a self-created dataset for student performance scores and the Titanic survival dataset.

🧪 Experiments and Results
Linear Regression
Dataset: Student performance scores.

Dependent Variable (y): Performance index (Student scores).
Independent Variable (x): Previous scores (Student previous scores).
Preprocessing: No normalization required as the scores range from 0 to 100.
Sample Size: Reduced from 1000 to 500 samples.
Optimization Techniques:

Gradient Descent with Momentum (GDM):
Learning rate: 0.001
Momentum: 0.9
Tolerance: 1e-6
Epochs: 1000
Adam Optimizer:
Learning rate: 0.001
Momentum: 0.9
Tolerance: 1e-6
Epochs: 1000
Results:

GDM: Struggled to fit the data well, getting trapped in a local minimum.
Adam: Achieved better optimization, lower error, and more accurate predictions.
Logistic Regression
Dataset: Titanic survival data.

Dependent Variable (y): Survival (1 for survived, 0 for not survived).
Independent Variables (X): Passenger information (gender, age, fare, etc.).
Preprocessing:
Removed rows with null values.
Normalized features 'Age' and 'Fare'.
Optimization:

Adam Optimizer:
Learning rate: 0.001
Tolerance: 1e-6
Epochs: 1000
Results:

Binary Cross-Entropy Loss: Minimized below 0.45.
Confusion Matrix: High accuracy in classification.
Accuracy: Achieved 98% accuracy in predictions.
🚀 Getting Started
To get started with the project, follow these steps:

Clone the repository:

git clone 
https://github.com/ArCNiX696/Linear-Logistic-Regression-with-Gradient-Descent-Momentum-Adam-Optimizer-from-Scratch-in-Python.git
cd Linear-Logistic-Regression-with-Gradient-Descent-Momentum-Adam-Optimizer-from-Scratch-in-Python

Run the scripts:
Execute the scripts for Linear and Logistic Regression to observe their performance on the provided datasets.

🎨 Visual Insights
The project includes various visualizations such as scatter plots, confusion matrices, and prediction result tables to illustrate the impact of different optimization techniques on model performance.

📌 Conclusion
This study demonstrates the implementation and effectiveness of different optimization techniques for Linear and Logistic Regression models. The Adam optimizer showed superior performance in this case, providing valuable insights for machine learning practitioners.

Feel free to explore the code, run the experiments, and gain deeper insights into the fascinating world of regression models and optimization techniques. Your feedback and contributions are welcome!

Happy coding! 🚀📊🧠

References
Various academic papers on Gradient Descent, Momentum, and Adam optimizers.
If you have any questions or need further assistance, don't hesitate to reach out via email or open an issue on GitHub.

Happy learning!






