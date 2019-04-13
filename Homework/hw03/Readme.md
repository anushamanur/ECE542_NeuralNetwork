# Homework 3 - MNIST handwritten digit recognition 
- The goal of this homework is to make you familiar with applying some machine learning algorithms using Python.
- You will be asked to use the logistic regression and random forest methods for handwritten digit recognition on the MNIST dataset.
- This is an **individual assignment**.

## Recommend Environment
- [Python3](https://www.python.org/download/releases/3.0/)
- [scikit-learn](http://scikit-learn.org/stable/) (You can use other machine learning libraries)

## Dataset
- Download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/). Do not use sklearn built-in functions to load the dataset, as it uses different training and testing set as the website. During evaluation, we will assume the results corresponds to the testing set from the website.
- Use only the training images for training (You can split it into training set and validation set).
- The test set will be used for evaluation.

## Assignment Description
- Apply logistic regression and random forest on MNIST. 
- For both approaches, use the raw pixel values as features.

## Deliverable
- Source code
- Results on testing set of both approaches saved in two separate `csv` files with file names `lr.csv` and `rf.cvs` for logistic regression and random forest, respectively. A sample submission result can be found [here](../../docs/samples/sample_submission/).
- Put the source code, result csv's and a `name.csv` file containing your unityID in a folder **`hw03`**, zip the folder and submit it to moodle. [Here](../../docs/samples/sample_submission/) is a sample submission.
- **Use the testing set from the provided link and do not shuffle the testing data for submission.**
- Make sure to follow this format or you may lose all the credits. Please read the instructions in the sample submission folder carefully.
- **Please do not submit the data and the trained models.**

## Evaluation for Credits
- We will evaluate the homework automatically. So please make sure to follow the submission format as mentioned above.
- The evaluation script can be found [here](../../src/eval/eval.py). Please follow the [instruction](../../src/eval/) to evaluate your result before submission. Make sure you can get correct output using the script or you may lose all the credits.

## Notes
- It may be easy for you to find the code and the homework can be done by just copy and paste. It is ok to read others' code before implementing your own code. Actually, it is recommended. After that, from my experience, one of the best ways to learn is to write the entire code by yourself, so that you can fully understand what is happening here.
- Also you can reach the 100% accuracy by simply overfitting the testing set or just manually generate the result `csv` files. But it does not make any sense to get the top performance in the class for this project and you can learn nothing.

## Useful Links
- [Logistic Regression on MNIST](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)
- [Random Forest on MNIST](https://www.kaggle.com/issatingzon/mnist-with-random-forests)
- [MNIST recognition using different approaches](http://brianfarris.me/static/digit_recognizer.html)
