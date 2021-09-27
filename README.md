# classification of speech recordings
 
 ![GitHub last commit](https://img.shields.io/github/last-commit/rrtty0/classification_of_speech_recordings?style=plastic)
 ![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/rrtty0/classification_of_speech_recordings?style=plastic)
 ![GitHub contributors](https://img.shields.io/github/contributors/rrtty0/classification_of_speech_recordings?style=plastic)
 ![GitHub repo size](https://img.shields.io/github/repo-size/rrtty0/classification_of_speech_recordings?style=plastic)

- [How To Install](#anc1)
- [How To Use](#anc2)
- [How To Contribute](#anc3)
- [How It Work](#anc4)
- [License](#anc5)

---
Classifier of speech recordings by emotionality.</br>
Implemented by [Python 3.8](https://www.python.org/downloads/).

<a id="anc1"></a>

## How To Install
- The sources of project can be downloaded from the [Github repo](https://github.com/rrtty0/classification_of_speech_recordings.git).

* You can either clone the public repository:
```
        $ git clone https://github.com/rrtty0/classification_of_speech_recordings.git 
```
<a id="anc2"></a>

## How To Use

To use this project you need:
- _Open_ root-folder with this project at your local computer
- _Run_ file [classification_of_speech_recordings.py](./classification_of_speech_recordings.py):
```
        $ python classification_of_speech_recordings.py
```

<a id="anc3"></a>

## How To Contribute
1. _Clone repo_ and _create a new branch_:
```
        $ git clone https://github.com/rrtty0/puzzle_8.git
        $ git branch name_for_new_branch
        $ git checkout name_for_new_branch
```
2. _Make changes_ and _test_
3. _Submit Pull Request_ with comprehensive description of changes

<a id="anc4"></a>

## How It Work

 - Folders ['1'](./train_data/1/), ['2'](./train_data/2/), ['3'](./train_data/3/), ['4'](./train_data/4/), ['5'](./train_data/5/) contain training data. These are speech recordings, divided into 5 emotional classes - each class in its own folder under a conditional number.
 All records are approximately the same in length, approximately 850 records in total. Each recording is one and the same text, read by different speakers with different
 emotions. Folder ['test_data'](./test_data/) contains a set of test speech recordings, the classification of which should occur based on the trained classifier on the training data. the
 classifier is implemented in several forms and can implement the following classification methods:
    * [k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
    * [Decision tree](https://en.wikipedia.org/wiki/Decision_tree)
    * [Support-vector machine](https://en.wikipedia.org/wiki/Support-vector_machine)
    * [Random forest](https://en.wikipedia.org/wiki/Random_forest)
    * [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
 
 - The result of the program is the prediction of emotional classes of speech recordings for the test sample with the likelihood of these predictions.

<a id="anc5"></a>

## License
Source Available License Agreement - [MIT](./LICENSE).