Die Neue Krone
==============

Goal
----
This project aims to develop a fully automated content driven website. The “Neue Krone” is the name of this generative news site where all the articles are created by machine learning models.

Usage
-----
The whole project can be run directly on Google AppEngine to serve generated articles on: ```https://yourappengineurl.com/sample```  
This can be done by simply cloning the repo to your AppEngine instance and running ```gcloud app deploy``` from the same directory.

Alternatively, the model.py file can be run locally to train the model, generate articles or dump the raw generated output.

#### Training
To train the model execute the following command: ```python model.py train```  
This will start training the model from scratch while outputting generated samples and saving checkpoints at regular intervals. These checkpoints can then be loaded later to generate articles.

#### Generating articles
To generate articles you can use the following command: ```python model.py articles```  
This will endlessly generate new articles and print them to the standard output

#### Raw generated output
Seeing the raw output of the model can be useful for debugging or to simply get a better understanding of what exactly the model does. To see the raw output you can run the following command: ```python model.py``` (without any arguments)  


How it works
------------
The model used internally is a multi-layer RNN that receives a single character at a time as input (one-hot encoded) and outputs probabilities for the next character (between 0.0 and 1.0). This approach does have certain limitations, but also has the advantage that it is very flexible and relatively easy to train.

To make the model generate titles and bodies separately, tags (```<t>``` and ```</t>```) are inserted around all the titles in the dataset. With sufficient training the model picks up this pattern and then the generated output will produce the tags in appropriate places as well. These can then be used during the post-processing of the output to separate the raw articles into titles and bodies.

The file ```model.py``` also contains a function to do this post-processing step, splitting the raw output into distinct articles and doing some simple sanity checks to discard invalid articles.

Limitations and possible improvements
------------------
Because of the character-level model it is very hard, maybe even impossible, for the model to remember context over long stretches of text. This results in the output generally having correct grammar but mostly incoherent/meaningless sentences.

One way of possibly solving this issue would be to switch to a word-level model, but this brings along a lot of complications and limitations of its own. This would be interesting to explore further but is currently outside the scope of this project.

Resources
---------

This project is heavily based on this talk and the accompanying code:
 - https://github.com/martin-gorner/tensorflow-rnn-shakespeare
 - https://www.youtube.com/watch?v=vq2nnJ4g6N0&t=107m25s

The dataset used for training the model: http://mlg.ucd.ie/datasets/bbc.html
