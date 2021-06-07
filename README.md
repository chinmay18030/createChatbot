# createChatbot
Making a chatbot is very difficult stuff. Processing data and training the model. 
This module will help you to make a python module very easily. 
#### Dependencies
+ `nltk`
+ `keras`
+ `numpy`

#### Installation
If you have python and pip installed in your computer, execute the following

```bash
pip install createChatbot
```
### Loading the data

```python
import createChatbot as cc
chatbot = cc.Chatbot(filename)
data = chatbot.load_data()
```
Give the location of file name for filename parameter

### Processing data
Now just give the data processing command
```python
x1,y1,words,document,classes = chatbot.data_processing(data,questionKey,tagKey)
```
This command returns x train values, y train values, words list, document list and the classes list. 
+ The first parameter is the dictionary in which all the intents are there.
+ The second parameter is the key in which all the patterns are
+ The third parameter is the key whose value is the type of intent

### Make a suitable model and training it
```python
model = chatbot.train_model([[123,"relu"],[234,"relu"]],x1,y1)
model.fit(x1,y1)
```
This returns the model. 

+ The first parameter is a list with units and layer activation
+ The second and third parameter are the x train and y train values

#### Predicting it's class

```python
asi = input("Message:")
ans = chatbot.predict_class(model,asi.lower(),classes,words)
print(ans)
```
Here, take the sentence from user and predicting the class of it's model. The chatbot.predict_class function returns a dictionary with the predicted class and probability of being correct

+ First parameter requires for the model
+ second is the sentence from the user
+ Third and fourth are the classes and words

Now as you have got the predicted class, you can find the dictionary which has the tag same to the predicted one.
### Want a example 
```bash
https://github.com/chinmay18030/createChatbot
```
