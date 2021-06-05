import main
import random
import requests
# import our module
import json

data = main.load_data("C://Users//My//Desktop//Ace Tech Academy//Data//JSON//Artificial Intelligence.json")

x1,y1,words,document,classes = main.data_processing(data["intents"],"patterns","tag")
model = main.train_model([[123,"relu"],[234,"relu"]],x1,y1)
model.fit(x1,y1,epochs=200,batch_size=9)
model.save("C://Users//My//Desktop//Ace Tech Academy//Models//Acrom.model")

while True:
    asi = input("Message:")
    ans = main.predict_class(model,asi.lower(),classes,words)
    print(ans)
