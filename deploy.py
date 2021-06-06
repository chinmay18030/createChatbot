import createChatbot as cc

chatbot = cc.Chatbot("intents.json")
#load the data from this
data = chatbot.load_data()
#x1 is the x train data 
#y1 is the y train data 
x1,y1,words,document,classes = chatbot.data_processing(data["intents"],"patterns","tag")
#train the data with the types of layer activations  you want, otherwise use the default one
model = chatbot.train_model([[123,"relu"],[234,"relu"]],x1,y1)
# model has been created , now fit the data
model.fit(x1,y1,epochs=200,batch_size=9)

# ask the message 
while True:
    asi = input("Message:")
    # this command helps to predict the intent in which the input by the user fall
    ans = chatbot.predict_class(model,asi.lower(),classes,words)
    print(ans)
