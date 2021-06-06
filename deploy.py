import createChatbot as cc

chatbot = cc.Chatbot("intents.json")

data = chatbot.load_data()

x1,y1,words,document,classes = chatbot.data_processing(data["intents"],"patterns","tag")
model = chatbot.train_model([[123,"relu"],[234,"relu"]],x1,y1)
model.fit(x1,y1,epochs=200,batch_size=9)


while True:
    asi = input("Message:")
    ans = chatbot.predict_class(model,asi.lower(),classes,words)
    print(ans)
