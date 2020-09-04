import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

filename = "../News_Category_Dataset_v2.json"
df = pd.read_json(filename,lines=True)

TextCNN=load_model('my_model_1.h5')

headline=df.headline[0:50]

headline=pd.Series(headline)
#print (headline)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(headline)
p = tokenizer.texts_to_sequences(headline)
p=pd.Series(p)
#print (type(p))

maxlen = 50
p = list(sequence.pad_sequences(p, maxlen=maxlen))
p=np.array(p)

categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})



predictions2= TextCNN.predict(p, 
                            batch_size=1024, 
                            verbose=1)


y=0
count=0
for tester in predictions2:
    #for i in range(len(tester)):
        #print (int_category[i],"-->",tester[i])
    print (headline[y],"-->",int_category[np.argmax(tester)],"-->",df.category[y])
    if int_category[np.argmax(tester)]==df.category[y]:
        print ("field is ",df.category[y])
        count+=1
        
    y=y+1
    
print ("total correct predictions: ",count)
