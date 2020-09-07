import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
global model, graph
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()



model = load_model('mymodel.h5') 
app = Flask(__name__) 
@app.route('/') 
def home():
    return render_template('index.html')
@app.route('/login',methods=['POST']) 
def login():
    a=request.form['a']
    b=request.form['b']
    c=request.form['c']
    d=request.form['s']
    if (d == "cairdin"):
        s1,s2=0,1
        if (d == "jnardino"):
            s1,s2=1,0
            total = [[s1, s2, a, b, c]]
            with graph.as_default():
                ypred = model.predict(np.array(total)) 
                print(ypred) 
                
                
                
            return render_template('index.html', abc = ypred)


if __name__ == "__main__":
    app.run(debug = True)
    
                