from flask import Flask, stream_with_context, request, Response, url_for, render_template, jsonify, redirect
import pandas
from collections import Counter 
from sklearn.externals import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matlab.engine


app = Flask(__name__)

ncol = 87067
Columns = 87067
eng = matlab.engine.connect_matlab()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/background_process_test')
def background_process_test():  
    eng.liam(nargout=0)
    Matlab_data = eng.workspace['a']
    df = pandas.DataFrame(Matlab_data)
    df = pandas.DataFrame(df[0].values.tolist())
    
    df_count = df.count(axis=1, numeric_only=True)[1]


    Total_df = 43510 - df_count


    nan1 = [None] * Total_df
    df = df.reindex(columns = df.columns.tolist() + nan1)
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    df = imp.fit_transform(df.T).T
    df = pandas.DataFrame(df)

    pipe = joblib.load('X:\\March Experiments\\rfcexp18.pkl')
    pred = pandas.Series(pipe.predict(df))
    prediction = list(pred)

    def most_frequent(List): 
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0] 

    List = prediction 
    output = (most_frequent(List)) 
    print 'Detected action is ' + output
    result =  'Detected action is ' + output
    return jsonify(result=result)

if __name__ == "__main__":
	app.run(debug = True) 

