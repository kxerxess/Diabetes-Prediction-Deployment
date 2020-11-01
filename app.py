# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request, url_for, redirect, render_template
import tensorflow
from joblib import dump, load

model = None
scaler = None
app = Flask(__name__)


def load_model_prereq():
    global model
    global scaler
    global pca
    # model variable refers to the global variable
    # with open('trained_models/final_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    model = tensorflow.keras.models.load_model('trained_models/20X20_90.11_2.12.h5')
    #with open('scalers/std_scaler.bin', 'rb') as f:
    #    scaler = pickle.load(f)
    #with open('scalers/pca.bin', 'rb') as f:
    #    pca = pickle.load(f)
    scaler = load('scalers/std_scaler.bin')
    pca = load('scalers/pca.bin')


@app.route('/')
def home_endpoint():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def get_prediction():
    # Works only for a single sample
    values = [x for x in request.form.values()]
    # Standard Properties
    user_dict = {"Pregnancies": int(values[0]), "Glucose": float(values[1]), "BloodPressure": float(values[2]),
                 "SkinThickness": float(values[3]), "Insulin": float(values[4]), "BMI": float(values[5]), "DiabetesPedigreeFunction": 0, "Age": int(values[6])}
    # Additional Features

    #Gen_GvP_0
    if user_dict['Pregnancies']<=5 and user_dict['Glucose']<=105:
        user_dict['Gen_GvP_0']=1
    else:
        user_dict['Gen_GvP_0']=0

    #Gen_GvBP_1
    if user_dict['BloodPressure']<=80 and user_dict['Glucose']<=105:
        user_dict['Gen_GvBP_1']=1
    else:
        user_dict['Gen_GvBP_1']=0

    #Gen_GvI_2
    if user_dict['Insulin']<=110 and user_dict['Glucose']<=105:
        user_dict['Gen_GvI_2']=1
    else:
        user_dict['Gen_GvI_2']=0

    #Gen_GvA_3
    if user_dict['Age']<=29 and user_dict['Glucose']<=120:
        user_dict['Gen_GvA_3']=1
    else:
        user_dict['Gen_GvA_3']=0

    #Gen_BMI_4
    if user_dict['BMI']<=30:
        user_dict['Gen_BMI_4']=1
    else:
        user_dict['Gen_BMI_4']=0

    #Gen_BvP_5
    if user_dict['BMI']<=31 and user_dict['Pregnancies']<=3:
        user_dict['Gen_BvP_5']=1
    else:
        user_dict['Gen_BvP_5']=0

    #Gen_AvP_6
    if user_dict['Age']<=30 and user_dict['Pregnancies']<=6:
        user_dict['Gen_AvP_6']=1
    else:
        user_dict['Gen_AvP_6']=0

    #Gen_GvB_7
    if user_dict['BMI']<=28 and user_dict['Glucose']<=105:
        user_dict['Gen_GvB_7']=1
    else:
        user_dict['Gen_GvB_7']=0

    #Gen_BvS_8
    if user_dict['BMI']<=28 and user_dict['SkinThickness']<=20:
        user_dict['Gen_BvS_8']=1
    else:
        user_dict['Gen_BvS_8']=0

    #Gen_BP_9
    if user_dict['BloodPressure']<=80:
        user_dict['Gen_BP_9']=1
    else:
        user_dict['Gen_BP_9']=0

    #Num_1 - Num_5
    user_dict['Num_1'] = user_dict['BMI'] * user_dict['SkinThickness']
    user_dict['Num_2'] =  user_dict['Pregnancies'] * user_dict['Age']
    user_dict['Num_3'] = user_dict['Age'] / user_dict['Insulin']
    user_dict['Num_4'] = user_dict['Insulin'] / user_dict['Glucose']
    user_dict['Num_5'] = user_dict['BloodPressure'] / user_dict['Age']

    #Gen_t1_10
    if user_dict['Num_1']<1034:
        user_dict['Gen_t1_10']=1
    else:
        user_dict['Gen_t1_10']=0

    cont_feat = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Num_1', 'Num_2','Num_3', 'Num_4', 'Num_5']

    PR = 4.0
    AG = 34.0

    user_dict['Pregnancies'] = 0.5*(user_dict['Pregnancies']- PR)
    user_dict['Age'] = 0.1*(user_dict['Age']- AG)

    to_scale = list()
    for key in user_dict:
        if key in cont_feat:
            to_scale.append(user_dict[key])
    print(to_scale)
    scaled = scaler.transform(np.array([to_scale]))[0]
    #print(scaled)
    idx=0
    for key in user_dict:
        if key in cont_feat:
            user_dict[key]=scaled[idx]
            idx+=1

    X_feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Age', 'Num_1', 'Num_2',
       'Num_3', 'Num_4', 'Num_5']

    for_pca = list()
    for key in user_dict:
        if key in X_feat:
            for_pca.append(user_dict[key])
    post_PCA = pca.transform([for_pca])

    pca_feat = [0,1,5,8]

    pcaf=list()
    for num,i in enumerate(post_PCA[0]):
        if num in pca_feat:
            pcaf.append(i)

    f = list(pcaf) + list(user_dict.values())
    print("\n\nVALUE OF F:\n\n", f, "\n\n")
    final_inp = f[0:10]+[f[11]]+f[22:27]+f[12:22]+[f[27]]
    print("\n\nVALUE OF F:\n\n", final_inp, "\n\n")
    print(type(final_inp))
    print(len(final_inp))
    final_inp = np.asarray(final_inp)
    final_inp = np.reshape(final_inp, (-1,27))
    print(final_inp.shape)
    pred = model.predict(final_inp)

    if pred>0.5:
        result='DIABETIC'
        accuracy=pred*100
        return render_template("result_diabetic.html", prediction=str(result), accuracy=accuracy[0][0])
    elif pred<=0.5:
        result='NOT DIABETIC'
        accuracy=(1-pred)*100
        return render_template("result_non.html", prediction=str(result), accuracy=accuracy[0][0])
    return None


if __name__ == '__main__':
    load_model_prereq()  # load model at the beginning once only
    app.run(host='127.0.0.1', port=5000, debug=False)
