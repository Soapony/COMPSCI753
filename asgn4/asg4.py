from contextlib import nullcontext
import numpy as np
import json
from scipy.sparse import csr_matrix
import math

def load_train():
    fileO = open("train.json", "r")
    user={}
    item={}
    row=[]
    col=[]
    data=[]
    userid=0
    itemid=0
    for line in fileO:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            user[jData["user_id"]]=userid
            userid+=1
        if(not item.__contains__(jData["business_id"])):
            item[jData["business_id"]]=itemid
            itemid+=1
        col.append(user[jData["user_id"]])
        row.append(item[jData["business_id"]])
        data.append(float(jData["stars"]))
    fileO.close()
    return user,item,row,col,data

def load_val():
    fileO = open("val.json", "r")
    user={}
    item={}
    row=[]
    col=[]
    data=[]
    userid=0
    itemid=0
    for line in fileO:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            user[jData["user_id"]]=userid
            userid+=1
        if(not item.__contains__(jData["business_id"])):
            item[jData["business_id"]]=itemid
            itemid+=1
        col.append(user[jData["user_id"]])
        row.append(item[jData["business_id"]])
        data.append(float(jData["stars"]))
    itemN=len(item)
    userN=len(user)
    fileO.close()
    return data,row,col,user,item

def load_test():
    fileO = open("test.json", "r")
    user={}
    item={}
    row=[]
    col=[]
    data=[]
    userid=0
    itemid=0
    for line in fileO:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            user[jData["user_id"]]=userid
            userid+=1
        if(not item.__contains__(jData["business_id"])):
            item[jData["business_id"]]=itemid
            itemid+=1
        col.append(user[jData["user_id"]])
        row.append(item[jData["business_id"]])
        data.append(float(jData["stars"]))
    itemN=len(item)
    userN=len(user)
    fileO.close()
    return data,row,col,user,item

def load_all():
    fileTrain = open("train.json", "r")
    fileVal = open("val.json","r")
    fileTest = open("test.json","r")
    user={}
    item={}
    userid=0
    itemid=0
    col=[]
    row=[]
    data=[]
    for line in fileTrain:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            userdata=[userid,float(jData["stars"]),1]
            user[jData["user_id"]]=userdata
            userid+=1
        else:
            user[jData["user_id"]][2]+=1
            user[jData["user_id"]][1]+=float(jData["stars"])
        if(not item.__contains__(jData["business_id"])):
            itemdata=[itemid,float(jData["stars"]),1]
            item[jData["business_id"]]=itemdata
            itemid+=1
        else:
            item[jData["business_id"]][2]+=1
            item[jData["business_id"]][1]+=float(jData["stars"])
        #col.append(user[jData["user_id"]])
        #row.append(item[jData["business_id"]])
        data.append(float(jData["stars"]))
    for line in fileVal:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            userdata=[userid,0.0,1]
            user[jData["user_id"]]=userdata
            userid+=1
        if(not item.__contains__(jData["business_id"])):
            itemdata=[itemid,0.0,1]
            item[jData["business_id"]]=itemdata
            itemid+=1
        #col.append(user[jData["user_id"]])
        #row.append(item[jData["business_id"]])
        #data.append(float(jData["stars"]))
    for line in fileTest:
        jData = json.loads(line)
        if(not user.__contains__(jData["user_id"])):
            userdata=[userid,0.0,1]
            user[jData["user_id"]]=userdata
            userid+=1
        if(not item.__contains__(jData["business_id"])):
            itemdata=[itemid,0.0,1]
            item[jData["business_id"]]=itemdata
            itemid+=1
        #col.append(user[jData["user_id"]])
        #row.append(item[jData["business_id"]])
        #data.append(float(jData["stars"]))
    fileTrain.close()
    fileTest.close()
    fileVal.close()
    return user,item,data

def q1():
    user,item,row,col,data = load_train()
    global_count = len(data)
    global_sum = sum(data)
    biasG = float(global_sum/global_count)
    user_id = user["b4aIMeXOx4cn3bjtdIOo6Q"]
    item_id = item["7VQYoXk3Tc8EZeKuXeixeg"]
    user_sum=0.0
    user_count=0
    for i in range(len(col)):
        if(col[i] == user_id):
            user_sum+=data[i]
            user_count+=1
    item_sum=0.0
    item_count=0
    for i in range(len(row)):
        if(row[i] == item_id):
            item_sum+=data[i]
            item_count+=1
    mean_usr = user_sum/float(user_count)
    mean_itm = item_sum/float(item_count)
    biasU = mean_usr - biasG
    biasI = mean_itm - biasG
    print("global bias: "+str(biasG))
    print("user specific bias: "+str(biasU))
    print("item specific bias: "+str(biasI))

def q2():
    user,item,row,col,data = load_train()
    k=8
    userN=len(user)
    itemN=len(item)
    mtx = csr_matrix((data, (row, col)), shape=(itemN, userN))
    del data
    learning_rate = 0.01
    hyperparameter = 0.3
    del user,item
    userM = np.random.normal(0.0,1,size=(userN, k))
    itemM = np.random.normal(0.0,1,size=(itemN, k))
    for epoch in range(10):
        for r in range(mtx.shape[0]):
            itemV = itemM[r,:]
            for ind in range(mtx.indptr[r], mtx.indptr[r+1]):
                c=mtx.indices[ind]
                ground_r = mtx.data[ind]
                predict_r = itemV.dot(userM[c,:])
                for f in range(k):
                    itemP = itemM[r,f]
                    userQ = userM[c,f]
                    delta_u = -2.0 * (ground_r - predict_r) * itemP + 2.0 * hyperparameter * userQ
                    delta_i = -2.0 * (ground_r - predict_r) * userQ + 2.0 * hyperparameter * itemP
                    userM[c,f] = userQ - learning_rate * delta_u
                    itemM[r,f] = itemP - learning_rate * delta_i
        T_userM = np.transpose(userM)
        predict_M = itemM.dot(T_userM)
        data=[]
        for i in range(len(row)):
            data.append(predict_M[row[i]][col[i]])
        predict_Msparse = csr_matrix((data, (row,col)),shape=(itemN, userN))
        mse = (mtx-predict_Msparse).power(2).mean()
        rmse = math.sqrt(mse)
        print("epoch",epoch,":",rmse)

def q3():
    user,item,row,col,data = load_train()
    userN=len(user)
    itemN=len(item)
    mtx = csr_matrix((data, (row, col)), shape=(itemN, userN))
    user,item,tmp = load_all()
    del tmp
    user_all = len(user)
    item_all = len(item)
    del row,col,data
    learning_rate = 0.01
    hyperparameter = 0.3
    k_set=[4,8,16]
    best_RMSE=100.0
    global bestU
    global bestI
    for k in k_set:
        userM = np.random.normal(0.0,1,size=(user_all, k))
        itemM = np.random.normal(0.0,1,size=(item_all, k))
        for epoch in range(10):
            print(k,epoch)
            for r in range(mtx.shape[0]):
                itemV = itemM[r,:]
                for ind in range(mtx.indptr[r], mtx.indptr[r+1]):
                    c=mtx.indices[ind]
                    ground_r = mtx.data[ind]
                    predict_r = itemV.dot(userM[c,:])
                    for f in range(k):
                        itemP = itemM[r,f]
                        userQ = userM[c,f]
                        delta_u = -2.0 * (ground_r - predict_r) * itemP + 2.0 * hyperparameter * userQ
                        delta_i = -2.0 * (ground_r - predict_r) * userQ + 2.0 * hyperparameter * itemP
                        userM[c,f] = userQ - learning_rate * delta_u
                        itemM[r,f] = itemP - learning_rate * delta_i
        new_userM=[]
        new_itemM=[]
        valD,valR,valC,valU,valI = load_val()
        valM = csr_matrix((valD,(valR,valC)),shape = (len(valI),len(valU)))
        for valK in valU.keys():
            uid=user[valK][0]
            new_userM.append(userM[uid,:])
        for valK in valI.keys():
            iid=item[valK][0]
            new_itemM.append(itemM[iid,:])
        new_userM=np.asarray(new_userM)
        new_itemM=np.asarray(new_itemM)
        T_userM = np.transpose(new_userM)
        predict_M = new_itemM.dot(T_userM)
        data=[]
        for i in range(len(valR)):
            data.append(predict_M[valR[i]][valC[i]])
        predict_Msparse = csr_matrix((data, (valR,valC)),shape=(len(valI), len(valU)))
        mse = (valM-predict_Msparse).power(2).mean()
        rmse = math.sqrt(mse)
        print("k:",k,"RMSE:",rmse)
        if(rmse < best_RMSE):
            bestU=userM
            bestI=itemM
    del valM,valR,valD,valC,valU,valI
    testD,testR,testC,testU,testI = load_test()
    testM = csr_matrix((testD,(testR,testC)),shape = (len(testI),len(testU)))
    new_userM=[]
    new_itemM=[]
    for testK in testU.keys():
        uid=user[testK][0]
        new_userM.append(userM[uid,:])
    for testK in testI.keys():
        iid=item[testK][0]
        new_itemM.append(itemM[iid,:])
    new_userM=np.asarray(new_userM)
    new_itemM=np.asarray(new_itemM)
    T_userM=np.transpose(new_userM)
    bestM=new_itemM.dot(T_userM)
    data=[]
    for i in range(len(testR)):
        data.append(bestM[testR[i]][testC[i]])
    bestMsparse = csr_matrix((data, (testR,testC)),shape=(len(testI), len(testU)))
    mse = (testM-bestMsparse).power(2).mean()
    rmse = math.sqrt(mse)
    print("RMSE on test.json:",rmse)

def q4():
    user,item,row,col,data = load_train()
    print("done load train")
    biasU,biasI,biasG = bias_calculation()
    print("done bias cal")
    k=8
    userN=len(user)
    itemN=len(item)
    mtx = csr_matrix((data, (row, col)), shape=(itemN, userN))
    del data
    learning_rate = 0.01
    hyperparameter = 0.3
    userM = np.random.normal(0.0,1,size=(userN, k))
    itemM = np.random.normal(0.0,1,size=(itemN, k))
    print("start training")
    for epoch in range(10):
        for r in range(mtx.shape[0]):
            itemV = itemM[r,:]
            for ind in range(mtx.indptr[r], mtx.indptr[r+1]):
                c=mtx.indices[ind]
                ground_r = mtx.data[ind]
                predict_r = itemV.dot(userM[c,:])
                bU = biasU[c]
                bI = biasI[r]
                deltaBU = -2.0 * (ground_r - predict_r) + 2.0 * hyperparameter * bU
                deltaBI = -2.0 * (ground_r - predict_r) + 2.0 * hyperparameter * bI
                biasU[c] = bU - learning_rate * deltaBU
                biasI[r] = bI - learning_rate * deltaBI
                for f in range(k):
                    itemP = itemM[r,f]
                    userQ = userM[c,f]
                    delta_u = -2.0 * (ground_r - predict_r) * itemP + 2.0 * hyperparameter * userQ
                    delta_i = -2.0 * (ground_r - predict_r) * userQ + 2.0 * hyperparameter * itemP
                    userM[c,f] = userQ - learning_rate * delta_u
                    itemM[r,f] = itemP - learning_rate * delta_i
        T_userM = np.transpose(userM)
        predict_M = itemM.dot(T_userM)
        data=[]
        for i in range(len(row)):
            predict_r = predict_M[row[i]][col[i]]+biasG+biasU[col[i]]+biasI[row[i]]
            data.append(predict_r)
        predict_Msparse = csr_matrix((data, (row,col)),shape=(itemN, userN))
        mse = (mtx-predict_Msparse).power(2).mean()
        rmse = math.sqrt(mse)
        print("epoch",epoch,":",rmse)
    uid = user["b4aIMeXOx4cn3bjtdIOo6Q"]
    iid = item["7VQYoXk3Tc8EZeKuXeixeg"]
    print("bias for user:",biasU[uid])
    print("bias for item:",biasI[iid])

def q5():
    user,item,row,col,data = load_train()
    print("done load train")
    userN=len(user)
    itemN=len(item)
    mtx = csr_matrix((data, (row, col)), shape=(itemN, userN))
    user,item,tmp = load_all()
    del tmp
    user_all = len(user)
    item_all = len(item)
    del row,col,data
    learning_rate = 0.01
    hyperparameter = 0.3
    k_set=[4,8,16]
    best_RMSE=100.0
    global bestU
    global bestI
    global bestbU
    global bestbI
    global bestbG
    for k in k_set:
        biasU,biasI,biasG = bias_calculation()
        print("done bias cal")
        userM = np.random.normal(0.0,1,size=(user_all, k))
        itemM = np.random.normal(0.0,1,size=(item_all, k))
        for epoch in range(10):
            print(k,epoch)
            for r in range(mtx.shape[0]):
                itemV = itemM[r,:]
                for ind in range(mtx.indptr[r], mtx.indptr[r+1]):
                    c=mtx.indices[ind]
                    ground_r = mtx.data[ind]
                    predict_r = itemV.dot(userM[c,:])
                    bU = biasU[c]
                    bI = biasI[r]
                    deltaBU = -2.0 * (ground_r - predict_r) + 2.0 * hyperparameter * bU
                    deltaBI = -2.0 * (ground_r - predict_r) + 2.0 * hyperparameter * bI
                    biasU[c] = bU - learning_rate * deltaBU
                    biasI[r] = bI - learning_rate * deltaBI
                    for f in range(k):
                        itemP = itemM[r,f]
                        userQ = userM[c,f]
                        delta_u = -2.0 * (ground_r - predict_r) * itemP + 2.0 * hyperparameter * userQ
                        delta_i = -2.0 * (ground_r - predict_r) * userQ + 2.0 * hyperparameter * itemP
                        userM[c,f] = userQ - learning_rate * delta_u
                        itemM[r,f] = itemP - learning_rate * delta_i
        new_userM=[]
        new_itemM=[]
        valD,valR,valC,valU,valI = load_val()
        valM = csr_matrix((valD,(valR,valC)),shape = (len(valI),len(valU)))
        for valK in valU.keys():
            uid=user[valK][0]
            new_userM.append(userM[uid,:])
        for valK in valI.keys():
            iid=item[valK][0]
            new_itemM.append(itemM[iid,:])
        new_userM=np.asarray(new_userM)
        new_itemM=np.asarray(new_itemM)
        T_userM = np.transpose(new_userM)
        predict_M = new_itemM.dot(T_userM)
        data=[]
        for i in range(len(valR)):
            predict_r = predict_M[valR[i]][valC[i]]+biasG+biasU[valC[i]]+biasI[valR[i]]
            data.append(predict_r)
        predict_Msparse = csr_matrix((data, (valR,valC)),shape=(len(valI), len(valU)))
        mse = (valM-predict_Msparse).power(2).mean()
        rmse = math.sqrt(mse)
        print("k:",k,"RMSE:",rmse)
        if(rmse < best_RMSE):
            bestU=userM
            bestI=itemM
            bestbU=biasU
            bestbI=biasI
            bestbG=biasG
    del valM,valR,valD,valC,valU,valI
    testD,testR,testC,testU,testI = load_test()
    testM = csr_matrix((testD,(testR,testC)),shape = (len(testI),len(testU)))
    new_userM=[]
    new_itemM=[]
    for testK in testU.keys():
        uid=user[testK][0]
        new_userM.append(userM[uid,:])
    for testK in testI.keys():
        iid=item[testK][0]
        new_itemM.append(itemM[iid,:])
    new_userM=np.asarray(new_userM)
    new_itemM=np.asarray(new_itemM)
    T_userM=np.transpose(new_userM)
    bestM=new_itemM.dot(T_userM)
    data=[]
    for i in range(len(testR)):
        predict_r = bestM[testR[i]][testC[i]]+biasG+biasU[testC[i]]+biasI[testR[i]]
        data.append(predict_r)
    bestMsparse = csr_matrix((data, (testR,testC)),shape=(len(testI), len(testU)))
    mse = (testM-bestMsparse).power(2).mean()
    rmse = math.sqrt(mse)
    print("RMSE on test.json:",rmse)

def bias_calculation():
    user,item,data=load_all()
    global_count = len(data)
    global_sum = sum(data)
    biasG = float(global_sum/global_count)
    biasU=[]
    biasI=[]
    for keyU in user.keys():
        mean = float(user[keyU][1]/user[keyU][2])
        biasU.append(mean - biasG)
            
    for keyI in item.keys():
        mean = float(item[keyI][1]/item[keyI][2])
        biasI.append(mean - biasG)
    
    print(len(biasU))
    print(len(biasI))
    return biasU,biasI,biasG

if __name__ == "__main__":
    #q1()
    #q2()
    #q3()
    q4()
    #q5()