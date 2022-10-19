from scipy.sparse import csr_matrix
import numpy as np
import time

N = 875713

def countOutlink():
    fileO=open("web-Google.txt","r")
    outlinks = {}
    for line in fileO:
        line = line.strip("\n")
        node = int(line.split(" ")[0])
        if(outlinks.__contains__(node)):
            outlinks[node]+=1
        else:
            outlinks[node] = 1
    fileO.close()
    return outlinks

def loadFile(outlinks):
    fileO=open("web-Google.txt","r")
    row=[]
    col=[]
    data=[]
    for line in fileO:
        line = line.strip("\n")
        nodes = line.split(" ")
        col.append(int(nodes[0]))
        row.append(int(nodes[1]))
        data.append(float(1/outlinks[int(nodes[0])]))
    return row,col,data


def q1():
    outlinks = countOutlink()
    row,col,data = loadFile(outlinks)
    mtx = csr_matrix((data, (row, col)), shape=(N, N))
    vec_data = [float(1/N) for i in range(N)]
    vec_col = [0 for i in range(N)]
    vec_row = [i for i in range(N)]
    vec_r = csr_matrix((vec_data, (vec_row, vec_col)), shape=(N,1))
    vec_r = vec_r.todense()
    iter_counter=0
    start = time.process_time()
    while(True):
        iter_counter+=1
        new_r = mtx.dot(vec_r)
        sum=0.0
        arr_new = np.squeeze(np.asarray(new_r))
        arr = np.squeeze(np.asarray(vec_r))
        for i in range(N):
            sum+=abs(arr_new[i] - arr[i])
        vec_r = new_r
        if(sum < 0.02):
            break
    end = time.process_time()
    print("time: "+str((end-start)*1000))
    print("iteration: "+str(iter_counter))
    arr = np.squeeze(np.asarray(vec_r))
    score_list=[]
    for i in range(len(arr)):
        score_list.append([i,arr[i]])
    score_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(score_list[i])

def q2():
    outlinks = countOutlink()
    row,col,data = loadFile(outlinks)
    mtx = csr_matrix((data, (row, col)), shape=(N, N))
    vec_data = [float(1/N) for i in range(N)]
    vec_col = [0 for i in range(N)]
    vec_row = [i for i in range(N)]
    vec_r = csr_matrix((vec_data, (vec_row, vec_col)), shape=(N,1))
    vec_r = vec_r.todense()
    iter_counter=0
    while(True):
        iter_counter+=1
        new_r = mtx.dot(vec_r)
        score_sum=0.0
        sum=0.0
        arr_new = np.squeeze(np.asarray(new_r))
        arr = np.squeeze(np.asarray(vec_r))
        for i in range(N):
            sum+=abs(arr_new[i] - arr[i])
            score_sum += arr_new[i]
        print("iteration "+ str(iter_counter))
        print("new total score "+ str(score_sum))
        print("leaked score "+ str(1.0-score_sum))
        vec_r = new_r
        if(sum < 0.02):
            break

def q3b():
    outlinks = countOutlink()
    beta=0.9
    row,col,data = loadFile(outlinks)
    mtx = csr_matrix((data, (row, col)), shape=(N, N))
    vec_data = [float(1/N) for i in range(N)]
    vec_col = [0 for i in range(N)]
    vec_row = [i for i in range(N)]
    vec_r = csr_matrix((vec_data, (vec_row, vec_col)), shape=(N,1))
    vec_r = vec_r.todense()
    iter_counter=0
    start = time.process_time()
    while(True):
        iter_counter+=1
        new_mtx = mtx.dot(beta)
        new_r = new_mtx.dot(vec_r)
        sum=0.0
        arr_new = np.squeeze(np.asarray(new_r))
        Sum = 0.0
        for i in arr_new:
            Sum+=i
        constant = (1.0-Sum)/float(N)
        new_r = np.add(new_r,constant)
        arr_new = np.squeeze(np.asarray(new_r))
        arr = np.squeeze(np.asarray(vec_r))
        for i in range(N):
            sum+=abs(arr_new[i] - arr[i])
        vec_r = new_r
        if(sum < 0.02):
            break
    end = time.process_time()
    print("time: "+str((end-start)*1000))
    print("iteration: "+str(iter_counter))
    arr = np.squeeze(np.asarray(vec_r))
    score_list=[]
    for i in range(len(arr)):
        score_list.append([i,arr[i]])
    score_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(score_list[i])

def q3c():
    outlinks = countOutlink()
    betas=[1.0,0.9,0.8,0.7,0.6,0.5]
    row,col,data = loadFile(outlinks)
    mtx = csr_matrix((data, (row, col)), shape=(N, N))
    vec_data = [float(1/N) for i in range(N)]
    vec_col = [0 for i in range(N)]
    vec_row = [i for i in range(N)]
    for beta in betas:
        print("beta = "+str(beta))
        vec_r = csr_matrix((vec_data, (vec_row, vec_col)), shape=(N,1))
        vec_r = vec_r.todense()
        iter_counter=0
        while(True):
            iter_counter+=1
            new_mtx = mtx.dot(beta)
            new_r = new_mtx.dot(vec_r)
            sum=0.0
            arr_new = np.squeeze(np.asarray(new_r))
            Sum = 0.0
            for i in arr_new:
                Sum+=i
            constant = (1.0-Sum)/float(N)
            new_r = np.add(new_r,constant)
            arr_new = np.squeeze(np.asarray(new_r))
            arr = np.squeeze(np.asarray(vec_r))
            for i in range(N):
                sum+=abs(arr_new[i] - arr[i])
            vec_r = new_r
            if(sum < 0.02):
                break
        print("iteration: "+str(iter_counter))

if __name__ == "__main__":
    #q1()
    #q2()
    #q3b()
    q3c()
