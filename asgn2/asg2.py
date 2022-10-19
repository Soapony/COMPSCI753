import json
import matplotlib.pyplot as plt
import math
import random
import time

def loadFile():
    with open("challenge_set.json","r") as a:
        json_file = json.load(a)
        playlists = json_file['playlists']
        uri_list = []
        for playlist in playlists:
            if(playlist['tracks'] != []):
                for track in playlist['tracks']:
                    uri=track['track_uri'][14:]
                    uri_list.append([uri,track['track_name']])
    return uri_list

def gen_track_dict(uri_list):
    id_num = 0
    track_dict = {}
    for uri in uri_list:
        if(track_dict.__contains__(uri[0])):
            track_dict[uri[0]][0]+=1
        else:
            track_dict[uri[0]] = [1,id_num,uri[1]]
            id_num+=1
    
    return track_dict

def part1(track_list):
    freq_sum = 0
    for freq in track_list:
        freq_sum+=freq
    print("average frequency: ",float(freq_sum/len(track_list)))
    track_list.sort(reverse=True)
    x_pos=[i for i in range(len(track_list))]
    plt.ylabel("frequency")
    plt.xlabel("all tracks")
    plt.plot(x_pos,track_list)
    plt.show()

def mg_summery(uri_list,size):
    #print("k = ",size)
    mg_sum={}
    dec_count=0
    for key in uri_list:
        if(mg_sum.__contains__(key[0])):
            mg_sum[key[0]]+=1
        elif(len(mg_sum)<=size):
            mg_sum[key[0]]=1
        else:
            empty_list=[]
            for (k,v) in mg_sum.items():
                mg_sum[k]-=1
                if(mg_sum[k] == 0):
                    empty_list.append(k)
            for empt_key in empty_list:
                mg_sum.pop(empt_key)
            dec_count+=1
    mg_list=[]
    for (k,v) in mg_sum.items():
        mg_list.append([k,v])
    #print(dec_count)
    return mg_list

def plot_mg(mg_sum):
    freq=[]
    for kv in mg_sum:
        freq.append(kv[1])
    freq.sort(reverse=True)
    x_pos=[i for i in range(len(freq))]
    plt.ylabel("Miisra-Gries summary")
    plt.xlabel("k tracks")
    plt.plot(x_pos,freq)
    plt.show()

def mg_RE(mg_sum,track_dict):
    mg_sum.sort(key=lambda x: x[1], reverse=True)
    est_f=[]
    for i in range(20):
        rel_error = float(1 - (mg_sum[i][1] / track_dict[mg_sum[i][0]][0]))
        print(i,"   ",rel_error)
        est_f.append(mg_sum[i][1])
    x_pos = [i for i in range(len(est_f))]
    plt.ylabel("estimated frequency")
    plt.xlabel("top-20 tracks")
    plt.plot(x_pos,est_f)
    plt.show()

def mg_avg_RE(mg_sum,track_dict):
    total = len(mg_sum)
    sum_sum = 0.0
    for sum in mg_sum:
        sum_sum += float(sum[1] / track_dict[sum[0]][0])
    avg_rel = 1.0 - (sum_sum / float(total))
    return avg_rel

def uri_to_uid(uri_list,track_dict):
    uid_list=[]
    for uri in uri_list:
        uid_list.append(track_dict[uri[0]][1])
    return uid_list

def construct_matrix(uid_list,epsilon,delta):
    w=round(2 / epsilon**2)
    d = round(math.log(1 / delta, 10))
    print("w = ",w,"d = ",d)
    p = genPrime(w)
    hashFunc_list = []
    signHash_list = []
    for i in range(d):
        hashFunc = genHashFunc(p)
        hashFunc_list.append(hashFunc)
        hashFunc = genHashFunc(p)
        signHash_list.append(hashFunc)
    counter_matrix = [[0 for i in range(w)] for j in range(d)]
    for uid in uid_list:
        for i in range(d):
            pos = (uid * hashFunc_list[i][0] + hashFunc_list[i][1]) % w
            sign = ((uid * signHash_list[i][0] + signHash_list[i][1]) % w) % 2
            if(sign == 0):
                counter_matrix[i][pos]+=1
            elif(sign == 1):
                counter_matrix[i][pos]-=1
            else:
                print("error!!!!")
    
    return counter_matrix,hashFunc_list,w,w*d

def CS_algo(matrix,hash_list,track_dict,w):
    freq_list=[]
    for k,v in track_dict.items():
        uid = v[1]
        freq=[]
        for i in range(len(hash_list)):
            pos = ((uid * hash_list[i][0] + hash_list[i][1]) % w)
            freq.append(matrix[i][pos])
        freq.sort()
        med = 0
        if(len(freq) % 2 == 0):
            med = int(freq[int(len(freq)/2)] + freq[int(len(freq)/2 - 1)] / 2)
        else:
            med = freq[int(len(freq)/2)]
        freq_list.append([med,k])
    return freq_list

def plot_cs(freq_list):
    freq=[]
    for f in freq_list:
        freq.append(f[0])
    freq.sort(reverse=True)
    xPos = [i for i in range(len(freq))]
    plt.ylabel("Count Sketch frequency")
    plt.xlabel("all tracks")
    plt.plot(xPos,freq)
    plt.show()

def cs_RE(freq_list,track_dict):
    re_list=[]
    for f in freq_list:
        est_f = f[0]
        true_f = track_dict[f[1]][0]
        re = abs(1.0 - (float(est_f) / float(true_f)))
        re_list.append([re,est_f])
    re_list.sort(key = lambda x:x[0],reverse=True)
    rr=[]
    f_list=[]
    for i in re_list:
        rr.append(i[0])
        f_list.append(i[1])
    xPos = [i for i in range(len(f_list))]
    plt.ylabel("relative error")
    plt.xlabel("all track")
    plt.plot(xPos, f_list)
    plt.plot(xPos,rr)
    plt.legend(["estimated frequency","relative error"])
    plt.show()

def is_prime(k):
    for i in range(2,k):
        if(k%i) == 0:
            return False
    return True

def genPrime(n):
    p=n
    while(not is_prime(p)):
        p+=1
    return p

def genHashFunc(p):
    a = random.randint(1,p-1)
    while(p%a == 0):
        a = random.randint(1,p-1)
    b = random.randint(0,p-1)
    return [a,b]

def cs_avg_RE(freq_list,track_dict):
    total = len(freq_list)
    cs_sum = 0.0
    for f in freq_list:
        est_f = f[0]
        true_f = track_dict[f[1]][0]
        cs_sum += float(est_f) / float(true_f)
    avg_rel = abs(1.0 - (cs_sum / float(total)))
    return avg_rel

def part3c_mg(uri_list,track_dict):
    size_list=[50,100,200,500,800,1000,1500,2000,4000,5000,8000,10000,15000,20000]
    rel_list=[]
    for size in size_list:
        mg_sum = mg_summery(uri_list,size)
        avg_rel = mg_avg_RE(mg_sum,track_dict)
        rel_list.append(avg_rel)
    plt.xlabel("k")
    plt.ylabel("avg relative error")
    plt.plot(size_list,rel_list)
    plt.show()

def part3c_cs(uid_list,track_dict):
    para_list=[[0.1,0.1],[0.1,0.01],[0.05,0.1],[0.05,0.01],[0.03,0.001],[0.02,0.001],[0.02,0.000001],[0.01,0.000001],[0.01,0.000000001]]
    k_list=[]
    rel_list=[]
    for para in para_list:
        counter_matrix,hash_list,w,k = construct_matrix(uid_list,para[0],para[1])
        freq_list = CS_algo(counter_matrix,hash_list,track_dict,w)
        avg_re = cs_avg_RE(freq_list,track_dict)
        rel_list.append(avg_re)
        k_list.append(k)
    plt.xlabel("k")
    plt.ylabel("avg relative error")
    plt.plot(k_list,rel_list)
    plt.show()


def part3d_mg(mg_sum,track_dict):
    mg_sum.sort(key=lambda x: x[1], reverse=True)
    for i in range(20):
        name=track_dict[mg_sum[i][0]][2]
        est_freq=mg_sum[i][1]
        true_freq=track_dict[mg_sum[i][0]][0]
        rel_e=abs(1-(float(est_freq)/float(true_freq)))
        print(name,est_freq,true_freq,"%.2f"%rel_e)

def part3d_cs(freq_list,track_dict):
    freq_list.sort(key=lambda x: x[0], reverse=True)
    for i in range(20):
        name=track_dict[freq_list[i][1]][2]
        est_f = freq_list[i][0]
        true_f = track_dict[freq_list[i][1]][0]
        re = abs(1 - (float(est_f) / float(true_f)))
        print(name,est_f,true_f,re)

if __name__ == "__main__":
    uri_list = loadFile()
    
    track_dict=gen_track_dict(uri_list)

    """
    track_list=[]
    for v in track_dict.values():
        track_list.append(v[0])
    """
    
    #part1(track_list)
    #mg_sum = mg_summery(uri_list,5000)
    #mg_RE(mg_sum,track_dict)
    #plot_mg(mg_sum)
    
    uid_list = uri_to_uid(uri_list,track_dict)
    #epsilon=0.01
    #delta=0.01
    #counter_matrix,hash_list,w = construct_matrix(uid_list,epsilon,delta)
    #freq_list = CS_algo(counter_matrix,hash_list,track_dict,w)
    #plot_cs(freq_list)
    #cs_RE(freq_list,track_dict)
    part3c_cs(uid_list,track_dict)
    #part3d_cs(freq_list,track_dict)

    #part3c_mg(uri_list,track_dict)
    #part3d_mg(mg_sum,track_dict)
    
    