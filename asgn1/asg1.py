import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

def readBit(filename):
    fileObj = open(filename,"r")
    bitVec=[]
    for line in fileObj:
        l=line.strip("\n")
        row=l.split('\t')
        bitVec.append(row)
    return bitVec

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

def genC(p,r):
    cList=[]
    cList.append(random.randint(0,p-1))
    for i in range(r):
        cList.append(random.randint(1,p-1))
    return cList

def genHashTable(n,p,l,k,m,bitVecA):
    minHashPairs=[]
    hashTable = [ [ [] for i in range(m) ] for j in range(l)]
    sigMatrix=[]
    cTable=[]
    #compute hash tables
    #for each band
    for i in range(l):
        cList=genC(p,k)
        cTable.append(cList)
        #for each row
        for j in range(k):
            #generate a & b
            pair=genHashFunc(p)
            minHashPairs.append(pair)
            permutation = [0]*n

            #generate permutation
            for index in range(n):
                hashValue = ((pair[0] * index + pair[1]) % p) % n
                permutation[hashValue]=index

            #for each article compute signature
            sig=[]
            for article in bitVecA:
                #for each article, check the first "1" in permutation's order
                for index in range(n):
                    if(article[permutation[index]+1] == "1"):
                        #find the first "1" break to next article
                        sig.append(permutation[index])
                        break

            #add to signature matrix for this band
            sigMatrix.append(sig)
            
        #compute hash table for this band
        for article in range(len(bitVecA)):
            #get the signature for this article
            sig=[]
            for index in range(k):
                sig.append(sigMatrix[index][article])

            #calculate level 2 signature
            tmpSum=cList[0]
            for index in range(1,len(cList)):
                tmpSum+=sig[index-1] * cList[index]
            bucket_no = tmpSum % p % m
            hashTable[i][bucket_no].append(str(article+1))

    #for part I (b)
    #for pair in minHashPairs:
    #    print("("+str(pair[0])+"x + "+str(pair[1])+") mod "+str(p)+" mod "+str(n))

    return hashTable,sigMatrix,minHashPairs,cTable

def plotHeatmap(hashTable):
    #plot heatmap
    heatmapTables=[[]for j in range(l)]
    for band in range(len(hashTable)):
        for buckets in hashTable[band]:
            heatmapTables[band].append(len(buckets))
    npArray = np.array(heatmapTables)
    ax = sns.heatmap(npArray, cmap="hot",annot=True,fmt="d",linewidths=0.1)
    plt.show()

def query(hashTable,sigMatrix,bitVecQ,minHashPairs,l,k,m,n,p,cTable):
    result_list=[[]for j in range(4)]
    sigMatrixQ=[]
    hitList=[[]for i in range(4)]
    counter=0
    #generate sigature matrix
    for i in range(l):
        cList=cTable[i]
        for j in range(k):
            permutation=[0]*n
            for index in range(n):
                hashValue = ((minHashPairs[counter][0] * index + minHashPairs[counter][1]) % p) % n
                permutation[hashValue]=index
            sig=[]
            for q_article in bitVecQ:
                for index in range(n):
                    if(q_article[permutation[index]+1] == "1"):
                        sig.append(permutation[index])
                        break
            sigMatrixQ.append(sig)
            counter+=1
        
        #get hashvalue and record collision article
        for q_article in range(len(bitVecQ)):
            sig=[]
            for index in range(k):
                sig.append(sigMatrixQ[index][q_article])
            tmpSum=cList[0]
            for index in range(1,len(cList)):
                tmpSum+=sig[index-1] * cList[index]
            bucket_no=tmpSum %p %m
            if(len(hashTable[i][bucket_no]) != 0):
                for id in hashTable[i][bucket_no]:
                    if(id not in hitList[q_article]):
                        hitList[q_article].append(id)

    #for those collision article, calculate jaccard similarity
    for i in range(len(hitList)):
        for id in range(len(hitList[i])):
            sigHit=0
            for j in range(l*k):
                if(sigMatrix[j][int(hitList[i][id])-1] == sigMatrixQ[j][i]):
                    sigHit+=1
            jac_sim=float(sigHit/(l*k))
            artId = hitList[i][id]
            art_c = bitVecA[int(artId)-1][n+1]
            result=[artId,jac_sim,art_c]
            result_list[i].append(result)
    
    #sort and get top 10 article
    id_list=[]
    for i in result_list:
        id=[]
        i.sort(key=lambda row:(row[1]),reverse=True)
        if(len(i) < 10):
            for j in range(len(i)):
                id.append(i[j][0])
                #print(i[j])
        else:
            for j in range(10):
                id.append(i[j][0])
                #print(i[j])
        #print("---------------------------")
        id_list.append(id)
    
    return id_list
        
    

def queryInBit(bitVecQ,bitVecA):
    jacca_list=[]
    #for each query bit matrix
    for bit_matrixQ in bitVecQ:
        sim_dict={}
        #compare to all bit matrix
        for bit_matrixA in bitVecA:
            hit_bits=0
            total_bits=0
            #compare every bits in bit matrix
            for bit in range(len(bit_matrixQ)-2):
                if((bit_matrixQ[bit+1] == "1") or (bit_matrixA[bit+1] == "1")):
                    total_bits+=1
                    if(bit_matrixQ[bit+1] == bit_matrixA[bit+1]):
                        hit_bits+=1

            jacca_sim = float(hit_bits/total_bits)
            sim_dict[bit_matrixA[0]]=jacca_sim
        
        #sort similarity
        sorted_sim={}
        for sim in sorted(sim_dict, key=sim_dict.get, reverse=True):
            sorted_sim[sim]=sim_dict[sim]
        
        jacca_list.append(sorted_sim)
    
    #sort and get top 10 article
    id_list=[]
    for q_list in jacca_list:
        dict_items = q_list.items()
        top_ten = list(dict_items)[:10]
        id=[]
        for top in top_ten:
            #print(top)
            id.append(top[0])
        #print("------------------------------")
        id_list.append(id)
    return id_list

def calF1(estimated_list,ground_truth):
    for i in estimated_list:
        print(i)
    print("#######################################################")
    for i in ground_truth:
        print(i)
    print("#################################################")
    total_f1_s=0
    for i in range(len(estimated_list)):
        tp=0.0
        fp=0.0
        fn=0.0
        for j in range(len(estimated_list[i])):
            if(estimated_list[i][j] in ground_truth[i]):
                tp+=1.0
            else:
                fp+=1.0
            if(ground_truth[i][j] not in estimated_list[i]):
                fn+=1.0
        f1_s=tp/(tp + (fp+fn)/2)
        print(tp,fp,fn,f1_s)
        total_f1_s+=f1_s
        print("--------------------")
    print(total_f1_s,total_f1_s/4)

if __name__ == "__main__":
    #read files
    bitVecA= readBit('bbc/bitvector_all.csv')
    bitVecQ = readBit('bbc/bitvector_query.csv')
    
    #calculate articles features
    articleCount=len(bitVecA)
    #for part 1 (a)
    #print(len(bitVecA),len(bitVecQ))
    #print(len(bitVecA[0])-2,len(bitVecQ[0])-2)
    
    #generate prime number p
    n = len(bitVecA[0])-2
    p = genPrime(n)
    #print(n,p)
    l=10
    k=2
    m=int(articleCount)
    #start=time.process_time()
    hashTable,sigMatrix,minHashPairs,cTable=genHashTable(n,p,l,k,m,bitVecA) #generate hashtable and sigature matrix, store the hash functions
    start=time.process_time()
    #plotHeatmap(hashTable)
    estimated_list=query(hashTable,sigMatrix,bitVecQ,minHashPairs,l,k,m,n,p,cTable) #query the articles by LSH
    #print("#######################################")
    end=time.process_time()
    print((end-start)*1000)
    #start=time.process_time()
    ground_truth=queryInBit(bitVecQ,bitVecA)        #query the articles by bit matrix
    #end=time.process_time()
    #print((end-start)*1000)

    #calculate F1 score
    calF1(estimated_list,ground_truth)
    