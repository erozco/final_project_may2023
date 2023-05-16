'''
Author: Erick Orozco

'''

from pyspark import SparkContext
from itertools import islice
from itertools import combinations
import time
import sys
import os
import math
import json
import xgboost
from sklearn.metrics import mean_squared_error

#packages only used to convert rdd to pandas df
from pyspark.sql import SparkSession
import pandas as pd

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable



#final formatting and outputting
def output_table(rdd,output_file):
    final = rdd.filter(lambda x: x is not None).collect()
    with open(output_file, 'w') as output:
        output.write('user_id, business_id, prediction\n')
        for f in sorted(final):
            output.write(f[0]+','+f[1]+','+str(f[2])+'\n')
        output.close()

def xgb_setup(xg_train,xg_val,user,business,rdd,val_rdd,c_info):

    tot_rdd = rdd.union(val_rdd)

    b_attributes = business.map(lambda r: (r['business_id'],r['attributes']))
    unique_business_rdd = tot_rdd.map(lambda b: b[1]).distinct().zipWithIndex()
    b_attributes = unique_business_rdd.leftOuterJoin(b_attributes).map(lambda r: r[1]).collectAsMap()

    attire = []
    wifi = []
    alc = []
    for key in b_attributes:
        try:
            attire.append(b_attributes[key]['RestaurantsAttire'])
            wifi.append(b_attributes[key]['WiFi'])
            alc.append(b_attributes[key]['Alcohol'])
        except:
            pass

    unique_attire = {k: v for v, k in enumerate(list(set(attire)))}
    unique_wifi = {k: v for v, k in enumerate(list(set(wifi)))}
    unique_alc = {k: v for v, k in enumerate(list(set(alc)))}


    #keep important features needed
    u_info = user.map(lambda u: (u["user_id"], (u["review_count"], u["average_stars"])))
    b_info = business.map(lambda b: (b["business_id"], (b["review_count"], b["stars"],\
    getInt(b["attributes"], "RestaurantsPriceRange2"),getTrue(b["attributes"], "BusinessAcceptsCreditCards"),getTrue(b["attributes"], "RestaurantsTakeOut")\
    ,getTrue(b["attributes"], "RestaurantsReservations"),getMulti(b["attributes"], "Alcohol",unique_alc),getMulti(b["attributes"], "WiFi",unique_wifi)\
    ,getMulti(b["attributes"], "RestaurantsAttire",unique_attire))))

    check_d = c_info.map(lambda r: (r['business_id'],sum(r['time'].values()))).reduceByKey(lambda a,b: a+b).collectAsMap()

    #keep dictionaries, important for later
    user_d = u_info.collectAsMap()
    bus_d = b_info.collectAsMap()

    #calulate features 
    u_star,b_star = averageStar(u_info,b_info)
    u_count,b_count = averageCount(u_info,b_info)

    train = preprocessPandas(xg_train,user_d,bus_d,u_star,u_count,b_star,b_count,check_d)
    val = preprocessPandas(xg_val,user_d,bus_d,u_star,u_count,b_star,b_count,check_d)

    return train,val,user_d,bus_d,u_star,b_star,u_count,b_count


def averageStar(user,business):
    return user.map(lambda x: float(x[1][1])).mean(),business.map(lambda x: float(x[1][1])).mean()


def averageCount(user,business):
    return user.map(lambda x: float(x[1][0])).mean(),business.map(lambda x: float(x[1][0])).mean()


def getTrue(x, y):
    if x:
        if y in x.keys():
            if x.get(y) == 'True':
                return 1
            else:
                return 0
    else:
        return None

def getInt(x, y):
    if x:
        if y in x.keys():
            return int(x.get(y))
    else:
        return 0

def getMulti(x,y,my_dict):
    if x:
        if y in x.keys():
            return my_dict[x.get(y)]
    else:
        return None


#prepare pandas dataframes for model input
#NOTE: all calculations are done with rdd prior, this is purely to
#      set up for the xgbregressor
def preprocessPandas(x,user,business,u_stars,u_count,b_stars,b_count,check_ins):
    countU = []
    starsU = []
    countB = []
    starsB = []
    price = []
    cards = []
    take = []
    res = []
    alc = []
    wifi = []
    attire = []
    checks = []

    #place values in correct place using dictionaries
    for index, row in x.iterrows():
        user_id = row["user_id"]
        if user_id in user.keys():
            countU.append(user[user_id][0])
            starsU.append(user[user_id][1])
        else:
            countU.append(u_count)
            starsU.append(u_stars)

    for index, row in x.iterrows():
        business_id = row["business_id"]
        if business_id in business.keys():
            countB.append(business[business_id][0])
            starsB.append(business[business_id][1])
            price.append(business[business_id][2])
            cards.append(business[business_id][3])
            take.append(business[business_id][4])
            res.append(business[business_id][5])
            alc.append(business[business_id][6])
            wifi.append(business[business_id][7])
            attire.append(business[business_id][8])
            
        else:
            countB.append(b_count)
            starsB.append(b_stars)
            price.append(0)
            cards.append(None)
            take.append(None)
            res.append(None)
            alc.append(None)
            wifi.append(None)
            attire.append(None)
        if business_id in check_ins.keys(): 
            checks.append(check_ins[business_id])
        else:
            checks.append(None)

    x["user_count"] = countU
    x["user_stars"] = starsU
    x["business_count"] = countB
    x["business_stars"] = starsB
    x["business_price_range"] = price
    x["accepts_cards"] = cards
    x["takeout"] = take
    x["reservation"] = res
    x["alcohol"] = alc
    x["wifi"] = wifi
    x["attire"] = attire
    x["check_ins"] - checks

    return x

def modelTrain(train):
    x = train.drop(["stars", "user_id", "business_id"], axis=1)
    y = train["stars"]
    model = xgboost.XGBRegressor(n_estimators = 100,max_depth=5)
    return model.fit(x, y)

def modelTest(model,test):
    x = test.drop(["stars", "user_id", "business_id"], axis=1)
    return model.predict(x)

def joinPred(preds,val):
    df = val[['user_id', 'business_id']]
    df['prediction'] = preds

    f_rdd = spark.createDataFrame(df).rdd.map(list)

    f_rdd = f_rdd.filter(lambda x: x is not None)
    
    final = f_rdd.map(lambda x: (x[0],x[1],x[2])) 
    
    return final

def cal_rmse(rdd,val_rdd):
    test = val_rdd.map(lambda x: ((x[0],x[1]),x[2]))
    pred = rdd.map(lambda x: ((x[0],x[1]),x[2]))
    rmse = test.join(pred).map(lambda r: (r[1][0],r[1][1])).map(lambda x: (x[0]-x[1])**2).mean()
    return math.sqrt(rmse)


if __name__ == '__main__':

    start = time.time()
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    val_rdd   = sc.textFile(test_file)
    val_rdd = val_rdd.mapPartitionsWithIndex(lambda index, iteartor: islice(iteartor, 1, None) if index == 0 else iteartor)\
        .map(lambda x: tuple(x.split(","))).map(lambda r: (r[0],r[1]))
    rdd = sc.textFile(folder_path+'data_train.csv')
    rdd  = rdd.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)\
    .map(lambda row: row.split(',')).map(lambda r: (r[0],r[1],float(r[2])))
    b_info = sc.textFile(folder_path+r'business.json').map(lambda x: json.loads(x))
    u_info = sc.textFile(folder_path+r'user.json').map(lambda x: json.loads(x))
    c_info = sc.textFile(folder_path+r'checkin.json').map(lambda x: json.loads(x))
    #xgboost 
    #------------------------------------------------------#
    train_xg = pd.read_csv(folder_path + "data_train.csv") 
    val_xg = pd.read_csv(sys.argv[2])

    train_xg,val_xg,user_d,bus_d,u_star,b_star,u_count,b_count = xgb_setup(train_xg,val_xg,u_info,b_info,rdd,val_rdd,c_info)

    #model = modelTrain(train_xg)
    #preds = modelTest(model,val_xg)




    #final_rdd = joinPred(preds,val_xg)

    #output_table(final_rdd, output_file)

    print('Duration: '+str(time.time()-start))

    #rmse = cal_rmse(final,val_rdd)
    #print("RMSE: "+str(rmse))

'''
Results:

Validation RMSE = 0.97973


'''