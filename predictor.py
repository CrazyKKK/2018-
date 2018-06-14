# -*- coding: utf-8 -*-
import datetime
import re
from copy import deepcopy
import random
import math
import copy

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result
    #物理服务器CPU核数,内存大小,硬盘大小,资源维度名称,flavor,测试起止时间
    phy_CPU,phy_MEM,phy_dis,inp_type,flavor,startTime,endTime=input_inf(input_lines)
    #phy_CPU=input_inf(input_lines)
    #需要分配的flavor预测,flavor总数
    pre,pre_tol,pre_con=pre_flavor(ecs_lines,flavor,startTime,endTime)
    predict=deepcopy(pre)
    allo = allocate(phy_CPU,phy_MEM,inp_type,pre)
    result=result_out(predict,pre_tol,allo)
    return result
    
def input_inf(input_lines):
    #获得需要的部分数据集
    input_va=[]
    for item in input_lines:
        values = item.strip().split('\t')
        input_va.append(values)
    #找到所有空行的位置
    ent=[ i for i in range(len(input_va)) if input_va[i] == ['']]
    #物理服务器CPU核数 内存大小（GB） 硬盘大小（GB）
    inp_phy=[]
    inp_phy=input_va[0][0].strip().split()
    phy_CPU=int(inp_phy[0])
    phy_MEM=int(inp_phy[1])
    phy_dis=int(inp_phy[2])    
    #获得需要测试的flavor
    flavor=input_va[ent[0]+2:ent[1]]
   #需要优化的资源维度名称
    inp_type=input_va[ent[1]+1][0]
    #起止时间
    startTime=input_va[ent[2]+1][0]
    endTime=input_va[ent[2]+2][0]
    return phy_CPU,phy_MEM,phy_dis,inp_type,flavor,startTime,endTime
    #return phy_CPU
    
def pre_flavor(ecs_lines,flavor,startTime,endTime):
    #测试集起止
    startDay=startTime.strip().split()[0]
    endDay=endTime.strip().split()[0]    
    flavor_list,flavor_inf=[],{}
    for line in flavor:
        fla=line[0].strip().split()
        flavor_inf.update({fla[0]:[fla[1],fla[2]]})
        flavor_list.append(fla[0])
    esc_train0,days=[],[]
    for item in ecs_lines:
        values = item.strip().split('\t')
        #uuid = values[0]
        flavorName = values[1]
        createTime = values[2]
        creatDay=createTime.strip().split()[0]
        days.append(creatDay)
        esc_train0.append([flavorName,creatDay])
    max_day,min_day=max(days),min(days)
    esc_train=[]
    for line in esc_train0:
        Name=line[0]
        if Name in flavor_list:
            esc_train.append(line)
    sort_data=sorted(esc_train,key=lambda x:x[0])
    dict_allData,len_num=getData(sort_data,max_day,min_day)
    pre_days={}
    for fla in flavor_list:
        if fla not in dict_allData.keys():
           pre_days.update({fla:[0]})
           flavor_list.remove(fla)
    need_data=outlier(dict_allData,flavor_list)
    #获得预测字典
    format="%Y-%m-%d"
    strptime = datetime.datetime.strptime
    days = (strptime(endDay, format) - strptime(startDay, format)).days
    testlist=[]
    for i in range(len_num,len_num+days):
        testlist.append([1,i])   
    for fla in flavor_list:
        xArr,yArr=[],[]
        for day in need_data[fla]:
            xArr.append(day[1][:2])
            yArr.append(day[1][2])
        yHat=stocGradAscent(testlist,xArr,yArr)
        pre_days.update({fla:yHat})
    pre_con={}
    for fla in pre_days:
        num=float(sum(pre_days[fla]))
        if num<=0:
            pre_con.update({fla:0})
        else:
            pre_con.update({fla:int(round(num+0.1))})
    
    pre,all_con={},[]
    for fla in flavor_inf:
        num=int(re.sub("\D","",fla))
        con=pre_con[fla]
        all_con.append(con)
        pre.update({num:[con,int(flavor_inf[fla][0]),int(flavor_inf[fla][1])/1024]})
    pre_tol=sum(all_con)
    return pre,pre_tol,pre_con
    
#从数据集中计算需要的数据，最终字典格式
def getData(datalist,max_day,min_day):
    #flavor,date分成两个列表存储
    flavor,date=[],[]
    for line in datalist:
        flavor.append(line[0])
        date.append(line[1])
    daynum_list=dateRange(min_day, max_day)
    len_num=len(daynum_list)
    #构建0元素字典
    dict_0,k,dict_1={},0,{}
    for j in daynum_list:
        dict_0.update({j:[1,k,0]})
        k+=1
    dict_1=deepcopy(dict_0)
    #对flavor去重复并按标号顺序排序,实际上并没有严格按照数字大小，是按数字从左到右位置比较大小，但是相同数字的在一起
    setflavor=sorted(set(flavor))
    #寻找每个flavor对应的日期，并统计不同日期出现的次数
    start=0
    dict_all={}
    pre_flavor=setflavor[0]
    for fla in setflavor[1:]:
        end=flavor.index(fla)
        #由于在allData()中对整体数据集按flavor排过序，则相同时间的位置连续，寻找下一个时间出现的第一个位置可找到上一个时间的截止位置
        flavor_x=date[start:end]
        start=end
        setflavor_x=set(flavor_x)
        dict_x = {}
        for day in setflavor_x:
            dict_x.update({day:[1,daynum_list.index(day),flavor_x.count(day)]})
        dict_0.update(dict_x)
        list_0=sorted(dict_0.items(),key=lambda x:x)
        dict_all.update({pre_flavor:list_0})
        dict_0.update(dict_1)
        pre_flavor=fla
    #最后一个时间的数据计算
    flavor_x=date[start:]
    setflavor_x=set(flavor_x)
    dict_x = {}
    for day in setflavor_x:
        dict_x.update({day:[1,daynum_list.index(day),flavor_x.count(day)]})
    dict_0.update(dict_x)
    list_0=sorted(dict_0.items(),key=lambda x:x)
    dict_all.update({pre_flavor:list_0})
    return dict_all,len_num

def outlier(dict_allData,flavor_list):
    need_Data={}
    for fla in flavor_list:
        data_fla=deepcopy(dict_allData[fla])
        point=[]
        for day in data_fla:
            point.append(day)
        point_sort_all=sorted(point,key=lambda x:x[1][2])
        point_sort=[]
        for line in point_sort_all:
            point_sort.append(line[1][2])
        con_all=len(point_sort)
        if con_all%2 !=0:
            median=point_sort[int(con_all/2)]
            low=point_sort[:int(con_all/2)]
            con_low=len(low)
            if con_low%2 !=0:
                Q1=low[int(con_low/2)]
            else:
                Q1=(low[int(con_low/2)]+low[int(con_low/2)-1])/2.0
            up=point_sort[int(con_all/2)+1:]
            con_up=len(up)
            if con_up%2 !=0:
                Q2=up[int(con_up/2)]
            else:
                Q2=(up[int(con_up/2)]+up[int(con_up/2)-1])/2.0
        else:
            median=(point_sort[int(con_all/2)]+point_sort[int(con_all/2)-1])/2.0
            low=point_sort[:int(con_all/2)]
            con_low=len(low)
            if con_low%2 !=0:
                Q1=low[int(con_low/2)]
            else:
                Q1=(low[int(con_low/2)]+low[int(con_low/2)-1])/2.0
            up=point_sort[int(con_all/2):]
            con_up=len(up)
            if con_up%2 !=0:
                Q2=up[int(con_up/2)]
            else:
                Q2=(up[int(con_up/2)]+up[int(con_up/2)-1])/2.0
        IQR=Q2-Q1+1
        out_low=Q1-4*IQR
        out_up=Q2+4*IQR
        i=len(point_sort)-1
        while point_sort[i] > out_up:
            del(point_sort_all[i])
            i-=1
        j=0
        while point_sort[j] < out_low:
            del(point_sort_all[i])
            j+=1
        new_data=sorted(point_sort_all,key=lambda x:x[1][1])
        need_Data.update({fla:new_data})
    return need_Data
    
#生成连续的时间段
def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days+1
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]

def stocGradAscent(testArr,xArr, yArr):
    m=len(xArr)
    n=len(xArr[0])
    weights = [1]*n
    for i in range(m):      #i为样本点下标
        alpha =0.0000457
        #randIndex = int(np.random.uniform(0,len(dataIndex)))    #随机生成一个0到len(dataIndex)之间的整数（左闭右开），若前面没有加int则返回的是浮点数
        h = sum(list(map(lambda x, y: x * y,xArr[i],weights)))
        error = yArr[i] - h
        s=list(map(lambda x, y: x * y,[alpha * error,alpha * error],xArr[i]))
        weights = list(map(lambda x, y: x + y,weights,s))
        #print(weights)#输出为array
    k = len(testArr)
    yHat = [0]*k
    for i in range(k):
        yHat[i]=sum(list(map(lambda x, y: x * y,testArr[i],weights)))
    return yHat

def result_out(pre,pre_tol,allo):
    result=[]
    pre_list=sorted(pre.keys(),key=lambda x:x)
    allo_t=sorted(allo.items(),key=lambda x:x[1][0])
    allo_list=[ID[0] for ID in allo_t]
    allo_tol=len(allo_list)
    result.append(str(pre_tol))
    for fla in pre_list:
        str_line='flavor'+'%d'%fla+' '+'%d'%pre[fla][0]
        result.append(str_line)
    result.append('')
    result.append(str(allo_tol))
    for phy in allo_list:
        str1=''
        for fn in allo[phy][1]:
            str0=str(fn)+' '
            str1=str1+str0
        str_need=str1[:-1]
        str_line='%d'%allo[phy][0]+' '+str_need
        result.append(str_line)
    return result

def allocate(CPUSIZE,MEMSIZE,TYPE,parameters):
    results = {}
    CPUsizecoll={}
    MEMsizecoll={}
    #parameters = parameters_value()

    #parameters = sorted(parameters.items(), key=lambda dict1: dict1[0], reverse=True)
    stop = 0
    sum = 500

    if (TYPE == "CPU"):
        # 倒序排列
        tempparameters = sorted(parameters.items(), key=lambda dict1: dict1[1][1], reverse=True)
        # print(tempparameters)
        lenth = len(parameters)

        T = 300.0  # 模拟退火初始温度
        Tmin = 1  # 模拟退火终止温度
        r = 0.999  # 温度下降系数

        lenth1 = len(parameters)
        while (lenth1 > 0):
            for i in range(len(parameters) - 1):
                while ((tempparameters[i][1][1] == tempparameters[i + 1][1][1]) and (
                            tempparameters[i][1][2] > tempparameters[i + 1][1][2])):
                    temp = tempparameters[i]
                    tempparameters[i] = tempparameters[i + 1]
                    tempparameters[i + 1] = temp
            lenth1 = lenth1 - 1
        for i in range(len(parameters) - 1):
            while ((tempparameters[i][1][2] == tempparameters[i][1][1] * 2) and (
                tempparameters[i][1][1] == tempparameters[i + 1][1][1])):
                temp = tempparameters[i]
                tempparameters[i] = tempparameters[i + 1]
                tempparameters[i + 1] = temp
        peff = 500.0
        # g=0
        while (T > Tmin):
            # 投骰子，交换两个顺序
            parameters1 = copy.deepcopy(parameters)
            changeF = random.randint(0, lenth - 1)
            changeB = random.randint(0, lenth - 1)

            # print score
            tempparameters1 = copy.deepcopy(tempparameters)
            # print "jiaohuan"
            # print tempparameters
            # print "--------------------------------"

            for x in range(len(parameters1)):
                stop = stop + tempparameters[x][1][0]
            # 最外面for循环，对每个物理机进行装包
            for n in range(sum):
                CPUsizecoll["CPUsize" + str(n)] = CPUSIZE
                MEMsizecoll["MEMsizecoll" + str(n)] = MEMSIZE
                templist = []
                # 对每种规格物理机进行遍历
                for i in range(len(parameters1)):
                    k = 0
                    # 直到一个物理机满了才跳出循环
                    while (((parameters1.get(tempparameters[i][0])[1]) <= CPUsizecoll["CPUsize" + str(n)]) and (
                                parameters1.get(tempparameters[i][0])[2] <= MEMsizecoll["MEMsizecoll" + str(n)]) and (
                                parameters1.get(tempparameters[i][0])[0] > 0)):
                        # 每放置一次，更新一下物理机剩余资源
                        CPUsizecoll["CPUsize" + str(n)] = CPUsizecoll["CPUsize" + str(n)] - tempparameters[i][1][1]
                        MEMsizecoll["MEMsizecoll" + str(n)] = MEMsizecoll["MEMsizecoll" + str(n)] - \
                                                              tempparameters[i][1][2]
                        # parameters[i][1][0] = parameters[i][1][0] - 1
                        # parameters.get(tempparameters[i][0])[0] = parameters.get(tempparameters[i][0])[0] - 1

                        # 放置一次，该种规格虚拟机数量-1
                        up = parameters1.get(tempparameters[i][0])[0] - 1
                        parameters1[tempparameters[i][0]][0] = up

                        stop = stop - 1
                        k = k + 1
                    if (k != 0):
                        templist.extend(["flavor" + str(tempparameters[i][0]), k])
                    # print(templist)

                    results["PHID" + str(n + 1)] = [n + 1, templist]

                # 所有虚拟机都放置了，则跳出循环
                if (stop == 0):
                    break

            # print len(results)
            score = len(results) - 1

            CPUusage = CPUSIZE - CPUsizecoll["CPUsize" + str(n)]

            # print results
            csum = 0.0
            for i in range(len(parameters)):
                csum = csum + parameters[tempparameters[i][0]][1] * parameters[tempparameters[i][0]][0]
            eff = csum / (len(results) * CPUSIZE)

            EFF1 = CPUusage / (CPUSIZE + 0.0)
            # tempscore = n-1

            finalscore = score + EFF1
            # print finalscore
            # print tempparameters
            sj = random.uniform(1, 2)

            # print CPUusage
            # print EFF1
            # print results
            # print sj
            # print peff
            tempresults = copy.deepcopy(results)

            if (finalscore <= peff):
                peff = finalscore
                presults = copy.deepcopy(tempresults)
                results = {}

                tempparameters = copy.deepcopy(tempparameters1)
                # print "cpu"
                # print peff
                # print score
                # print presults
                # parameters = copy.deepcopy(parameters1)
                # print tempparameters1
            elif (math.exp(peff - finalscore) / T > sj):
                peff = finalscore
                presults = copy.deepcopy(results)
                results = {}
                tempparameters = copy.deepcopy(tempparameters1)
                # print "********"
                # print finalscore
                # print math.exp(-(finalscore - peff) / T)
                # print sj
                # parameters = copy.deepcopy(parameters1)
            # tempparameters = copy.deepcopy(tempparameters1)
            # print tempparameters
            # print peff


            middle = tempparameters[changeF]
            tempparameters[changeF] = tempparameters[changeB]
            tempparameters[changeB] = middle

            '''
            if (eff >= peff):
                peff = eff
                presults = copy.deepcopy(results)
                results = {}
                #print results


                tempparameters = copy.deepcopy(tempparameters1)
                #parameters = copy.deepcopy(parameters1)
                #print tempparameters1
            elif ((math.exp(-(peff-eff)/T))<sj):
                peff = eff
                presults = copy.deepcopy(results)
                results = {}
                tempparameters = copy.deepcopy(tempparameters1)
                #parameters = copy.deepcopy(parameters1)
            #print "CPU:"
            #print peff
            '''
            T = r * T  # 一次循环结束，温度降低
            # g = g+1
            # print g
            # print T
    if (TYPE == "MEM"):
        # 倒序排列
        tempparameters = sorted(parameters.items(), key=lambda dict1: dict1[1][1], reverse=True)
        # print(tempparameters)
        lenth = len(parameters)

        T = 300.0  # 模拟退火初始温度
        Tmin = 1  # 模拟退火终止温度
        r = 0.999  # 温度下降系数

        peff = 500.0

        while (T > Tmin):
            # 投骰子，交换两个顺序
            parameters1 = copy.deepcopy(parameters)
            changeF = random.randint(0, lenth - 1)
            changeB = random.randint(0, lenth - 1)

            tempparameters1 = copy.deepcopy(tempparameters)
            # print "jiaohuan"
            # print tempparameters
            # print "--------------------------------"

            for x in range(len(parameters1)):
                stop = stop + tempparameters[x][1][0]
            # 最外面for循环，对每个物理机进行装包
            for n in range(sum):
                CPUsizecoll["CPUsize" + str(n)] = CPUSIZE
                MEMsizecoll["MEMsizecoll" + str(n)] = MEMSIZE
                templist = []
                # 对每种规格物理机进行遍历
                for i in range(len(parameters1)):
                    k = 0
                    # 直到一个物理机满了才跳出循环
                    while (((parameters1.get(tempparameters[i][0])[1]) <= CPUsizecoll["CPUsize" + str(n)]) and (
                                parameters1.get(tempparameters[i][0])[2] <= MEMsizecoll["MEMsizecoll" + str(n)]) and (
                                parameters1.get(tempparameters[i][0])[0] > 0)):
                        CPUsizecoll["CPUsize" + str(n)] = CPUsizecoll["CPUsize" + str(n)] - tempparameters[i][1][1]
                        MEMsizecoll["MEMsizecoll" + str(n)] = MEMsizecoll["MEMsizecoll" + str(n)] - \
                                                              tempparameters[i][1][2]

                        # parameters[i][1][0] = parameters[i][1][0] - 1
                        # parameters.get(tempparameters[i][0])[0] = parameters.get(tempparameters[i][0])[0] - 1
                        up = parameters1.get(tempparameters[i][0])[0] - 1
                        parameters1[tempparameters[i][0]][0] = up
                        stop = stop - 1
                        k = k + 1
                    if (k != 0):
                        templist.extend(["flavor" + str(tempparameters[i][0]), k])
                        # print(templist)
                    results["PHID" + str(n + 1)] = [n + 1, templist]
                if (stop == 0):
                    break

            score = len(results) - 1
            # print MEMsizecoll["MEMsizecoll" + str(n)]
            MEMusage = MEMSIZE - MEMsizecoll["MEMsizecoll" + str(n)]

            csum = 0.0
            for i in range(len(parameters)):
                csum = csum + parameters[tempparameters[i][0]][2] * parameters[tempparameters[i][0]][0]
            eff = csum / (len(results) * MEMSIZE)

            EFF1 = MEMusage / (MEMSIZE + 0.0)
            # tempscore = n-1

            finalscore = score + EFF1
            # print finalscore
            sj = random.uniform(1, 2)
            tempresults = copy.deepcopy(results)

            if (finalscore <= peff):
                peff = finalscore
                presults = copy.deepcopy(tempresults)
                results = {}

                tempparameters = copy.deepcopy(tempparameters1)
                # parameters = copy.deepcopy(parameters1)
                # print tempparameters1
                # print "mem"
                # print peff
                # print presults
            elif (math.exp(peff - finalscore) / T > sj):
                peff = finalscore
                presults = copy.deepcopy(results)
                results = {}
                tempparameters = copy.deepcopy(tempparameters1)
                print "********"
                print peff
            '''
            msum = 0.0
            for i in range(len(parameters)):
                msum = msum + parameters[tempparameters[i][0]][2] * parameters[tempparameters[i][0]][0]
                #print msum
            eff = msum / (len(results) * MEMSIZE)
            sj = random.uniform(0, 0.5)




            #print sj
            if (eff >= peff):
                peff = eff
                presults = copy.deepcopy(results)
                results = {}
                # print results
                tempparameters = copy.deepcopy(tempparameters1)
                #print "-----"
                #print peff
                # parameters = copy.deepcopy(parameters1)
                # print tempparameters1
            else:
                if((math.exp(-(peff - eff) / T)) < sj):
                    print "zhuyile"

                    print math.exp(-(peff - eff) / T)
                    print sj
                    peff = eff
                    presults = copy.deepcopy(results)
                    results = {}
                    tempparameters = copy.deepcopy(tempparameters1)
                # parameters = copy.deepcopy(parameters1)
                #print peff
            #print "MEM"
            #print peff
            '''
            middle = tempparameters[changeF]
            tempparameters[changeF] = tempparameters[changeB]
            tempparameters[changeB] = middle

            T = r * T  # 一次循环结束，温度降低

    return presults

