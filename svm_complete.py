#导入相关的库，这里的copy模块是为了后面做深拷贝用的
import numpy as np
import copy
import matplotlib.pyplot as plt

#读入本地的数据，读出来的数据格式都是列表形式的
def load_data(filename):
    dataset=[]
    labelset=[]
    f=open(filename)
    for line in f.readlines():
        new_line=line.strip().split()
        dataset.append([float(new_line[0]),float(new_line[1])])
        labelset.append(int(new_line[2]))
    return dataset,labelset

#随机在0到m之间选择一个整型数，用来进行一对alpha值的更新，SMO算法的思想是每次选取两个不同的alpha进行更新
def select_j(i,m):
    """parameter:i 表示第一个选定的alpha,对应第i行数据
                 m  表示数据集的总数
    """
    j=-1
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

#用来限定alpha值的上下限
def regular(H,L,a):
    if a>H:
        a=H
    if a<L:
        a=L
    return a

#简化版的SMO算法的实现，最终返回模型需要的参数alpha和b,这里后面我直接紧接着根据alpha也把w求出来了。
#注意：自己亲测试过，简化版的算法实现起来确实稍显简单，但缺点是速度比较慢，这一点你可以在与后面的完整版
#实现进行比较一下就深有体会了。但是本人觉得最大的缺点并不是这个，二是这个模型很难拟合的比较理想，试过狠多参数
#c基本都是同样的结果，这一点我觉得可能是与完整版SMO算法在实现上的最大缺陷了。
#之所以称为简化版最主要的简化就是在于第二个alpha的选定上，是从第一个alpha所对应的样本除外的其他样本中随机选取一个alpha
def simple_SMO(data,label,C,toler,max_iters):
    """C:对于线性SVM来说，C是在调参时唯一需要调节的参数了，也是非常重要的参数，它的目的在于使得分离间隔尽可能大的同时也要保证
         误分类点的个数尽可能少，调节二者平衡的重要参数。
       toler:容错率，这是自己定义的误差范围，它的由来是由KKT条件决定的，本身的kkT是有三个约束条件，但是在用于程序判断时可以等效成
       通过容错率来进行判断的两个条件，这个在于自己给定，无需进行调节。
       max_iter:最大循环次数，用于控制外围的循环，防止无限制循环
       """
    #整篇数据操作均把它转换成矩阵的形式，方便统一进行运算
    data=np.mat(data)
    label=np.mat(label).transpose()
    m,n=np.shape(data)
    alphas=np.mat(np.zeros((m,1)))
    b=0
    iter=0
    while(iter<max_iters):
        alpha_pairs_changed=0
        for i in range(m):
            Fxi=float(np.multiply(alphas,label).T*(data*data[i].T))+b
            Ei=Fxi-float(label[i])
            if ((label[i]*Ei<-toler) and (alphas[i]<C)) or ((label[i]*Ei>toler) and (alphas[i]>0)):#由kkt条件确定的，具体公式推导可见博客https://blog.csdn.net/youhuakongzhi/article/details/86660281
                j=select_j(i,m)
                Fxj=float(np.multiply(alphas,label).T*(data*data[j].T))+b
                Ej=Fxj-float(label[j])
                alphaIold=copy.deepcopy(alphas[i])#深拷贝
                alphaJold=copy.deepcopy(alphas[j])
                if label[i]!=label[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if H==L:continue
                eta=data[i]*data[i].T+data[j]*data[j].T-2*data[i]*data[j].T#迭代更新中需要eta，相当于是最小化函数对alphaj求最小值时的二阶导数，所以后面需要保证它大于0
                if eta<=0:
                    continue
                alphas[j]+=label[j]*(Ei-Ej)/eta#往下均是迭代更新的公式，需要自己去查阅相关的SVM理论公式推导，理解过程就好了
                alphas[j]=regular(H,L,alphas[j])
                if abs(alphas[j]-alphaJold)<0.00001:
                    continue
                alphas[i]+=label[i]*label[j]*(alphaJold-alphas[j])
                b1=b-Ei-label[i]*(alphas[i]-alphaIold)*data[i]*data[i].T-label[j]*\
                    (alphas[j]-alphaJold)*data[i]*data[j].T
                b2=b-Ej-label[i]*(alphas[i]-alphaIold)*data[i]*data[j].T-label[j]*\
                    (alphas[j]-alphaJold)*data[j]*data[j].T
                if 0<alphas[i]<C:
                    b=b1
                elif 0<alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2
                alpha_pairs_changed+=1         
        if alpha_pairs_changed==0:#这里的alpha_pairs_changedz主要是用来防止alpha在若干次循环中都没有更新的情况下还在继续无休止的循环下去
            iter+=1
        else:
            iter=0
    w=np.mat(np.zeros((n,1)))#往后直接根据公式求出w
    for i in range(m):
        w+=np.multiply(alphas[i]*label[i],data[i].T)
    return alphas,b,w

#定义数据结构，用于完整版SMO算法在实现中的数据存储
class data_structer(object):
    def __init__(self,data,label,C,toler):
        self.data=np.mat(data)
        self.label=np.mat(label).transpose()
        self.C=C
        self.toler=toler
        self.m=np.shape(np.mat(data))[0]
        self.n=np.shape(np.mat(data))[1]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.error=np.mat(np.zeros((self.m,2)))
        self.k=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.k[:,i]=kernel_trans(self.data,self.data[i],('rbf',1.25))
        
#计算所对应数据的误差值
def calculator_error(os,i):
    Fxi=float(np.multiply(os.alphas,os.label).T*(os.data*os.data[i].T))+os.b
    #Fxi=float(np.multiply(os.alphas,os.label).T*os.k[:,i])+os.b
    Ei=Fxi-os.label[i]
    return Ei

#这里就是与简化版SMO算法最大的不同的地方，选择两者误差最大的样本所对应的alpha进行更新
def select_j_ErrorMax(os,Ei,i):
    maxj=-1
    maxerror_j=0
    maxerror_diff=0
    not_zeros_index=np.nonzero(os.error[:,0].A)[0]
    if len(not_zeros_index)>1:
        for j in not_zeros_index:
            if j==i:continue
            Ej=calculator_error(os,j)
            error_diff=abs(Ei-Ej)
            if error_diff==0:
                continue
            elif error_diff>maxerror_diff:
                maxerror_diff=error_diff
                maxerror_j=Ej
                maxj=j
    else:
        maxj=select_j(i,os.m)
        maxerror_j=calculator_error(os,maxj)
    return maxj,maxerror_j

#更新误差值
def update_error(os,i):
    Ei=calculator_error(os,i)
    os.error[i]=[1,Ei]

#基本上是SMO算法最核心的地方了，除了选择alphaj的原则不同以外，其余均和简化的SMO算法一样
def inner(os,i):
    Ei=calculator_error(os,i)
    if (os.label[i]*Ei<-os.toler and os.alphas[i]<os.C) or (os.label[i]*Ei>os.toler and\
       os.alphas[i]>0):
        j,Ej=select_j_ErrorMax(os,Ei,i)
        alphaIold=copy.deepcopy(os.alphas[i])
        alphaJold=copy.deepcopy(os.alphas[j])
        if os.label[i]!=os.label[j]:
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:
            L=max(0,os.alphas[i]+os.alphas[j]-os.C)
            H=min(os.C,os.alphas[i]+os.alphas[j])
        eta=os.data[i]*os.data[i].T+os.data[j]*os.data[j].T-2*os.data[i]*os.data[j].T
        if eta<=0:
            return 0
        os.alphas[j]+=os.label[j]*(Ei-Ej)/eta
        os.alphas[j]=regular(H,L,os.alphas[j])
        update_error(os,j)
        if abs(os.alphas[j]-alphaJold)<0.0001:
            return 0
        os.alphas[i]+=os.label[i]*os.label[j]*(alphaJold-os.alphas[j])
        update_error(os,i)
        b1=os.b-Ei-os.label[i]*(os.alphas[i]-alphaIold)*os.data[i]*os.data[i].T-\
           os.label[j]*(os.alphas[j]-alphaJold)*os.data[i]*os.data[j].T
        b2=os.b-Ej-os.label[i]*(os.alphas[i]-alphaIold)*os.data[i]*os.data[j].T-\
            os.label[j]*(os.alphas[j]-alphaJold)*os.data[j]*os.data[j].T
        #这里的表示形式在非线性中引进核函数之后是写成下面的形式，本人在线性数据集中引入核函数，结果运用下面的更新公式是错误的，切记一点，如果是现象可分的千万不要再去引入核函数
        #b1=os.b-Ei-os.label[i]*(os.alphas[i]-alphaIold)*os.k[i,i]-os.label[j]*\
        #    (os.alphas[j]-alphaJold)*os.k[i,j]
        #b2=os.b-Ej-os.label[i]*(os.alphas[i]-alphaIold)*os.k[i,j]-os.label[j]*\
        #    (os.alphas[j]-alphaJold)*os.k[j,j]
        if 0<os.alphas[i]<os.C:
            os.b=b1
        elif 0<os.alphas[j]<os.C:
            os.b=b2
        else:
            os.b=(b1+b2)/2.0
        return 1
    else:
        return 0
    
#完整SMO的算法实现，外环控制循环次数以及遍历的形式，首先在完整的数据集上遍历，寻找满足条件的alpha对进行更新，
#在已经实现了完整数据集上更新alpha对的基础上，再寻找非边缘的alpha值进行更新
def SMO(data,label,C,toler,max_iter):
    os=data_structer(data,label,C,toler)
    iter=0
    set_iter=True
    alpha_pairs_changed=0
    while (iter<max_iter) and (alpha_pairs_changed>0 or set_iter==True):
        if set_iter:
            for i in range(os.m):
                
                alpha_pairs_changed+=inner(os,i)
            iter+=1
        else:
            not_side_index=np.nonzero((os.alphas.A>0)*(os.alphas.A<os.C))[0]
            for index in not_side_index:
                alpha_pairs_changed+=inner(os,index)
            iter+=1
        if set_iter:
            set_iter=False
        if alpha_pairs_changed==0:
            set_iter=True
        print("alpha_pairs_changed:%d,iter:%d"%(alpha_pairs_changed,iter))
    w=np.mat(np.zeros((os.n,1)))
    for i in range(os.m):
        w+=np.multiply(os.alphas[i]*os.label[i],os.data[i].T)
    return os.alphas,os.b,w
        
#核转化函数，将低维数据映射到高维，其实可以看出就是两个向量点乘的结果，把所有的结果放到一个统一的矩阵中，矩阵的维数是由点乘的向量的维数来确定的
def kernel_trans(data,A,kernel):
    """parameter:kernel是核函数的形式说明，是一个元组，一个数据是字符串名称，第二个是对应的核函数的参数
    """
    data=np.mat(data)
    A=np.mat(A).transpose()
    m,n=np.shape(data)
    k=np.mat(np.zeros((m,1)))
    for i in range(m):
        if kernel[0]=='linear':
            k[i]=data[i]*A
        elif kernel[0]=='rbf':
            diff=data[i].T-A
            k[i]=diff.T*diff
            k=np.exp(-k/(2*kernel[1]**2))
        else:raise Nameerror("the wrong name!")
    return k

#这里的测试函数我之所以把它注释掉是因为测试了很多次误差都是极高的，算法的实现本身是没有问题的，问题在于自身用的数据集是线性的数据集，所以自己坑了自己一把
##def test_kernel(kernel=('rbf',0.055)):
##    dataset,labelset=load_data('testSet.txt')
##    alphas,b,w=SMO(dataset,labelset,0.6,0.001,100,kernel)
##    data=np.mat(dataset)
##    label=np.mat(labelset).transpose()
##    svIndex=np.nonzero(alphas.A>0)[0]
##    svdata=data[svIndex]
##    svlabel=label[svIndex]
##    m=np.shape(data)[0]
##    n=np.shape(svdata)[0]
##    error_count=0
##    k=np.mat(np.zeros((n,m)))
##    for i in range(m):
##        k[:,i]=kernel_trans(svdata,data[i],kernel)
##        y_pre=k[:,i].T*np.multiply(alphas[svIndex],svlabel)+b
##        print(y_pre,label[i])
##        if np.sign(y_pre)!=np.sign(label[i]):
##            error_count+=1
##    error_ratio=float(error_count)/m
##    return error_ratio


def test():
    dataset,labelset=load_data('testSet.txt')
    alphas,b,w=SMO(dataset,labelset,0.6,0.001,100,('rbf',1.25))
    data=np.mat(dataset)
    label=np.mat(labelset).transpose()
    m=np.shape(data)[0]
    error_count=0
    for i in range(m):
        y=float(np.multiply(alphas,label).T*(data*data[i].T))+b
        if np.sign(y)!=np.sign(label[i]):
            error_count+=1
    error_ratio=float(error_count)/m
    return error_ratio

#最终的可视化，数据集以及分类边界，在运用完整版算法时效果很好，简化版比较差
#这里自己还掉了个坑就是自己自SMO算法的公式更新中运用了核矩阵，所以最终的结果总是错误
def plot():
    dataset,labelset=load_data('testSet.txt')
    alphas,b,w=SMO(dataset,labelset,0.01,0.01,200)
    w=np.array(w)
    b=np.array(b)
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(len(dataset)):
        if labelset[i]==1:
            xcord1.append(dataset[i][0])
            ycord1.append(dataset[i][1])
        elif labelset[i]==-1:
            xcord2.append(dataset[i][0])
            ycord2.append(dataset[i][1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,c='blue',s=50)
    ax.scatter(xcord2,ycord2,c='green',s=70)
    x=list(np.arange(1,10,0.01))
    y=[(-w[0]*i-b[0])/w[1] for i in x]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    
    
    
    
    
            
            
        
        
        
        
    


    

    


         
                
                
        
            
        
        
        
       
        
        
        
            
        
    
    
    


        
           
           
        
        
            
            
    
    
