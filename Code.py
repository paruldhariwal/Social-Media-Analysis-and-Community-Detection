!pip install python-igraph
from igraph import *

from igraph import *
g=Graph()





def addVertex(g,name_str):
    try:
        if(name_str not in g.vs['name']):
            print('Inserted node ',name_str)
            g.add_vertex(name=name_str)
        else:
            print ('Node ',name_str,' already present')
            print(g.vs.find(name_str).index)
    except KeyError:
        g.add_vertex(name=name_str)
    return g



def write_tuple_to_file(f,t):
    string=str(t[0])+' '+str(t[1])+'\n'
    f.write(string)

def retrieve_edge_name_tuple(g,t):
    a=(g.vs[t[0]]['name'],g.vs[t[1]]['name'])
    return a



def load_dataset(fileName,g):
    fileNums=[0]
    for i,eachNum in enumerate(fileNums):
        print(eachNum)
        fileName="/content/0.edges"
        print('fileName=',fileName)
        f=open(fileName)
        line=f.readline()
        while(line!=''):
            c=(line.split())
            g=addVertex(g,c[0])
            g=addVertex(g,c[1])
            print('Adding ',c[0],'-->',c[1])
            g.add_edge(c[0],c[1])
            line=f.readline()
    g.simplify()
    return




load_dataset('abd',g)
print(len(g.vs))


def calculate_eigen(g):
    eigen=g.evcent(directed=False)
    for i in range(1,6):
        maxVal=max(eigen)
        print(i,'==node',g.vs[eigen.index(maxVal)]['name'],' with score of ',maxVal)
        eigen.remove(maxVal)
    eigen=g.evcent(directed=False)
    return eigen


def calculate_closeness(g):
    close=g.closeness(g.vs)
    for i in range(1,6):
        maxVal=max(close)
        print(i,'==node',g.vs[close.index(maxVal)]['name'],' with score of ',maxVal)
        close.remove(maxVal)
    close=g.closeness(g.vs)
    return close


def calculate_between(g):
    between=g.betweenness(g.vs)
    for i in range(1,6):
        maxVal=max(between)
        print(i,'==node',g.vs[between.index(maxVal)]['name'],' with score of ',maxVal)
        between.remove(maxVal)
    between=g.betweenness(g.vs)
    return between

print('Eigen Vector')
global eigen
eigen=calculate_eigen(g)

global close
global between
print('Closeness')
close=calculate_closeness(g)
print('Betweenness')
between=calculate_between(g)

print(close)
from igraph import *
G=Graph()

load_dataset('/content/0.edges',G)


N=len(G.vs)
layt=G.layout('kk', dim=3)

labels=[]
print(type(labels))
for eachNde in G.vs:
    labels.append(eachNde['name'])

Edges=list()
print(type(Edges))
for eachTuple in G.es:
    Edges.append(eachTuple.tuple)

Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
Zn=[layt[k][2] for k in range(N)]# z-coordinates
Xe=[]
Ye=[]
Ze=[]

for e in Edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]

!pip install chart-studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Scatter3d, Line, Marker, ColorBar, Layout, Scene, XAxis, YAxis, ZAxis, Margin, Annotations, Annotation, Font, Data, Figure
from plotly.offline import plot
import numpy as np




trace1 = Scatter3d(
    x=Xe,
    y=Ye,
    z=Ze,
    mode='lines',
    line=dict(color='rgb(125,125,125)', width=1),
    hoverinfo='none'
)

trace2 = Scatter3d(
    x=Xn,
    y=Yn,
    z=Zn,
    mode='markers',
    name='actors',
    marker=dict(
        symbol='circle',  # Use a valid symbol like 'circle'
        color=eigen,
        size=6,
        colorbar=dict(
            title='Colorbar'
        ),
        colorscale='Viridis',
        line=dict(color='rgb(158,18,130)', width=0.5)
    ),
    text=labels,
    hoverinfo='text'
)
axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = Layout(
    title="3D Visualization of the Facebook nodes",
    width=1000,
    height=1000,
    showlegend=False,
    scene=Scene(
        xaxis=XAxis(axis),
        yaxis=YAxis(axis),
        zaxis=ZAxis(axis),
    ),
    margin=Margin(t=100),
    hovermode='closest',
    annotations=Annotations([
        Annotation(
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=Font(size=14)
        )
    ]),
)

data=Data([trace1, trace2])

fig = Figure(data=data, layout=layout)

# Save the visualization as an HTML file
plot(fig, filename='3d_visualization.html')

from google.colab import files

#generating other datasets

# Download the HTML file
files.download('3d_visualization.html')

!pip install igraph
from igraph import *
def load_dataset(g):
    fileNums=[0]
    for i,eachNum in enumerate(fileNums):
        print(eachNum)
        fileName="/content/"+str(eachNum)+".edges"
        print('fileName=',fileName)
        f=open(fileName,'a+')
        nodeID=eachNum
        line=f.readline()
        print(line)
        print("here")
        while(line!=''):
            c=(line.split())
            g=addVertex(g,c[0])
            g=addVertex(g,c[1])
            print('Adding ',c[0],'-->',c[1])
            g.add_edge(c[0],c[1])
            line=f.readline()
    g.simplify()
    return


def addVertex(g,name_str):
    try:
        if(name_str not in g.vs['name']):
            print('Inserted node ',name_str)
            g.add_vertex(name=name_str)
        else:
            print ('Node ',name_str,' already present')
            print(g.vs.find(name_str).index)
    except KeyError:
        g.add_vertex(name=name_str)
    return g
def write_tuple_to_file(f,t):
    string=str(t[0])+' '+str(t[1])+'\n'
    f.write(string)

def retrieve_edge_name_tuple(g,t):
    a=(g.vs[t[0]]['name'],g.vs[t[1]]['name'])
    return a

g=Graph()
# load_dataset(g)
import random

def generate_datasets(g,num,train_filename,valid_filename,test_filename):
    load_dataset(g)
    print(g)
    f=open(train_filename,'a+');
    global train_num
    train_num=int(len(g.es)*0.5)
    print('train length=',train_num)
    global test_num
    test_num=int(len(g.es)*0.25)
    global valid_num
    valid_num=int(len(g.es)*0.15)
    print('valid num=',valid_num)
    for i in range(train_num):
        edgeSet=g.es;
        r=random.randint(0,len(edgeSet)-1);
        t=edgeSet[r].tuple
        g.delete_edges(t);
        print('len of es=',len(edgeSet))
        write_tuple_to_file(f,retrieve_edge_name_tuple(g,t))
    f.close()
    f=open(test_filename,'a+');
    for i in range(test_num):
        edgeSet=g.es;
        r=random.randint(0,len(edgeSet)-1);
        print('r=',r)
        t=edgeSet[r].tuple
        g.delete_edges(t);
        print('len of es=',len(edgeSet))
        write_tuple_to_file(f,retrieve_edge_name_tuple(g,t))
    f.close()
    f=open(valid_filename,'a+');
    for i in range(valid_num):
        edgeSet=g.es;
        if(len(g.es)==0):
            break
        else:
            print('len of es=',len(edgeSet))
            r=random.randint(0,len(edgeSet)-1);
            print('r=',r)
            t=edgeSet[r].tuple
            g.delete_edges(t);
            write_tuple_to_file(f,retrieve_edge_name_tuple(g,t))
            if(len(g.es)==0):
                f.close()
                break
    print ('I am done')







# In[12]:

generate_datasets(G,len(G.es)/10,'/content/SelfDatasets/sample_train.edges','/content/SelfDatasets/sample_valid.edges','/content/SelfDatasets/sample_test.edges')


# In[13]:

# train length=1426 valid=427
print(train_num)


# In[15]:

#Generate negative examples with class label 0.0
mat=G.get_adjacency()

pool_of_empty=list()
count=0
for i,entireNode in enumerate(mat):
    for j,eachVal in enumerate(entireNode):
        if(eachVal==0 and i!=j):
            count+=1;
            pool_of_empty.append((i,j))
print('count=',count)


original_pool_of_empty = pool_of_empty.copy()
new_pool_of_empty = [each for each in pool_of_empty if each[0] != 0]
pool_of_empty = new_pool_of_empty


# print(pool_of_empty)
for each in pool_of_empty:
    if(each[0]==0):
        pool_of_empty.remove(each)


# In[21]:

import random
def generate_negative_examples(pool,trainfilename,trainnum,validfilename,validnum,testfilename,testnum):
    f=open(trainfilename,'a+')
    for i in range(0,trainnum):
        r=random.randint(0,len(pool)-1);
        t=pool[r];
        pool.remove(t);
        f.write(str(t[0])+' '+str(t[1])+'\n');
    f.close()
    f=open(validfilename,'a+')
    for i in range(0,validnum):
        r=random.randint(0,len(pool)-1);
        t=pool[r];
        pool.remove(t);
        f.write(str(t[0])+' '+str(t[1])+'\n');
    f.close()
    f=open(testfilename,'a+')
    for i in range(0,testnum):
        r=random.randint(0,len(pool)-1);
        t=pool[r];
        pool.remove(t);
        f.write(str(t[0])+' '+str(t[1])+'\n');
    f.close()




# In[22]:

generate_negative_examples(original_pool_of_empty,'/content/SelfDatasets/negative_train.edges',train_num,'/content/SelfDatasets/negative_valid.edges',valid_num,'/content/SelfDatasets/negative_test.edges',test_num)

from igraph import *

nodes=set()
fileNums=[0]
for i,eachNum in enumerate(fileNums):
    print(eachNum)
    fileName="/content/"+str(eachNum)+".edges"
    print('fileName=',fileName)
    f=open(fileName)
    nodes.add(eachNum)
    line=f.readline()
    while(line!=''):
        c=line.split()
        nodes.add(c[0])
        print('added ',c[0])
        nodes.add(c[1])
        print('added ',c[1])
        line=f.readline()
    print('Length=',len(nodes))
    print(nodes)

#LINK PREDICTION
import numpy as np
from igraph import *
global num_of_feat
num_of_feat=347
def load_dataset(fileName,g):
    fileNums=[0]
    for i,eachNum in enumerate(fileNums):
        print(eachNum)
        print('fileName=',fileName)
        f=open(fileName)
        line=f.readline()
        while(line!=''):
            c=(line.split())
            g=addVertex(g,c[0])
            g=addVertex(g,c[1])
            print('Adding ',c[0],'-->',c[1])
            g.add_edge(c[0],c[1])
            line=f.readline()
    g.simplify()
    return

def load_neg_dataset(fileName,g):
    fileNums=[0]
    for i,eachNum in enumerate(fileNums):
        print(eachNum)
        print('fileName=',fileName)
        f=open(fileName)
        nodeID=eachNum
        line=f.readline()
        while(line!=''):
            c=(line.split())
            g=addVertex(g,c[0])
            g=addVertex(g,c[1])
            print('Adding ',c[0],'-->',c[1])
            g.add_edge(c[0],c[1])
            line=f.readline()
    g.simplify()
    return

def load_and_shape_input(file_name):
    a=np.loadtxt(fname=file_name)
    slice_D =[a[i][1:] for i in range(0,num_of_feat)]
    c=np.asarray(slice_D)
    return c

def load_shape_input(file_name_array):
    features=dict()
    for eachname in file_name_array:
        file_name='/content/'+str(eachname)+'.feat'
        a=np.loadtxt(file_name)
        for eachFeat in a:
            features[eachFeat[0]]=np.asarray(eachFeat[1:])
    return features

def addVertex(g,name_str):
    try:
        if(name_str not in g.vs['name']):
            print('Inserted node ',name_str)
            g.add_vertex(name=name_str)
        else:
            print ('Node ',name_str,' already present')
            print(g.vs.find(name_str).index)
    except KeyError:
        g.add_vertex(name=name_str)
    return g

def write_tuple_to_file(f,t):
    string=str(t[0])+' '+str(t[1])+'\n'
    f.write(string)

def retrieve_edge_name_tuple(g,t):
    a=(g.vs[t[0]]['name'],g.vs[t[1]]['name'])
    return a

def write_tuple_to_file(f,t):
    string=str(t[0])+' '+str(t[1])+'\n'
    f.write(string)

def retrieve_edge_name_tuple(g,t):
    a=(g.vs[t[0]]['name'],g.vs[t[1]]['name'])
    return a

# Load Feature vectors
li={0}
node_feat=load_shape_input(li)


# In[5]:

g=Graph()
load_dataset('/content/SelfDatasets/sample_train.edges',G)


not_g=Graph()
load_dataset('/content/SelfDatasets/negative_train.edges',not_g)

print(type(node_feat))
for eachKey in node_feat.values():
    print(len(eachKey))
    print(type(eachKey))

def make_class_arrays(g,datalabel):
    output_list=list()
    edgeSet=g.es
    for eachTuple in edgeSet:
        tuple_name=retrieve_edge_name_tuple(g,eachTuple.tuple)
        print('eachTuple=',tuple_name)
        output=np.add(node_feat[np.float64(tuple_name[0])],node_feat[np.float64(tuple_name[1])])
        output_list.append(output)
    return np.asarray(output_list)

valid_g=Graph()
load_dataset('/content/SelfDatasets/sample_valid.edges',valid_g)
# node_feat=load_and_shape_input("Datasets/facebook/0.feat")


valid_not_g=Graph()
load_dataset('/content/SelfDatasets/negative_valid.edges',valid_not_g)

x_positive=make_class_arrays(G,1)
x_negative=make_class_arrays(not_g,1)
print(x_positive.shape)
print(x_negative.shape)

valid_x_positive=make_class_arrays(valid_g,1)
valid_x_negative=make_class_arrays(valid_not_g,1)
print(valid_x_positive.shape)
print(valid_x_negative.shape)

y_positive=np.full(shape=(x_positive.shape[0],1),fill_value=1.0)
y_negative=np.full(shape=(x_negative.shape[0],1),fill_value=0.0)
print(y_positive.shape)
print(y_negative.shape)

valid_y_positive=np.full(shape=(valid_x_positive.shape[0],1),fill_value=1.0)
valid_y_negative=np.full(shape=(valid_x_negative.shape[0],1),fill_value=0.0)
print(valid_x_positive.shape)
print(valid_x_negative.shape)
print(valid_y_positive.shape)
print(valid_y_negative.shape)

train_X=np.append(x_positive,x_negative,axis=0)
train_Y=np.append(y_positive,y_negative,axis=0)

valid_X=np.append(valid_x_positive,valid_x_negative,axis=0)
valid_Y=np.append(valid_y_positive,valid_y_negative,axis=0)
print(type(x_positive))
print(valid_X.shape)
print(type(x_negative))
print(valid_Y.shape)
print(type(y_positive))
print(y_positive.shape)
print(train_X.shape)
print(1592+1748)

!pip install scikit-learn
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)

reg.fit(X=train_X[:-1],y=train_Y[:-1])

reg.predict(train_X[-1:])

len(reg.predict(valid_X))
np.mean((reg.predict(valid_X)-valid_Y)**2)

from sklearn.metrics import log_loss
log_loss(valid_Y,reg.predict(valid_X))

from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X=train_X[:-1],y=train_Y[:-1].ravel())

from sklearn.metrics import log_loss
log_loss(valid_Y,clf_svm.predict(valid_X))

#Community Detection
!pip install python-igraph
import igraph as ig
import numpy as np

# Load the edge list to create an igraph graph
edge_file_path = '/content/107.edges'  # Replace with the path to your .edges file
G = ig.Graph.Read_Ncol(edge_file_path, directed=False)

# Perform community detection using the Louvain method
partition = G.community_multilevel()

# Print the communities
for idx, community in enumerate(partition):
    print(f"Community {idx + 1}: {community}")

# You can also access the community membership for each node
community_membership = partition.membership
print("Community membership for nodes:")
print(community_membership)

