# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA,PCA
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer, normalize
#from cbrain import utils
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#%% Data Loading
def load_df():
# method 1
    df=pd.read_csv('test_df.csv')
# method 2
#    query = r"""
#    SELECT *
#    FROM cbrain_dev.map_incident_signature_solution_TD_t yy, cbrain_dev.solution_TD_t sol
#    WHERE sol.id=yy.solution_id AND yy.backtrace<>'';
#    """
#    df = utils.query_to_df(query)
#    
    df = df.fillna('')

    return df

#%% To Distance Matrix
def backtrace_truncater(df,depth=10):
    n = df.shape[0]
    temp = []
    for k in range(n):
        bt_items = re.split(r'[,]',df.backtrace[k])[:depth]
        temp.append(','.join(bt_items))
    ds = pd.Series(temp)
    return ds

def backtrace_to_matrix(df):
    # backtrace mistake encoder
    # need to throw away .so
    count_vect = CountVectorizer(token_pattern=r'(?u)\b\w\w\w+\b',ngram_range=(1,1))
    temp = backtrace_truncater(df,depth=10)
    X_train_backtrace = count_vect.fit_transform(temp).todense()
    # tsvd = KernelPCA(n_components=1000,kernel='cosine')
    # X_train_backtrace = tsvd.fit_transform(X_train_backtrace)
    return X_train_backtrace

# site_id encoder
def site_id_encoder(df):
    n = df.shape[0]
    main_name = []
    digit_name = []
    other_name = []
    for k in range(n):
        s_items = re.split(r'(\d+|PROD|DEV|DR|TEST|TST|UAT|PRD)',df.site_id[k])
        s_txt_items = [x for x in s_items if (x and (not x.isdigit()))]
        main_name.append(s_txt_items[0][:5])
        s_id_no = [str(int(x)) for x in s_items if (x and x.isdigit())]
        digit_name.append(s_id_no)
        other_name.append(s_txt_items[1:])
    ds = pd.DataFrame({'main_name':main_name,'digit_name':digit_name,'other_name':other_name})
    return ds

def site_id_to_matrix(df,method='parse'):
    lb = LabelBinarizer()
    if method == 'original':
        X_train_site_id = lb.fit_transform(df.site_id.tolist())
    else:
        ds = site_id_encoder(df)
        mlb = MultiLabelBinarizer()
        X_main = lb.fit_transform(ds.main_name)
        X_digit = mlb.fit_transform(ds.digit_name)/5
        X_other = mlb.fit_transform(ds.other_name)/5
        X_train_site_id = np.hstack((X_main,X_digit,X_other))
        
#    tsvd = KernelPCA(n_components=100,kernel='cosine')
#    X_train_site_id = tsvd.fit_transform(X_train_site_id)
    return X_train_site_id

# description encoder
def description_encoder(df):
    n = df.shape[0]
    synopsis = []
    meaning =[]
    cause = []
    react = []
    for k in range(n):
        des_items_all = re.split(r'\n',df.description[k])
        des_items = [x for x in des_items_all if x]
        add_meaning = False
        add_cause = False
        add_react = False
        if len(des_items) == 0:
            synopsis.append('')
            meaning.append('')
            cause.append('')
            react.append('')
            continue
        synopsis.append(des_items[0])
        for j in range(len(des_items)):
            if des_items[j].startswith("Meaning:"):
                meaning.append(des_items[j][len("Meaning: "):])
                add_meaning = True
                continue
            elif des_items[j].startswith("Probable Cause:"):
                cause.append(des_items[j][len("Probable Cause: "):])
                add_cause = True
                continue
            elif des_items[j].startswith("Recommended Action:"):
                react.append(des_items[j][len("Recommended Action: "):])
                add_react = True
                continue
        if add_meaning == False:
            meaning.append("")
        if add_cause == False:
            cause.append("")
        if add_react == False:
            react.append("")
    ds = pd.DataFrame({'Synopsis':synopsis,'Meaning':meaning,
                       'Probable Cause':cause, 'Recommended Action':react},
                      index = df.incid)            
    return ds

def description_to_matrix(df, method='parse'):
    tfidf = TfidfVectorizer(token_pattern=r'(?u)[a-zA-Z-][a-zA-Z-]+\b', max_df=0.5)
    if method == 'original' :
        X_train_description = tfidf.fit_transform(df.description).todense()
    elif method == 'paragraph':
        lb = LabelBinarizer()
        ds = description_encoder(df)
        X_train_synopsis = lb.fit_transform(ds['Synopsis'])
        X_train_meaning = lb.fit_transform(ds['Meaning'])
        X_train_cause = lb.fit_transform(ds['Probable Cause'])
        X_train_react = lb.fit_transform(ds['Recommended Action'])
        X_train_description = np.hstack((X_train_synopsis,X_train_meaning,X_train_cause,
                                      X_train_react))
    elif method == 'parse':
        ds = description_encoder(df)
        X_train_synopsis = tfidf.fit_transform(ds['Synopsis'])
        X_train_meaning = tfidf.fit_transform(ds['Meaning'])
        X_train_cause = tfidf.fit_transform(ds['Probable Cause'])
        X_train_react = tfidf.fit_transform(ds['Recommended Action'])
        X_train_description = hstack((X_train_synopsis,X_train_meaning,X_train_cause,
                                      X_train_react)).todense()
    else:
        raise ValueError("Expect method name is not 'parse','original' and"
                         "'paragraph'. Got a %s instead" % method)    
#    tsvd = KernelPCA(kernel='cosine')
#    X_train_description = tsvd.fit_transform(X_train_description)
    return X_train_description

#from sklearn import preprocessing
def concatmpp_to_matrix(df):
    count_vect=CountVectorizer(token_pattern=r'(?u)[0-9a-zA-Z-]+\b')
    X_train_concatmpp = count_vect.fit_transform(df.concatmpp.tolist()).todense()
    return X_train_concatmpp

# msgid encoder
def msgid_to_matrix(df):
    lb = LabelBinarizer()
    X_train_msgid = lb.fit_transform(df.msgid.tolist())
    return X_train_msgid


# version encoder
# version encoder
def version_transform(df):
    """
    Transform and encode a version string number
    :param df: a pandas series representing a version string in form of "XX.XX.XX.XX"
    :return: a matrix with the same number of rows as df
    """
    lb = LabelBinarizer()
    num_rows = df.shape[0]
    part1 = []
    part2 = []
    part3 = []
    nonzeros = []
    for rows in range(num_rows):
        version_element = re.split(r'\W+', df.iloc[rows])
        if len(version_element) == 4:
            part1.append(version_element[0])
            part2.append(version_element[0] + '.' + version_element[1])
            part3.append(
                version_element[0] + '.' + version_element[1] + '.' + version_element[2])
            nonzeros.append(rows)
    X = normalize(np.hstack((lb.fit_transform(part1), lb.fit_transform(part2), lb.fit_transform(part3))))
    X_transform = np.zeros([num_rows, X.shape[1]])
    X_transform[nonzeros, :] = X
    return X_transform

def pde_version_to_matrix(df):
    X_train_pde_version = version_transform(df.pde_version)
    return X_train_pde_version

def dbs_version_to_matrix(df):
    X_train_dbs_version = version_transform(df.dbs_version)
    return X_train_dbs_version

# group all training encoders
def get_distance_matrix(df):
    n = df.shape[0]
    if 'backtrace' in df :
        X_train_backtrace = backtrace_to_matrix(df)
    else:
        X_train_backtrace = np.empty([n,0])
    if 'msgid' in df :
        X_train_msgid = msgid_to_matrix(df)
    else:
        X_train_msgid = np.empty([n,0])
    if 'concatmpp' in df :
        X_train_concatmpp = concatmpp_to_matrix(df)
    else:
        X_train_concatmpp = np.empty([n,0])
    if 'pde_version' in df :
        X_train_pde_version = pde_version_to_matrix(df)
    else:
        X_train_pde_version = np.empty([n,0])
    if 'dbs_version' in df :
        X_train_dbs_version = dbs_version_to_matrix(df)
    else:
        X_train_dbs_version = np.empty([n,0])
    if 'description' in df :
        X_train_description = description_to_matrix(df,method='parse')
    else:
        X_train_description = np.empty([n,0])
    if 'site_id' in df :
        X_train_site_id = site_id_to_matrix(df)
    else:
        X_train_site_id = np.empty([n,0])
## for non-sparse matrix
# we can add and adjust weights here
    X_train = np.hstack((4*X_train_backtrace, X_train_msgid, 
                         X_train_concatmpp, X_train_pde_version, 
                         X_train_dbs_version, X_train_description/4,
                         X_train_site_id/4))
    
    tsvd = KernelPCA(kernel='cosine')
    X_train = tsvd.fit_transform(X_train)
    return X_train

#label
def get_solution_id_label(df): 
    y=df.xrefdb+df.xrefid
    return y
# dimension reduction to visualize
from sklearn.manifold import TSNE

def vis_distance_matrix_2D(X_train):
    X_embedded = TSNE(n_components=2).fit_transform(X_train)
    return X_embedded
# plot
def plot_2D_matrix(X,y,label):
    plt.close()
    if X.shape[1] == 2:
        plt.scatter(X[:,0],X[:,1])
        if label == True :
            for i, txt in enumerate(y):
                plt.annotate(txt,(X[i,0],X[i,1]))       
        plt.show()
    else:
        raise ValueError("Wrong input matrix")
        
#%% Tests
df = load_df()

y = get_solution_id_label(df)
X_train = get_distance_matrix(df)
        
#%% Clustering
def get_cluster(X_train,method,k=1000):
    from sklearn import cluster 
    if method == 'kmeans':
        km = cluster.KMeans(n_clusters=k)
        km = km.fit(X_train)
        labels = km.labels_
#        inertia = km.inertia_
#        centers = km.cluster_centers_
    elif method == 'DBSCAN':
        af = cluster.DBSCAN(min_samples=1,eps=2.25)
        af = af.fit(X_train)
        labels = af.labels_
    elif method == 'Ward':    
        hie = cluster.AgglomerativeClustering(n_clusters=k,linkage="ward")
        hie = hie.fit(X_train)
        labels = hie.labels_
    elif method == 'avr':
        hie = cluster.AgglomerativeClustering(n_clusters=k,linkage="average")
        hie = hie.fit(X_train)
        labels = hie.labels_
    elif method == 'KindAP':
        import KindAP
        ki = KindAP.KindAP(n_clusters = k, algorithm = 'L', tol_in = 1e-5, 
                           max_iter_in = 500, init= old_labels, if_print=True)
        ki = ki.fit(X_train)
        labels = ki.labels_
    else:
        raise ValueError('Invalid method %s!'% method)
    return labels
#

# adjust_mutual_info_score    
from sklearn.metrics.cluster import adjusted_mutual_info_score
def evaluate_cluster(y,labels):
    score = adjusted_mutual_info_score(y,labels)
    return score

# add label column to df
# query labeled data
def map_label_solution(df,labels):
    temp = df
    if 'C_label' not in df :
        temp['C_label'] = pd.Series(labels,index=temp.index)
    return temp

# explore the constitution of cluster #lb
def explore_cluster(df,labels,lb):
    y = (df.xrefdb+df.xrefid).value_counts()
    x = (df.xrefdb+df.xrefid)[np.asarray(np.nonzero(labels==lb))[0]].value_counts()
    x.columns = lb
    output_cluster = pd.concat([x,x.div(y).dropna()],axis=1).reindex(x.index)
    output_cluster.columns=['multiplicity','perc_of_soln']
    output_cluster['perc_of_cluster'] = pd.Series(x/x.sum(),index=x.index)
    return output_cluster

#%% Clustering Test
method_name = "kmeans"
labels = get_cluster(X_train,method_name)
print (evaluate_cluster(y,labels))
#%% Output clustering stats
f = open("cluster_stat_"+ method_name +".txt","w+")
f.write("The evaluation score is %lf\n"%evaluate_cluster(y,labels))

C_count_ex = 0
inc_count_ex = 0
C_count_gr = 0
inc_count_gr = 0
C_count_g = 0
inc_count_g = 0

C_ind_ex = []

for lb in range(1500):

    if np.asarray(np.nonzero(labels==lb))[0].shape[0]>0:
        f.write ("------------------------------------------------------------\n")
        f.write ("This is Cluster Number %d\n" % lb)
        stat_cluster = explore_cluster(df,labels,lb)
        f.write (stat_cluster.to_string())
        f.write ("\n\n")
        if (stat_cluster.shape[0] == 1) and (stat_cluster.multiplicity[0] > 5):
            f.write ("Excellent Cluster because the cluster contains a uniform soln_id.\n")
            C_count_ex = C_count_ex+1
            inc_count_ex = inc_count_ex+stat_cluster.multiplicity[0]
            C_ind_ex.append(lb)
        elif (stat_cluster.shape[0] == 1) and ((stat_cluster.multiplicity[0] > 1) or (stat_cluster.perc_of_soln[0] > 0.99)):
            f.write ("Great Cluster because the cluster contains a uniform soln_id.\n")
            C_count_gr = C_count_gr+1
            inc_count_gr = inc_count_gr+stat_cluster.multiplicity[0]
        elif stat_cluster.shape[0] == 1:
            f.write ("Good Cluster because the cluster contains a uniform soln_id.\n")
            C_count_g = C_count_g+1
            inc_count_g = inc_count_g+stat_cluster.multiplicity[0]
        elif stat_cluster.perc_of_cluster[0]-stat_cluster.perc_of_cluster[1]>0.85:
            f.write ("Excellent Cluster because almost all incidents in the cluster have a uniform soln_id.\n")
            C_count_ex = C_count_ex+1
            inc_count_ex = inc_count_ex+stat_cluster.multiplicity[0]
            C_ind_ex.append(lb)
        elif stat_cluster.perc_of_cluster[0]-stat_cluster.perc_of_cluster[1]>0.5:
            f.write ("Great Cluster because the vast majority incidents in the cluster have a uniform soln_id.\n")
            C_count_gr = C_count_gr+1
            inc_count_gr = inc_count_gr+stat_cluster.multiplicity[0]
        elif (stat_cluster.multiplicity.div(stat_cluster.perc_of_soln)-stat_cluster.multiplicity).sum() < 0.05*stat_cluster.multiplicity.sum():
            f.write ("Good Cluster because almost all the solutions with the same soln_id are found.\n")
            C_count_g = C_count_g+1
            inc_count_g = inc_count_g+stat_cluster[stat_cluster.perc_of_soln>0.95].multiplicity.sum()
        elif (stat_cluster.multiplicity.sum()>10) and (stat_cluster[stat_cluster.perc_of_cluster>0.03].shape[0]<4) and (stat_cluster.perc_of_cluster[0]>0.1) :
            f.write ("Good cluster because only a small number of solutions remain without considering outliers.\n")
            C_count_g = C_count_g+1
            inc_count_g = inc_count_g+stat_cluster[stat_cluster.perc_of_cluster>0.03].multiplicity.sum()
        else:
            f.write ("Not a good cluster.\n")
f.write("*******************************************************\n")
f.write("Excellent number clusters: %lf\n" %C_count_ex)
f.write("Great number clusters: %lf \n" %C_count_gr)
f.write("Good number clusters: %lf \n" %C_count_g)
f.write("\n")
f.write("Excellent number incidents: %lf \n" %inc_count_ex)
f.write("Great number incidents: %lf \n" %inc_count_gr)
f.write("Good number incidents: %lf \n" %inc_count_g)
f.close()

#%% Plot Pie
from matplotlib.gridspec import GridSpec
legends = ['Excellent','Great','Good','Not good']
colors = ['chartreuse','gold','orange','red']
frac_C = [C_count_ex,C_count_gr,C_count_g,len(np.unique(labels))-C_count_ex-C_count_gr-C_count_g]
frac_inc = [inc_count_ex,inc_count_gr,inc_count_g,len(labels)-inc_count_ex-inc_count_gr-inc_count_g]

the_grid = GridSpec(1, 2)
fig = plt.figure()
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(frac_C, labels=legends,colors=colors, autopct='%1.1f%%', shadow=True)
plt.title("Clusters")

plt.subplot(the_grid[0, 1], aspect=1)
plt.pie(frac_inc, labels=legends,colors=colors, autopct='%1.1f%%', shadow=True)
plt.title("Incidents")

plt.suptitle("Clustering performace evaluation by "+ method_name, size=16)
fig.savefig("cluster_eval_"+method_name+".png")
plt.close()

#%% Plot
#X_embedded = vis_distance_matrix_2D(X_train)
#plot_2D_matrix(X_embedded,y,label=True)
#plot_2D_matrix(X=X_embedded,y=labels,label=True)
#%% Further Insight to Excellent Cluster
df=map_label_solution(df,labels)
fp = open('Suspicious_'+method_name+'.txt',"w+")
for lb in range(200):
    if lb in C_ind_ex:
        fp.write("---------------------------------\n")
        fp.write("This is cluster %d\n"%lb)
        stat_cluster = explore_cluster(df,labels,lb)       
        fp.write(stat_cluster.to_string())
        fp.write ("\n\n")
        if stat_cluster.shape[0]>1:
            fp.write("The detailed information is :\n")
            fp.write(df[df.C_label==lb].to_string())
            fp.write ("\n\n")
fp.close()