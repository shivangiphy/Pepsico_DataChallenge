#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels as sm


# In[2]:


sns.set_context("poster")
sns.set_style("ticks")


# In[3]:


df = pd.read_excel("./shelf-life-study-data-for-analytics-challenge_prediction.xlsx")


# In[4]:


df.head()


# In[5]:


for c in df.columns:
    df_t = df[c].fillna("NONE")
    print(c, df[c].dtype, df_t.dtype)
    print(df_t.value_counts())
    if df[c].dtype == "float64":
        fig, ax = plt.subplots(1,1)
        ax = df.loc[~df[c].isnull(), c].hist(log=True, ax=ax)
        ax = df.loc[df[c].isnull(), c].fillna(0.0).hist(bins=1, log=True, alpha=0.5, ax=ax)
        ax.set_title(c)
        sns.despine(offset=10)


# ## Product wise splits

# In[6]:


df.columns


# In[7]:


linestyles = {
    "Warm Climate": "-",
    "NONE": ":",
    "Cold Climate": "--",
    "High Temperature and Humidity": "-."
}
for product_type in sorted(df['Product Type'].unique()):
    fig, ax = plt.subplots(1,1, figsize=(12, 8))
    for sample_id in df.loc[df["Product Type"] == product_type, "Sample ID"].unique():
        df_t = df[df["Sample ID"] == sample_id]
        linestyle_keys = df_t["Storage Conditions"].fillna("NONE").unique()
        assert len(linestyle_keys) == 1, f"Many process types: {linestyle_keys}"
        df[df["Sample ID"] == sample_id].plot(
            x="Sample Age (Weeks)", y="Difference From Fresh", kind="line", ax=ax, label=sample_id,
            lw=2, linestyle=linestyles[linestyle_keys[0]], marker="o"
        )
    plt.axhline(y=20, lw=2, linestyle="--", color="0.5")
    plt.ylabel("Difference From Fresh")
    plt.title(f"Product type = {product_type}")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    sns.despine(offset=10)


# In[8]:


from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


# In[9]:


def plot_isotonic(x, y, **plot_kwargs):
    model = IsotonicRegression(y_min=0).fit(x, y)
    x_pred = np.arange(x.min(), x.max(), 0.1)
    y_pred = model.predict(x_pred)
    plt.plot(x_pred, y_pred, **plot_kwargs)
       


# In[10]:


def get_least_x_above_threashold(ax):
    lines = ax.get_children()[2]
    critical_point = [dict(x=x, y=y) for x,y in lines.get_xydata() if y > 20][0]
    return critical_point

def get_critical_x_for_all_sample(g, sample_clusters):
    critical_points = {
        cluster_id: get_least_x_above_threashold(ax)
        for cluster_id, ax in zip(g.col_names, g.axes.flatten())
    }
    sample_critical_points = {
        sample_id: dict(cluster_id=cluster_id, **critical_points[cluster_id])
        for sample_id, cluster_id in sample_clusters.items()
    }
    return sample_critical_points
    

def plot_product(product_type, sample_clusters, regression_order=1):
    sns.set_context("poster", font_scale=1, rc={"lines.linewidth": 2.5})
    sns.set_style("ticks")
    g = sns.FacetGrid(
        data=df[df["Product Type"] == product_type].assign(
            sample_cluster=lambda x: x["Sample ID"].map(sample_clusters)
        ),
        col="sample_cluster", 
        #hue="Sample ID",
        col_wrap=3,
        height=5,
        aspect=1.5,
        xlim=[0,150],
        ylim=[0,40]
    )

    """g.map(
        plt.plot,
        "Sample Age (Weeks)", 
        "Difference From Fresh", 
        lw=2,
        marker="+",
        linestyle="None"
    )"""

    g.map(
        sns.regplot,
        "Sample Age (Weeks)", 
        "Difference From Fresh",
        order=regression_order,
        marker="+"
    )

    g.map(
        plt.axhline,
        y=20,
        lw=1, linestyle="--", color="0.5"
    )

    g.map(
        plot_isotonic,
        "Sample Age (Weeks)", 
        "Difference From Fresh", 
        lw=2,
        linestyle="--"
    )
    
    sample_critical_points = get_critical_x_for_all_sample(g, sample_clusters)
    df_sample_critical_points = pd.DataFrame(sample_critical_points).T
    display(df_sample_critical_points)
    return sample_critical_points, g
    


# ## Instructions for creating the plots based on product clusters
# 
# * Update the sample_clusters with the "Sample ID" and the cluster_id. 
# * You can assign any cluster ID as you like. 
# * I have used the convention of product_type and then a number starting from 0 to denote clusters
# * Create a new cell for each product type and assign the variable `product_type` for the product type you want to investigate. 
# * Run `sample_critical_points, g = plot_product(product_type, sample_clusters)`, this will generate the plot and will create a table for critical points (the smallest x for which the y is greater than the threshhold *(20)*). 
# * The default regression order is `regression_order=1` you can pass a higher number to fit higher order regression models.
# 

# In[11]:


all_sample_critical_points = {}


# In[12]:


sample_clusters = {
    '1411697-1': "A1", 
    '1614359-1-Southern': "A1", 
    '1614359-1-Northern': "A0",
    '1614359-2-Southern': "A1", 
    '1614359-2-Northern': "A0", 
    '1614359-3-Southern': "A1",
    '1614359-3-Northern': "A0",
    '1714706-1': "A1"
}


# In[13]:


product_type = "A"
sample_critical_points, g = plot_product(product_type, sample_clusters, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[14]:


ax = g.axes[0]
lines = ax.get_children()[2]
plt.plot(lines.get_xdata(), lines.get_ydata())
plt.axhline(y=20)
critical_point = [(x,y) for x,y in lines.get_xydata() if y > 20][0]
plt.axhline(y=critical_point[1], lw=1, linestyle="--", color="0.5")
plt.axvline(x=critical_point[0], lw=1, linestyle="--", color="0.5")
offset = 20
arrowprops = dict(
    arrowstyle = "->",
    color="k",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")
bbox = dict(boxstyle="round", fc="0.8")
plt.annotate(
    f"(x={critical_point[0]:.2f}, y={critical_point[1]:.2f})", 
    critical_point,
    xytext=(0, 30),
    bbox=bbox, arrowprops=arrowprops
)


# In[15]:


sample_clusters_I = {
    '1513327-1': "I1", 
    '1613454-1': "I1",
    '1613454-2': "I1"
}


# In[16]:


product_type = "I"
sample_critical_points, g = plot_product(product_type, sample_clusters_I, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[17]:


ax = g.axes[0]
lines = ax.get_children()[2]
plt.plot(lines.get_xdata(), lines.get_ydata())
plt.axhline(y=20)
critical_point = [(x,y) for x,y in lines.get_xydata() if y > 20][0]
plt.axhline(y=critical_point[1], lw=1, linestyle="--", color="0.5")
plt.axvline(x=critical_point[0], lw=1, linestyle="--", color="0.5")
offset = 20
arrowprops = dict(
    arrowstyle = "->",
    color="k",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")
bbox = dict(boxstyle="round", fc="0.8")
plt.annotate(
    f"(x={critical_point[0]:.2f}, y={critical_point[1]:.2f})", 
    critical_point,
    xytext=(0, 30),
    bbox=bbox, arrowprops=arrowprops
)


# In[18]:


sample_clusters_G = {
    '1411761-1': "G1", 
    '1512296-1': "G1",
    '1411878-1': "G2",
    '1513340-1': "G2",
    '1513340-2': "G2",
    '1613998-1': "G2"
}


# In[19]:


product_type = "G"
sample_critical_points, g = plot_product(product_type, sample_clusters_G, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[20]:


sample_clusters_C = {
    '1411014-1': "C1", 
    '1512578-1': "C1",
    '1512900-1': "C2",
    '1513338-1': "C2",
    '1513376-1': "C2",
    '1513377-1': "C2",
    '1614028-1': "C2"
}
#1512654_1 has 0 slope  (warm climate + processing stabilizer added)


# In[21]:


product_type = "C"
sample_critical_points, g = plot_product(product_type, sample_clusters_C, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[22]:


sample_clusters_D = {
    '1310186-1': "D1", 
    '1411909-1': "D1",
    '1512305-1': "D1",
    '1512627-1-Southern': "D1",
    '1715434-1': "D1"
}


# In[23]:


product_type = "D"
sample_critical_points, g = plot_product(product_type, sample_clusters_D, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[24]:


sample_clusters_F = {
    '1411193-1': "F1", 
    '1714526-1': "F1",
    '1614373-1': "F1",
    '1411145-1': "F1",
    '1512098-1': "F1",
    '1411193-2': "F2",
    '1512421-1': "F1",
    '1512598-1': "F2",
    '1512712-1': "F2",
    '1613611-1': "F2",
    '1613611-2': "F2",
    '1613878-1': "F2",
    '1715394-1': "F2"
}


# In[25]:


product_type = "F"
sample_critical_points, g = plot_product(product_type, sample_clusters_F, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[26]:


sample_clusters_B = { 
    '1411647-1': "B1",
    '1512161-1': "B1",
    '1512161-2': "B1",
    '1512161-3': "B1",
    '1512161-4': "B1",
    '1512534-1': "B1",
    '1512534-2': "B1",
    '1512637-1': "B1",
    '1512637-2': "B1",
    '1613616-1': "B1",
    '1613782-1': "B1",
    '1613793-1': "B1"
}


# In[27]:


product_type = "B"
sample_critical_points, g = plot_product(product_type, sample_clusters_B, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[28]:


sample_clusters_H = { 
    '1310881-1': "H1",
    '1310893-1': "H1",
    '1310893-2': "H1",
    '1411032-1': "H1",
    '1411253-1': "H1",
    '1411933-1': "H1",
    '1512305-2': "H1",
    '1513103-1': "H1",
    '1513115-1': "H1",
    '1513116-1': "H1",
    '1513130-1': "H1",
    '1513280-1': "H1",
    '1513280-2': "H1",
    '1613520-1': "H1",
    '1613520-2': "H1",
    '1613531-1': "H1",
    '1614242-1': "H1",
    '1614242-2': "H1",
    '1714648-1': "H1"
}


# In[29]:


product_type = "H"
sample_critical_points, g = plot_product(product_type, sample_clusters_H, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[30]:


sample_clusters_E = { 
    '1310746-1': "E1",
    '1310746-2': "E1",
    '1310746-3': "E1",
    '1310746-4': "E1",
    '1310746-5': "E1",
    '1310746-6': "E1",
    '1310957-1': "E1",
    '1411084-1': "E1",
    '1411084-2': "E1",
    '1411465-1': "E1",
    '1411999-1': "E1",
    '1412001-1': "E1",
    '1412003-1': "E1",
    '1412034-1': "E1",
    '1412048-1': "E1",
    '1512111-1': "E1",
    '1512884-1': "E1",
    '1512884-2': "E1",
    '1512884-3': "E1",
    '1512886-1': "E1",
    '1512886-2': "E1",
    '1512888-1': "E1",
    '1512890-1': "E1",
    '1512890-2': "E1",
    '1513368-1': "E1",
    '1613469-1': "E1",
    '1613530-1': "E1",
    '1613606-1': "E1",
    '1613606-2': "E1",
    '1613648-1': "E1",
    '1613704-1': "E1",
    '1613716-1': "E1",
    '1614480-1': "E1",
    '1614480-2': "E1",
    '1714863-1': "E1",
    '1815739-1': "E1",
    '1815741-1': "E1",
    '1815742-1': "E1"
}


# In[31]:


product_type = "E"
sample_critical_points, g = plot_product(product_type, sample_clusters_E, regression_order=1)
all_sample_critical_points.update(sample_critical_points)


# In[32]:


df.head()


# In[33]:


len(all_sample_critical_points)


# In[34]:


df["Sample ID"].unique().shape


# In[35]:


skipped_items = set(df["Sample ID"].unique().tolist()) - set(all_sample_critical_points.keys())
skipped_items


# In[36]:


set(all_sample_critical_points.keys()) - set(df["Sample ID"].unique().tolist())


# In[37]:


df[df["Sample ID"].isin(skipped_items)][["Product Type", "Sample ID"]]


# In[38]:


for k,v in all_sample_critical_points.items():
    print(k,v)


# In[39]:


overall_mean = np.mean(list([x["x"] for x in all_sample_critical_points.values()]))
overall_mean


# In[40]:


df.columns


# In[41]:


df["Prediction"].head()


# In[42]:


df_predictions = df[["Sample ID"]].assign(
    predictions=df["Sample ID"].apply(lambda k: all_sample_critical_points.get(k, {"x": overall_mean})["x"])
)
df_predictions.head()


# In[43]:


df_predictions.to_csv("./predictions.csv", index=False)


# In[ ]:




