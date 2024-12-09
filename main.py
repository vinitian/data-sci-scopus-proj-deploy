import streamlit as st
import networkx as nx
import pandas as pd
import json
import plotly.graph_objects as go
from collections import Counter
from os import listdir
from os.path import isfile, join
import pandas as pd
from itertools import combinations
import numpy as np
import random
import pickle
import sklearn

st.set_page_config(layout="wide")

#----------LINE GRAPH OF KEYWORDS/SUBJECT AREA-----------#

df = pd.read_csv("df.csv", index_col=0)
df2 = pd.read_csv("df2.csv", index_col=0)
df3 = pd.read_csv("df3.csv", index_col=0)
df4 = pd.read_csv("df4.csv", index_col=0)

df_transposed = df.T
df2_transposed = df2.T
df3_transposed = df3.T
df4_transposed = df4.T

#----------NETWORK GRAPH OF RELATED KEYWORDS-----------#

path = 'sorted_papers/'
fileNames = [f for f in listdir(path) if isfile(join(path, f))]

tag_counter = Counter()
co_occurrence = Counter()

for i in range(0,29217):
    with open(path+fileNames[i], 'r', encoding="utf8") as file:
        d = json.load(file)
        if d["keywords"] != []:
            kw = d["keywords"]
            tag_counter.update(kw)
            for pair in combinations(kw, 2):
                co_occurrence[frozenset(pair)] += 1

top_tags = [tag for tag, _ in tag_counter.most_common(30)]
filtered_co_occurrence = {
    pair: count for pair, count in co_occurrence.items()
    if all(tag in top_tags for tag in pair)
}


edges = []
for pair, weight in filtered_co_occurrence.items():
    if len(pair) == 2:
        edges.append((list(pair)[0], list(pair)[1], weight))
edges_df = pd.DataFrame(edges, columns=['node1', 'node2', 'weight'])

edges_df.sort_values(by='weight', ascending=False, inplace=True)
edges_df = edges_df.head(30)

G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['node1'], row['node2'], weight=row['weight'])

if "pos" not in st.session_state:
    st.session_state.pos = nx.spring_layout(G, k=2, iterations=50)

pos = st.session_state.pos

node_x, node_y, node_sizes, labels = [], [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_sizes = [np.log(tag_counter[node] + 1) * 4 for node in G.nodes()]
    labels.append(node)

edge_traces = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']
    edge_traces.append(
        go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=max(weight / 10, 1), color='white'),
            mode='lines'
        )
    )

def random_rgb():
    return f'rgba({random.randint(50, 250)}, {random.randint(50, 250)}, {random.randint(50, 250)}, 1)'

node_colors = [random_rgb() for _ in G.nodes()]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    marker=dict(size=node_sizes, color=node_colors, opacity=1),
    text=list(G.nodes()),
    textposition="top center",
    textfont=dict(color="#6acdff"),
)

fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(
    showlegend=False,
    margin=dict(b=0, l=0, r=0, t=0),
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    dragmode='pan'
)

#----------BARCHART----------#

subjdf = pd.read_csv("uniqueCountSubjArea.csv", index_col=0)

subjdf = subjdf.rename(index={10: "Multidisciplinary",
    11: "Agricultural and Biological Sciences",
    12: "Arts and Humanities",
    13: "Biochemistry",
    14: "Business",
    15: "Chemical Engineering",
    16: "Chemistry",
    17: "Computer Science",
    18: "Decision Sciences",
    19: "Earth and Planetary Sciences",
    20: "Economics",
    21: "Energy",
    22: "Engineering",
    23: "Environmental Science",
    24: "Immunology and Microbiology",
    25: "Materials Science",
    26: "Mathematics",
    27: "Medicine",
    28: "Neuroscience",
    29: "Nursing",
    30: "Pharmacology",
    31: "Physics and Astronomy",
    32: "Psychology",
    33: "Social Sciences",
    34: "Veterinary",
    35: "Dentistry",
    36: "Health Professions"})

#--------ORGANIZE FUND-------#

ogfdf = pd.read_csv("funderSubjArea.csv", index_col=0)
ogfdf = ogfdf.sort_values(by=['sum'], ascending=False)

ogfdf = ogfdf.rename(columns={'10': "Multidisciplinary",
    '11': "Agricultural and Biological Sciences",
    '12': "Arts and Humanities",
    '13': "Biochemistry",
    '14': "Business",
    '15': "Chemical Engineering",
    '16': "Chemistry",
    '17': "Computer Science",
    '18': "Decision Sciences",
    '19': "Earth and Planetary Sciences",
    '20': "Economics",
    '21': "Energy",
    '22': "Engineering",
    '23': "Environmental Science",
    '24': "Immunology and Microbiology",
    '25': "Materials Science",
    '26': "Mathematics",
    '27': "Medicine",
    '28': "Neuroscience",
    '29': "Nursing",
    '30': "Pharmacology",
    '31': "Physics and Astronomy",
    '32': "Psychology",
    '33': "Social Sciences",
    '34': "Veterinary",
    '35': "Dentistry",
    '36': "Health Professions"})

ogfdf2 = ogfdf.drop(columns=['sum'])

ogfdf2_tran = ogfdf2.T

ogfdf = ogfdf[['sum']].sort_values(by='sum', ascending=False).head(15)
ogfdf = ogfdf.reset_index()

#----------DISPLAY-----------#



col1, space, col2 = st.columns([8, 1, 12])

with col1:
    st.header("Top 10 Most Popular Keywords or Subject Area")
    option = st.selectbox("Choose data type", ("Keywords", "Subject Area"))

    if option == "Keywords":
        hd1, hd2 = st.columns([3, 2])
        with hd1:
            st.subheader("DataFrame")
        with hd2:
            opt = st.selectbox("Choose mode", ("Non Accumulative", "Accumulative"))
        if opt == "Non Accumulative":
            st.dataframe(df, use_container_width=True)
            st.subheader("Line Graph")
            st.line_chart(df_transposed)
        elif opt == "Accumulative":
            st.dataframe(df2, use_container_width=True)
            st.subheader("Line Graph")
            st.line_chart(df2_transposed)

    elif option == "Subject Area":
        hd1, hd2 = st.columns([3, 2])
        with hd1:
            st.subheader("DataFrame")
        with hd2:
            opt = st.selectbox("Choose mode", ("Non Accumulative", "Accumulative"))
        if opt == "Non Accumulative":
            st.dataframe(df3, use_container_width=True)
            st.subheader("Line Graph")
            st.line_chart(df3_transposed)
        elif opt == "Accumulative":
            st.dataframe(df4, use_container_width=True)
            st.subheader("Line Graph")
            st.line_chart(df4_transposed)

    st.divider()
    
    st.header("Funding Percentage (AI Prediction)")
    st.write("Percentage for papers with the following variables to be funded")

    co1, co2, co3, co4, co5 = st.columns([2,2,2,2,5])

    with co1:
        author_count = st.number_input("Author Count", min_value=0, step=1)

    with co2:
        ref_count = st.number_input("Ref Count", min_value=0, step=1)

    with co3:
        cited_by_count = st.number_input("Cited By Count", min_value=0, step=1)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    with co4:
        m = st.selectbox("Month", months)
        if m == 'Jan' or m ==  'Feb' or m == 'Mar':
            quantile = 1
        elif m == 'Apr' or m ==  'May' or m == 'Jun':
            quantile = 2
        elif m == 'Jul' or m ==  'Aug' or m == 'Sep':
            quantile = 3
        elif m == 'Oct' or m ==  'Nov' or m == 'Dec':
            quantile = 4

    subject_areas = [
        'Multidisciplinary', 'Agricultural and Biological Sciences', 'Arts and Humanities',
        'Biochemistry', 'Business', 'Chemical Engineering', 'Chemistry', 'Computer Science',
        'Decision Sciences', 'Earth and Planetary Sciences', 'Economics', 'Energy',
        'Engineering', 'Environmental Science', 'Immunology and Microbiology',
        'Materials Science', 'Mathematics', 'Medicine', 'Neuroscience', 'Nursing',
        'Pharmacology', 'Physics and Astronomy', 'Psychology', 'Social Sciences',
        'Veterinary', 'Dentistry', 'Health Professions'
    ]
    with co5:
        subject_area = st.multiselect("Subject Area", subject_areas)

    #---AI START---#

    subject_area_mapping = {
    '10': "Multidisciplinary",
    '11': "Agricultural and Biological Sciences",
    '12': "Arts and Humanities",
    '13': "Biochemistry",
    '14': "Business",
    '15': "Chemical Engineering",
    '16': "Chemistry",
    '17': "Computer Science",
    '18': "Decision Sciences",
    '19': "Earth and Planetary Sciences",
    '20': "Economics",
    '21': "Energy",
    '22': "Engineering",
    '23': "Environmental Science",
    '24': "Immunology and Microbiology",
    '25': "Materials Science",
    '26': "Mathematics",
    '27': "Medicine",
    '28': "Neuroscience",
    '29': "Nursing",
    '30': "Pharmacology",
    '31': "Physics and Astronomy",
    '32': "Psychology",
    '33': "Social Sciences",
    '34': "Veterinary",
    '35': "Dentistry",
    '36': "Health Professions"
    }

    reverse_mapping = {v: k for k, v in subject_area_mapping.items()}

    data = {
        "authorCount": author_count,
        "refCount": ref_count,
        "citedByCount": cited_by_count,
        "quatile": quantile,
    }

    for code in subject_area_mapping.keys():
        data[code] = 0

    for area in subject_area:
        if area in reverse_mapping:
            data[reverse_mapping[area]] = 1

    df11 = pd.DataFrame([data])
    
    with open('fundingMLP.model', 'rb') as file:
        model1 = pickle.load(file)

    pre = model1.predict_proba(df11[:1])

    yes_probability = pre[0][1]

    st.header(f"Fund Percentage: {round(yes_probability * 100, 2)}% " )

    #---AI END---#

    st.divider()

    st.header("Funded Papers By Different Affiliations")
    st.bar_chart(ogfdf.set_index('index')['sum'])
    
    st.divider()
    
    st.header("Funded Subject Areas by Chulalongkorn University")
    st.bar_chart(ogfdf2_tran['Chulalongkorn University'])

with col2:
    st.title("Network Graph of Top Keywords")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()


    st.title("Network Graph of Closest Keywords of an Input Keyword")
   
    query_keyword = st.text_input("Enter keyword...", value="SARS-CoV-2", placeholder="e.g. SARS-CoV-2")
    top_n = st.number_input("Top N related tags", min_value=0, value=10, placeholder=10, step=1, ) 
    
    tag_with_query = Counter({ k:v for k, v in co_occurrence.items() if query_keyword in k})
    if tag_with_query == Counter():
        st.text("Keyword not found")
    else:
        
    #----------NETWORK GRAPH OF CLOSEST KEYWORDS part 2/2-----------#
        top_tags = { e[0]:e[1] for e in tag_with_query.most_common(top_n) }

        edges = []
        for pair, weight in top_tags.items():
            edges.append((list(pair)[0], list(pair)[1], weight))
        edges_df = pd.DataFrame(edges, columns=['node1', 'node2', 'weight'])

        edges_df.sort_values(by='weight', ascending=False, inplace=True)
        edges_df = edges_df.head(30)

        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['node1'], row['node2'], weight=row['weight'])

        pos = nx.spring_layout(G, k=2, iterations=50)

        node_sizes = [np.log(top_tags[frozenset({query_keyword, node})] + 1) * 4 for node in G.nodes() if query_keyword != node]

        node_x, node_y, labels = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            labels.append(node)

        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=max(weight / 10, 1), color='white'),
                    mode='lines'
                )
            )

        node_colors = [random_rgb() for _ in G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, opacity=1),
            text=list(G.nodes()),
            textposition="top center",
            textfont=dict(color="#6acdff"),
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=0),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            dragmode='pan'
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    sb1, sb2 = st.columns([3, 2])
    with sb1:
        st.header("Unique Data Types per Subject Area")
    with sb2:
        op2 = st.selectbox("Choose data type", ("Paper Count", "Keywords", "Authors", "Cited by", "Reference Count", "Funding"))

    if op2 == "Paper Count":
        st.bar_chart(subjdf['paperCount'], height=500)
    elif op2 == "Keywords":
        st.bar_chart(subjdf['keywords'], height=500)
    elif op2 == "Authors":
        st.bar_chart(subjdf['authors'], height=500)
    elif op2 == "Cited by":
        st.bar_chart(subjdf['citedBy'], height=500)
    elif op2 == "Reference Count":
        st.bar_chart(subjdf['refCount'], height=500)
    elif op2 == "Funding":
        st.bar_chart(subjdf['funding'], height=500)


