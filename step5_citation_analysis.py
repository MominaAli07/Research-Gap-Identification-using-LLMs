# step5_citation_analysis.py
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import config

def build_citation_network(papers_df):
    print("Building a citation network from Semantic Scholar references...")
    G = nx.DiGraph()
    paper_ids = set(papers_df['id'])
    for index, row in papers_df.iterrows():
        node_id = row['id']
        G.add_node(node_id,
                   year=row.get('year'),
                   title=row.get('title'),
                   topic_id=row.get('topic_id', -1))
    for index, row in papers_df.iterrows():
        paper_id = row['id']
        references = row.get('references', "[]")
        if isinstance(references, str):
            try:
                references = eval(references)
            except:
                references = []
        for ref in references:
            ref_id = f"s2:{ref}"
            if ref_id in paper_ids:
                G.add_edge(paper_id, ref_id)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def analyze_citation_network(G, papers_df):
    print("Analyzing citation network...")
    analysis_results = {}
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    low_in_degree_nodes = [node for node, degree in in_degrees.items() if degree < 1]
    analysis_results['low_in_degree_count'] = len(low_in_degree_nodes)
    low_out_degree_nodes = [node for node, degree in out_degrees.items() if degree < 1]
    analysis_results['low_out_degree_count'] = len(low_out_degree_nodes)
    isolated_topics = set()
    node_topic_map = pd.Series(papers_df.topic_id.values, index=papers_df.id).to_dict()
    for node in low_in_degree_nodes + low_out_degree_nodes:
         topic = node_topic_map.get(node)
         if topic is not None and topic != -1:
             isolated_topics.add(topic)
    analysis_results['potentially_isolated_topics'] = list(isolated_topics)
    print(f"Potentially isolated topics: {list(isolated_topics)}")
    topic_years = {}
    for node, data in G.nodes(data=True):
        topic = data.get('topic_id')
        year = data.get('year')
        if topic is not None and topic != -1 and year is not None:
            topic_years.setdefault(topic, []).append(int(year))
    avg_topic_year = {topic: sum(years)/len(years) for topic, years in topic_years.items() if years}
    analysis_results['average_topic_year'] = avg_topic_year
    sorted_avg_year = sorted(avg_topic_year.items(), key=lambda item: item[1], reverse=True)
    print(f"Most recent topics (avg year): {sorted_avg_year[:5]}")
    print(f"Oldest topics (avg year): {sorted_avg_year[-5:]}")
    graph_path = os.path.join(config.RESULTS_DIR, "citation_network.gexf")
    nx.write_gexf(G, graph_path)
    print(f"Citation network graph saved to {graph_path}")
    try:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1, iterations=20)
        nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=False)
        plt.title("Citation Network (Simplified)")
        plt.axis('off')
        plot_path = os.path.join(config.RESULTS_DIR, "citation_network_plot.png")
        plt.savefig(plot_path)
        print(f"Network plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Network plot error: {e}")
    return analysis_results

if __name__ == "__main__":
    print("\nStep 5: Starting Citation Analysis...")
    topics_path = os.path.join(config.PROCESSED_DIR, "papers_with_topics.csv")
    if os.path.exists(topics_path):
        papers_df = pd.read_csv(topics_path)
        G = build_citation_network(papers_df)
        network_analysis = analyze_citation_network(G, papers_df)
        print("\nCitation Network Analysis Summary:")
        print(network_analysis)
        print("\nStep 5: Citation Analysis Finished.")
    else:
        print(f"Error: Papers with topics file not found at {topics_path}. Run previous steps first.")
