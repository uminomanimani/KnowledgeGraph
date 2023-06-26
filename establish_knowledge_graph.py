from py2neo import Graph
from extract_triplet import extract_triplet
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher#导入我们需要的头文件
 
graph = Graph('http://localhost:7474',auth=('neo4j','12345678'), name = 'neo4j')
graph.query('MATCH ()-[r]->() DELETE r')
graph.query('MATCH (n) DETACH DELETE n')

triplets = extract_triplet()
for triplet in triplets:
    head = triplet[0][0]
    head_type = triplet[0][1]

    tail = triplet[1][0]
    tail_type = triplet[1][1]

    relation = triplet[2]

    graph.query(f'MERGE (h:{head_type} {{name: \"{head}\"}})')
    graph.query(f'MERGE (t:{tail_type} {{name: \"{tail}\"}})')
    graph.query(f'MATCH (a:{head_type} {{name:\"{head}\"}}),(b:{tail_type} {{name:\"{tail}\"}}) MERGE (a)-[:{relation}]->(b)')



