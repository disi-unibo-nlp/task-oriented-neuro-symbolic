# A Window to the World of Neuro-Symbolic Artificial Intelligence: A Task-Directed Survey in the Black-Box Era
## Submitted at IJCAI 25 - Survey Track
This is a repository containing notes over the papers collected to write a survey on the world of Neuro-Symbolic (NeSy) approaches.

```sparql
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?paper ?title ?conference ?year
WHERE {
  FILTER (year(?year) <= 2024 && year(?year) >= 2017) {
    FILTER (REGEX(?conference, "^J. Mach. Learn. Res.$|^Int. J. Comput. Vis.$|^Trans. Assoc. Comput. Linguistics$|^ijcai$|^aaai$|^neurips$|^icml$|^iclr$|^ACL \\\\(\\d+|^emnlp$|^naacl-hlt$|^cvpr$|^iccv$|^eccv \\\\(\\d+","i")) {
      SELECT ?paper ?title ?conference ?year where {
        ?paper rdf:type dblp:Publication .
        ?paper dblp:title ?title .
        ?paper dblp:publishedIn ?conference .
        ?paper dblp:yearOfPublication ?year .
        FILTER (REGEX(?title, "inductive-logic programming|inductive logic programming|nesy|neurosymbolic|neuro-symbolic|neuro symbolic|rule-based|rule based|concept learning|concept-learning|logic based|logic-based|soft logic|soft-logic|fuzzy-logic|fuzzy logic|probabilistic-logic|probabilistic-reasoning|probabilistic logic|probabilistic reasoning?", "i")).
      }
    }
  }
}
ORDER BY DESC(?year)
```

This query should be executed in the SPARQL endpoint of DBLP, or it can be run at the following link:

```
https://sparql.dblp.org/m2mK1m
```

In order to explore further the intersection between Knowledge Graph Reasoning and NeSy, we run the following query:
```sparql
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?paper ?title ?conference ?year
WHERE {
FILTER (year(?year) <= 2024 && year(?year) >= 2017) {
FILTER (REGEX(?conference, "^J. Mach. Learn. Res.$|^Int. J. Comput. Vis.$|^Trans. Assoc. Comput. Linguistics$|^ijcai$|^aaai$|^neurips$|^icml$|^iclr$|^ACL \\(\\d+|^emnlp$|^naacl-hlt$|^cvpr$|^iccv$|^eccv \\(\\d+","i")) {
SELECT ?paper ?title ?conference ?year where {
  ?paper rdf:type dblp:Publication .
  ?paper dblp:title ?title .
  ?paper dblp:publishedIn ?conference .
  ?paper dblp:yearOfPublication ?year .
  FILTER (REGEX(?title, "neuro", "i") && REGEX(?title, "graph", "i")).
}}}}
ORDER BY DESC(?year)
```

Like in the above case, it can be executed at the following link:

```
https://sparql.dblp.org/mU4l2C
```


Each folder contains the complete list of papers in the corresponding category, along with a link to the paper’s page and a brief description of its content. 

Additionally, we provide a short explanation for exclusion from the survey.
