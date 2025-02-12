- [Federated Neuro-Symbolic Learning](https://openreview.net/pdf?id=EQXZqBXeW9) (ICML 2024):  
    
    **TASK:** Federated ILP  
    
    **TL;DR:** This technique learns rules in a federated manner by passing the posterior distribution of rules from client to server and optimizing a local loss with a global regularization term.  


- [Neuro-Symbolic Temporal Point Processes](https://arxiv.org/pdf/2406.03914) (ICML 2024):

  **TASK:** Event Prediction  

  **TL;DR:** The optimization process performs rule discovery until all cases in the dataset are covered; whenever a rule covering a fixed amount of examples is discovered, those samples are removed from the dataset.  


- [NeSyFOLD: A Framework for Interpretable Image Classification](https://ojs.aaai.org/index.php/AAAI/article/view/28235) (AAAI 2024):

  **TASK:** Image Classification  

  **TL;DR:** The authors used an ILP method to transition from CNN activation maps to a labeled rule set used for classification.  


- [Confidence is not Timeless: Modeling Temporal Validity for Rule-based Temporal Knowledge Graph Forecasting](https://aclanthology.org/2024.acl-long.580/) (ACL 2024):
    
    **TASK:** temporal knowledge graph forecasting.
    
    **TL;DR:** Here the extrapolation is performed by means of rules rather than using a black box. Each rule is associated with a learnable confidence score which is learnt jointly with a learnable decay coefficient, following the assumption that the validity of rules decays over time.

    **Excluded** since this is a very specific problem and the paper mainly investigates the decay of rule confidence over time. Furthermore, mixed results are presented.


- [From Statistical Relational to Neuro-Symbolic Artificial Intelligence](https://arxiv.org/pdf/2003.08316) (AAAI 2024/[IJCAI](https://www.ijcai.org/proceedings/2020/0688.pdf) 2020):
    
    **TL;DR:** NeSy survery, even though it has been published in 2024 the covered (8) axes regard implementative details of the purest NeSy methods, all of the presented approaches either use probabilistic logic or some form of formal language (FOL) to solve problems. Interesting future directions are given as well. Tasks are not covered and there’s no quantitative comparison of performances, which is where our margin lies. Furthermore, the analyzed techniques are getting old (2017-2019).


- [QA-NatVer: Question Answering for Natural Logic-based Fact Verification](https://arxiv.org/pdf/2310.14198) (EMNLP 2023):

  **TASK:** Textual Fact Verification  

  **TL;DR:** Combines neural span alignment and NLI operator assignment with a symbolic automaton for veracity prediction and justification.  

- [NeuSTIP: A Neuro-Symbolic Model for Link and Time Prediction in Temporal Knowledge Graphs](https://aclanthology.org/2023.emnlp-main.274.pdf) (EMNLP 2023)
     
     **TASK:** temporal knowledge graph reasoning

     **TL;DR:** The task can be divided in two subtasks: link prediction, where the tail entity of a temporal relation is predicted, and time interval prediction, where the interval associated to each relation is predicted. The authors proposed a model able to learn compact and interpretable FOL rules with the addition of Allen algebra predicates, enfocing constraints on both time intervals and tail entities. 

     **Excluded** as other methods akin to this one, but more recent, have been found and supported the claims of our survey. Specifically, given that [TIELP](https://ojs.aaai.org/index.php/AAAI/article/view/29544) is a black-box that performs better on this task, we decided that it was useless to include this additional example. 


- [Learning Neuro-Symbolic World Models with Conversational Proprioception](https://aclanthology.org/2023.acl-short.57/) (ACL 2023):

     **TASK:** Text-based Games  

     **TL;DR:** Uses ILP and Logical Neural Networks for planning in text-based games.  

     **Excluded** due to unclear applicability.  


-[LogicDP: Creating Labels for Graph Data via Inductive Logic Programming](https://openreview.net/pdf?id=2b2s9vd7wYv) (ICLR 2023)

     **TASK:** Knowledge Graph Completion

     **TL;DR:** Uses ILP to refine rules used to create labels for incomplete graphs.

     **Excluded** as the task is evaluated on graph reasoning benchmarks that have been treated by more promising techniques in a clearer fashion.


- [Globally-Consistent Rule-Based Summary-Explanations for Machine Learning Models: Application to Credit-Risk Evaluation](https://jmlr.org/papers/volume24/21-0488/21-0488.pdf) (J. Mach. Learn. Res.,2023)

     **TASK:** Credit-Risk Evaluation

     **TL;DR:** This is a paper on local explanations of model predictions.

     **Excluded** as this is a method for post-hoc explainability.


- [Sequential Recommendation with Probabilistic Logical Reasoning](https://www.ijcai.org/proceedings/2023/0270.pdf) (IJCAI 2023)

     **TASK:** Sequential Recommendation

     **TL;DR:** This paper proposes to decouple feature embeddings and logic embeddings and to relax logic operations to the embedding space.

     **Excluded** as it is not clear how to pass from embedding to hard rules, making this model opaque.



- [Neuron Dependency Graphs: A Causal Abstraction of Neural Networks](https://proceedings.mlr.press/v162/hu22b/hu22b.pdf) (ICML 2022)

     **TASK** Image Classification, Sentiment Analysis and Natural Language Inference

     **TL;DR:** this paper extracts dependency rules from neurons, correlating them.

     **Excluded** as this paper is about post-hoc explainability of black-boxes more than creating a NeSy solution.


- [Neuro-symbolic Natural Logic with Introspective Revision for Natural Language Inference](https://dblp.org/rec/journals/tacl/FengYZG22.html) (TACL 2022):
  
     **TASK:** Natural Language Inference

     **TL;DR:** The authors used GPT-2 for operator assignment, reinforced learning for composition, and introspective revision with WordNet knowledge.
     Outperforms baselines except on SNLI, where BERT’s bi-directional attention dominates. Sensitive to noise from adverbs or prepositional phrases.


- [Neuro-Symbolic Hierarchical Rule Induction](https://proceedings.mlr.press/v162/glanois22a/glanois22a.pdf) (ICML 2022):

     **TASK:** Inductive Logic Programming

     **TL;DR:** A differentiable ILP model using hierarchical meta-rules to learn interpretable first-order rules.
     Combines supervised and RL training, with interpretability regularization.

     **Excluded** as the architecture is overly complicated and the benchmarks are quite outdated and not clear.


- [Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/20795) (AAAI 2022):

     **TASK:** KG Completion

     **TL;DR:** In this paper rules are learned according to different t-norms, in such a way that it is scalable enough to test it over WN18RR and FB15K.

     **Excluded** for the sake of brevity of the discourse as this technique has been outperformed
     by [Gao et al.](https://www.sciencedirect.com/science/article/pii/S0004370224000444?ref=pdf_download&fr=RR-2&rr=906106645d78a25e)


- [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://proceedings.mlr.press/v162/alon22a/alon22a.pdf) (ICML 2022):

     **TASK:** Retrieval Augmented Language Modelling

     **TL;DR:** The symbolic part consists of a probabilistic automaton obtained through means of clustering similar
     elements represented by triples (key, value, pointer) where the key is the LM hidden state, the value is
     the corresponding token and the pointer points to the next token in the datastore.

     **Excluded** due to the obsolescence of clustering-based methods for language models due to the
     high popularity of modern LLMs which use attention-based encoder-decoder architectures.


- [Search Space Expansion for Efficient Incremental Inductive Logic Programming from Streamed Data](https://www.ijcai.org/proceedings/2022/0374.pdf) (IJCAI 2022)

     **TASK:** Inductive Logic Programming

     **TL;DR:** This paper presents a method to perform learning from answer sets on data streams.

     **Excluded** as evaluation is carried out on ILP-specific benchmarks only, with performance evaluation relative to runtime efficiency. 


- [Lifting Symmetry Breaking Constraints with Inductive Logic Programming](https://www.ijcai.org/proceedings/2021/0284.pdf) (IJCAI 2021)

     **TASK** Combinatorial Search

     **TL;DR:** Instead of computing symmetry-breaking constraints for each specific problem instance, this approach learns general first-order constraints from small instances that can be reused across different problems.

     **Excluded** as the contribution lies in using a rule-mining technique (ILP) to lift symmetry-breaking constraints in Combinatorial Search. More specifically, the rule-mining proxy task is used to boost the performance of a solver, making it opaque, while the scope of this survey is to display methods that pass from black-box to interpretable using a proxy task.  


- [Differentiable Inductive Logic Programming for Structured Examples](https://arxiv.org/pdf/2103.01719) (AAAI 2021):

     **TASK:** ILP over structured samples

     **TL;DR:** This method uses adaptive clause search, an enumeration algorithm for ground atoms, and a soft logic program composition method to learn rules from trees or sequences.

     **Excluded** for a reason similar to the above one: the objective is to learn rules.


- [Scalable Rule-Based Representation Learning for Interpretable Classification](https://proceedings.neurips.cc/paper/2021/hash/ffbd6cbb019a1413183c8d08f2929307-Abstract.html) (NeurIPS 2021):

     **TASK:** Rule Mining for Classification

     **TL;DR:** The authors here provide mechanism to learn rules which is called gradient grafting.
     It consists in projecting discrete rules in a continuous space to allow backpropagation and adjust parameters. The obtained rules are the model used for prediction.

     **Excluded** as performance evaluation is performed against ancient models such as SVMs over a plethora of dataset that don't have much in common.


- [Dynamic Neuro-Symbolic Knowledge Graph Construction for Zero-shot Commonsense Question Answering](https://arxiv.org/pdf/1911.03876) (AAAI 2021)

     **TASK:** Commonsense reasoning (Multiple-choice QA)

     **TL;DR:** This method creates a graph of possible partial answers connected to those of the multiple-choice quiz using an LLM.

     **Excluded** because nodes of this graph are created using an LLM, plus we don't believe that mining a graph is the same as mining rules given the complex nature of the first data structure (with an increasing number of connections comes reduced interpretability).


- [Turning 30: New Ideas in Inductive Logic Programming](https://www.ijcai.org/proceedings/2020/0673.pdf) (IJCAI 2020):

     **TL;DR:** Old survey on ILP techniques.

     **Excluded** since it's a survey.

- [FastLAS: Scalable Inductive Logic Programming Incorporating Domain-Specific Optimisation Criteria](https://ojs.aaai.org/index.php/AAAI/article/view/5678) (AAAI 2020):

     **TASK** Inductive Logic Programming

     **TL;DR:**  the main contribution of this paper is the introduction of scoring function for rules to improve optimization by including domain knowledge.

     **Excluded** as evaluation is limited to ILP-related benchmarks.


