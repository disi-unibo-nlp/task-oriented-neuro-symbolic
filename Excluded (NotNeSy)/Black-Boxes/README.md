- [NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks.](https://arxiv.org/pdf/2110.14053) (ICLR 2024):

    **TL;DR:** Black-box method powered by GNNs to improve initial assignments for SAT problems.


- [ContPhy: Continuum Physical Concept Learning and Reasoning from Videos](https://arxiv.org/pdf/2402.06119) (ICML 2024):
    
    **TL;DR:** Benchmark for physics common sense reasoning over videos.
    
- [CLiC: Concept Learning in Context](https://openaccess.thecvf.com/content/CVPR2024/papers/Safaee_CLiC_Concept_Learning_in_Context_CVPR_2024_paper.pdf) (CVPR 2024):
    
    **TL;DR:** Here concepts are local visual patterns that will be learned in the form of a visual token that can be implanted in images where a pattern is missing, allowing for the migration of characteristics of certain objects (like ornaments) to other objects that didn’t present them in the original picture.

    
- [TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning](https://arxiv.org/pdf/2402.19467) (EMNLP 2024):
    
    **RL;DR:** Explainability is provided through means of entailment trees that are built recursively using LLMs. This method is prone to hallucinations and it’s hard to classify as a NeSy method (it’s a bit borderline).

    
- [Towards Compositionality in Concept Learning](https://arxiv.org/abs/2406.18534) (ICML 2024):
    
    **TL;DR:** This is an improvement in the family of models that derive concepts from black-box models, since the extracted concepts can be composed when they belong to different attributes (blue and cube are composable, while red and green are not).
    
- [What Are the Odds? Language Models Are Capable of Probabilistic Reasoning](https://arxiv.org/pdf/2406.12830) (EMNLP 2024):
    
    **TL;DR:** This paper illustrates a benchmark for LLMs to evaluate their probabilistic reasoning capabilities.
    Here probabilistic reasoning is intended as extrapolating statistics or mathematical aspects of probability distributions, something that could be performed in by a NeSy model in principle, though it’s pretty distant from probabilistic networks such as markov chains that could be used as a structure upon which a NeSy model could be built.

    
- [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://arxiv.org/pdf/2402.17709) (ICML 2024):
    
    **TL;DR:** First it is demonstrated that Transformers use case-based reasoning, then a fine-tuning approach is used to teach transformers to follow mathematical rules to solve mathematical problems. One may argue that this method is not purely NeSy as rules are not embedded in the system, instead they are generally followed by the model which could hallucinate in the process (there’s no grounding/substitution). 

    
- [Language-Informed Visual Concept Learning](https://openreview.net/pdf?id=juuyW8B8ig) (ICLR 2024):
    
    **TL;DR:** This is not a neuro-symbolic method as concepts are visual embeddings that are anchored to textual embeddings to disentangle visual features. The employed technique is distillation of Visual Language Model.

    
- [NELLIE: A Neuro-Symbolic Inference Engine for Grounded, Compositional, and Explainable Reasoning](https://www.ijcai.org/proceedings/2024/0399.pdf) (IJCAI 2024):
    
    **TL;DR:** Explainability as entailment tree generation through means of a prolog procedure.

    
- [ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models](https://ojs.aaai.org/index.php/AAAI/article/view/29371) (AAAI 2024):
    
    **TL;DR:** A dataset to evaluate concept learning abilities of TTI models, an evaluation metric is proposed as well. Interestingly, “The neuro-symbolic concept learner” is cited.

    
- [A Self-explaining Neural Architecture for Generalizable Concept Learning](https://arxiv.org/pdf/2405.00349) (IJCAI 2024):
    
    **TL;DR:** Here the authors propose another architecture to learn abstract entities from images which align with human understanding.


- [AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neuron Activity](https://proceedings.neurips.cc/paper_files/paper/2023/file/1c70ba3591d0694a535089e1c25888d7-Paper-Conference.pdf) (NeurIPS 2023):

    **TL;DR:** This is actually a paper on the brain.


- [NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics](https://arxiv.org/pdf/2306.06202) (NeurIPS 2023):

    **TL;DR:** This is actually a paper on the brain.


- [Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis](https://arxiv.org/pdf/2208.10609) (AAAI 2023):

    **TL;DR:** The tackled task is similar to what is carried out by CAM methods for CNNs and Visual Transformers, mapping classes or concepts to specific neuron activations of the graph. Interestingly, some activations can be expressed in terms of FOL rules, however the paper does not employ such rules for the sake of a prediction, thus this paper has no direct connection with the methods discussed in the survey.

    
- [Decompose Novel into Known: Part Concept Learning For 3D Novel Class Discovery](https://proceedings.neurips.cc/paper_files/paper/2023/file/aa31eee8f2351176ddd4d14646d4a950-Paper-Conference.pdf) (NeurIPS 2023):
    
    **TL;DR:** The model takes a point cloud as input, it recognize parts with a dedicated module and stores it in a part concept bank, while learning a part relation using a part relation encoder. Both part embeddings and part relation embeddings are encoded and fed to a classifier to perform object recognition using such parts.

    
- [Weakly Supervised Explainable Phrasal Reasoning with Neural Fuzzy Logic](https://arxiv.org/pdf/2109.08927) (ICLR 2023):
    
    **TL;DR:** This method caches facts in a dedicated factual memory and then computes attention over it.

    
- [ThinkSum: Probabilistic reasoning over sets using large language models](https://arxiv.org/pdf/2210.01293) (ACL 2023):
    
    **TL;DR:** Non-NeSy approach to QA where sets are enumerated to answer the question and elements of the set are aggregated through various means. The result of the aggregation will guide the answer.


- [Neuro-Symbolic Procedural Planning with Commonsense Prompting](https://openreview.net/pdf?id=iOc57X9KM54) (ICLR 2023):

    **TL;DR:** Knowledge infusion from KG in LLM prompts to solve procedural planning problems.


- [Beyond Object Recognition: A New Benchmark towards Object Concept Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Beyond_Object_Recognition_A_New_Benchmark_towards_Object_Concept_Learning_ICCV_2023_paper.pdf) (ICCV 2023):

    **TL;DR:** It's an object recognition benchmark where the aim is to learn a causal graph.

    
- [3D Concept Learning and Reasoning from Multi-View Images](https://openaccess.thecvf.com/content/CVPR2023/papers/Hong_3D_Concept_Learning_and_Reasoning_From_Multi-View_Images_CVPR_2023_paper.pdf) (CVPR 2023):
    
    **TL;DR:** semantic segmentation with QA over environment pictures and reconstructed point clouds of such environments.

    
- [Robust Neuro-Symbolic Goal and Plan Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/26408) (AAAI 2023):
    
    **TL;DR:** Here the task is goal and plan recognition, the proposed approach is Predictive Plan Recognition, which aim at reconstructing the goal as a series of
    intermediate steps. The methodology looks similar to Q-learning though it is not a RL pipeline.

    
- [Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation](https://ojs.aaai.org/index.php/AAAI/article/view/25243) (AAAI 2023):
    
    **TL;DR:** Here concepts are intended as atomic actions like “Turn left” or “go back”, used to align VLM with a navigation dataset to achieve good performances on REVERIE.
    
- [House of Cans: Covert Transmission of Internal Datasets via Capacity-Aware Neuron Steganography](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d65080f3be61f4dcc5ca4c293308104-Paper-Conference.pdf) (NeurIPS 2022):

    **TL;DR:** This paper describes a new method called "Cans" (capacity-aware neuron steganography) that allows people to secretly hide large amounts of private machine learning data inside a neural network that's planned to be published.


- [ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time](https://arxiv.org/pdf/2206.15049) (NeurIPS 2022):

    **TL;DR:** Autoencoders for hierarchical concept learning.


- [PhysGNN: A Physics-Driven Graph Neural Network Based Model for Predicting Soft Tissue Deformation in Image-Guided Neurosurgery](https://arxiv.org/pdf/2109.04352) (NeurIPS 2022):

    **TL;DR:** This is actually a paper on the brain.


- [Drawing out of Distribution with Neuro-Symbolic Generative Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6248a3b8279a39b3668a8a7c0e29164d-Abstract-Conference.html) (NeurIPS 2022):
    
    **TL;DR:** The presented model is a neuro-symbolic generative model of stroke-based drawing that can learn general-purpose representations. As this seems to be a network to recognize and reproduce strokes.

    
- [NS3: Neuro-symbolic Semantic Code Search](https://proceedings.neurips.cc/paper_files/paper/2022/file/43f5f6c5cb333115914c8448b8506411-Paper-Conference.pdf) (NeurIPS 2022):
    
    **TL;DR:** The task consists of retrieving a code snippet given a description of its functionality. Code is not generated but retrieved and the NeSy part lies probably in the fact that alignment scores between entities are computed.

    
- [LIMREF: Local Interpretable Model Agnostic Rule-Based Explanations for Forecasting, with an Application to Electricity Smart Meter Data](https://ojs.aaai.org/index.php/AAAI/article/view/21469) (AAAI 2022):
    
    **TL;DR:** Here rule mining is employed for the sake of post-hoc explainability to motivate predictions of electricity demands.


- [On the Cryptographic Hardness of Learning Single Periodic Neurons](https://arxiv.org/pdf/2106.10744) (NeurIPS 2021):

    **TL;DR:** This paper explores the computational difficulty of training a specific type of neural network, specifically a single periodic neuron, when working with noisy data. Single periodic neurons are a simple type of artificial neuron that applies a periodic activation function (like sine or cosine) to its input.

    
- [Disentangling 3D Prototypical Networks for Few-Shot Concept Learning](https://arxiv.org/pdf/2011.03367) (ICLR 2021):
    
    **TL;DR:** Meta-concepts (Shape, Size etc.) of each object in a 3D environment are disentangled to improve prediction.
    

- [Learning Task-General Representations with Generative Neuro-Symbolic Modeling](https://arxiv.org/pdf/2006.14448) (ICLR 2021):
    
    **TL;DR:** Black-box model to predict stroke prototypes.  


- [Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding](https://www.ijcai.org/proceedings/2021/0278.pdf) (IJCAI 2021):

    **TL;DR:** Black-box model for graph alignment.
    

- [Multi-graph Fusion for Functional Neuroimaging Biomarker Detection](https://www.ijcai.org/proceedings/2020/0081.pdf) (IJCAI 2020):

    **TL;DR:** actually a paper on the brain.

- [Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning](https://proceedings.neurips.cc/paper_files/paper/2020/file/bf15e9bbff22c7719020f9df4badc20a-Paper.pdf) (NeurIPS 2020):
    
    **TL;DR:** Old concept learning benchmark for images.
    

- [Generative Continual Concept Learning](https://ojs.aaai.org/index.php/AAAI/article/view/6006) (AAAI 2020):
      
    **TL;DR:** The task of mitigating catastrophic forgetting is tackled by learning concept in a latent space. 
    Evaluation is performed on multiple tasks over the MNIST dataset.

    
- [Failure-Scenario Maker for Rule-Based Agent using Multi-agent Adversarial Reinforcement Learning and its Application to Autonomous Driving](https://arxiv.org/pdf/1903.10654) (IJCAI 2019):
    
    **TL;DR:** The task consists of learning failure scenarios, the method is an opaque RL pipeline.


- [Graph-Based Object Classification for Neuromorphic Vision Sensing](https://arxiv.org/pdf/1908.06648) (ICCV 2019):

    **TL;DR** Old method for object detection.

    
- [Interactive Language Acquisition with One-shot Visual Concept Learning through a Conversational Game](https://arxiv.org/pdf/1805.00462) (ACL 2018):
    
    **TL;DR:** Black-box model to perform language modelling.

    
- [Multimodal Visual Concept Learning With Weakly Supervised Techniques](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bouritsas_Multimodal_Visual_Concept_CVPR_2018_paper.pdf) (CVPR 2018):
    
    **TL;DR:** Integration of weak-supervision in the form of textual transcripts to carry out concept learning over videos.
    
- [Deep Multi-View Concept Learning](https://www.ijcai.org/Proceedings/2018/0402.pdf) (IJCAI 2018):
    
    **TL;DR:** Here the task is hierarchical visual concept learning and the model is a black-box.
    
- [Joint Concept Learning and Semantic Parsing from Natural Language Explanations](https://aclanthology.org/D17-1161.pdf) (EMNLP 2017):
    
    **TL;DR:** The task is concept learning from text.
    
- [Evaluation of Human-AI Teams for Learned and Rule-Based Agents in Hanabi](https://proceedings.neurips.cc/paper/2021/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) (NeurIPS 2021):
    
    **TL;DR:** Here the task is the evaluation of human+AI teams in the card game Hanabi
    This paper provides insights regarding the preference of rule-based AI teammates by human for the purpose of playing a competitive game. RL techniques from 2015 to 2021 are used for the sake of this evaluation.
