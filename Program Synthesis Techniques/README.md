- [NESTER: An Adaptive Neurosymbolic Method for Causal Effect Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/29398) (AAAI 2024):
    
    **TASK:** Causal-Effect Estimation 
    
    **TL;DR:** The method leverages a domain-specific language (DSL) expressed via a context-free grammar and a heuristically guided A* algorithm to iteratively select the most appropriate grammar rule at each generation step. The DSL was specifically designed for causal effect estimation and includes constructs for incorporating neural network calls.
    

- [GENOME: Generative Neuro-Symbolic Visual Reasoning by Growing and Reusing Modules](https://arxiv.org/pdf/2311.04901) (ICLR 2024):
    
    **TASK:** Visual Reasoning
    
    **TL;DR:** The method generates programs that can be expanded or re-used and drive the decisions for Visual Reasoning tasks.

    
- [StackSight: Unveiling WebAssembly through Large Language Models and Neurosymbolic Chain-of-Thought Decompilation](https://arxiv.org/pdf/2406.04568) (ICML 2024):
    
    **TASK:** Semantic (Code) Parsing
    
    **TL;DR:** The proposed method decompiles WebAssembly into readable C++ code snippets by using static analysis to track virtual stack changes and applying chain-of-thought prompting to leverage LLMs’ reasoning capabilities.

    **Excluded** as this task has very limited applicability. 

    
- [Neurosymbolic Grounding for Compositional World Models](https://arxiv.org/pdf/2310.12690) (ICLR 2024):
    
    **Task:** Compositional World Modelling 

    **TL;DR:** This neurosymbolic framework for object-centric world modeling, designed to achieve compositional generalization i.e. performing well on unseen scenes created from known components. 

    **Excluded** as this task has very limited applicability. 

    
- [Large Language Models Are Neurosymbolic Reasoners](https://ojs.aaai.org/index.php/AAAI/article/view/29754) (AAAI 2024):
    
    **TASK:** Solve Text-based Games

    **TL;DR:** This method employs LAM-like calls to modules to solve text-based games.

    **Excluded** as the task seems to have very limited applicability and LAM have been developed as a different family of techniques during recent years.

    
- [Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal](https://arxiv.org/pdf/2302.01242) (ICML 2023):
    
    **TASK:** Visual Reasoning

    **TL;DR:** Here it is proposed a novel loss function for continual learning in neuro-symbolic (CL) models, designed to address both catastrophic forgetting and reasoning shortcuts.
    
- [LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers](https://arxiv.org/pdf/2310.15164) (EMNLP 2023):
    
    **TASK:** Solving Logical Problems
    
    **TL;DR:** This paper addressed the challenge of logical reasoning—deductively inferring the truth value of a conclusion from a set of premises—by framing it as a semantic parsing task that translates natural language inputs into first-order logic (FOL) clauses. These clauses are subsequently processed by the Prover9 solver following a decoding step, rendering the method non-differentiable after the parsing stage.
    
- [Image Manipulation via Multi-Hop Instructions - A New Dataset and Weakly-Supervised Neuro-Symbolic Approach](https://arxiv.org/pdf/2305.14410) (EMNLP 2023):
    
    **TASK:** Image Manipulation
    
    **TL;DR:** Image manipulation on the CLEVR dataset under a distant supervision setting. Users can provide an utterance describing the desired image manipulation, and the method generates the corresponding output. A DSL is involved, as this is a follow-up work of NS-CL.

    
- [Neuro-symbolic Learning Yielding Logical Constraints](https://proceedings.neurips.cc/paper_files/paper/2023/file/4459c3c143db74ee52afebdf56836375-Paper-Conference.pdf) (NeurIPS 2023):
    
    **TASK:** Logical Reasoning
    
    **TL;DR:** This method is composed by a neural network for perception and symbol grounding, and an SMT solver for solution computation.
    
    **Excluded** as this approach is evaluated on grid-based problems (Visual Sudoku and 10x10 grid path planning with obstacles).
    
- [Neuro-Symbolic Learning of Answer Set Programs from Raw Data](https://arxiv.org/pdf/2205.12735) (IJCAI 2023):
    
    **TASK:** Arithmetic over MNIST Digits
    
    **TL;DR:** This method consists of a Learning from Answer Set pipeline. We have an iterative loop that jointly learns the representation of the input data as well as the ASP (Answer Set Programming) program. The approach is quite promising as the learned knowledge is expressed as prolog rules.

    **Excluded** as benchmarking is limited to MNIST dataset.
    
- [NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations](https://openaccess.thecvf.com/content/CVPR2023/papers/Hsu_NS3D_Neuro-Symbolic_Grounding_of_3D_Objects_and_Relations_CVPR_2023_paper.pdf) (CVPR 2023):

    **TASK:** QA over 3D scenes (similar to the above task)
    
    **TL;DR:** The method comprises three fundamental modules: a semantic parser, which translates the input utterance into a DSL program, where each function keyword is manually mapped to a vector operation (e.g., vector aggregation for filtering objects of a specific class); an object encoder, which is a neural network that processes objects in the form of point clouds and computes relation tensors for building ternary relation encodings; and a program executor, which interprets the DSL program and applies the defined vector operations to the encoded objects and relations, ultimately yielding the target object.

    
- [Neurosymbolic Deep Generative Models for Sequence Data with Relational Constraints](https://proceedings.neurips.cc/paper_files/paper/2022/file/f13ceb1b94145aad0e54186373cc86d7-Paper-Conference.pdf) (NeurIPS 2022):
    
    **TASK:** Music and Poetry generation
    
    **TL;DR:** Here the tasks are tackled by two separated models: one that generates a program that represents relational constraints and the other that synthesizes data that respect those contraints.

    **Excluded** even though the tasks were interesting since the data was very old and the program synthesis method relies on prototypes, an obsolete methodology.

    
- [Programmatic Concept Learning for Human Motion Description and Synthesis](https://openaccess.thecvf.com/content/CVPR2022/papers/Kulal_Programmatic_Concept_Learning_for_Human_Motion_Description_and_Synthesis_CVPR_2022_paper.pdf) (CVPR 2022):
    
    **TASK:** video synthesis
    
    **TL;DR:** Here the authors tackle the problem of video synthesis by concept learning, where each concept is a piece of a program that will be used as guide to generate the video.

    **Excluded** as similar techniques are already treated for images and this porting to the video domain does not display significant improvements as the demonstrations do not include a video synthesized with multiple constraints in the prompt.
    
- [FALCON: Fast Visual Concept Learning by Integrating Images, Linguistic descriptions, and Conceptual Relations](https://arxiv.org/pdf/2203.16639) (ICLR 2022):
    
    **TASK:** Visual Question Answering
    
    **TL;DR:** This paper introduced a complex concept encoder to enhance the visual encoder, enabling meta-learning of concepts from multiple data streams, such as example images and contextual text. The method consists of two main stages: Pre-training where objects are encoded into a geometric space; Meta-training, where concepts are learned as hyperboxes, grouping related objects or objects with shared relations. This stage employs another GNN with the same structure as the one used for pre-training but trained with distinct parameters. The embeddings are refined using a loss function that evaluates the alignment of the prior concept embeddings with the learned hyperboxes.

    
- [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning over Text](https://ojs.aaai.org/index.php/AAAI/article/view/21374) (AAAI 2022):
    
    **TASK:** Numerical Question Answering
    
    **TL;DR:** In this paper the approach is quite straightforward: we have a query which is parsed to a dependency tree and from that a program is extracted to decide which operations and which arguments have to be selected. The selection of arguments and operators depends on the passage relative to the query and a discrete reasoner is used to get the answer. Overall this paper lies in the parse-then-solve family of solutions. 

    **Excluded** in favor of more recent solutions of the same family.
    
- [Weakly Supervised Knowledge Transfer with Probabilistic Logical Reasoning for Object Detection](https://arxiv.org/pdf/2303.05148) (ICLR 2023) (?):
    
    **TASK:** object detection
    
    **TL;DR:** Weakly supervised method for Object detection employing domain adaptation.

    **Excluded** as this methodology doesn't seem to use any DSL though the dataset is CLEVR, treated extensively for the NS-CL family of models. 
    
- [Detect, Understand, Act: A Neuro-Symbolic Hierarchical Reinforcement Learning Framework](https://link.springer.com/article/10.1007/s10994-022-06142-7) (IJCAI 2022):
    
    **TASK:** RL for reasoning in environments
    
    **TL;DR:** The authors here have built a RL-based framework to enable reasoning in environments. The NeSy part lies in the construction of ASP programs (Answer Set Programs are akin to prolog programs) and the use of a meta policy to select the next action given such programs.

    **Excluded** as this RL method does not employ Q-learning and the arenas are custom and only briefly described. More recent and interesting methodologies have been treated.

    
- [Improving Coherence and Consistency in Neural Sequence Models with Dual-System, Neuro-Symbolic Reasoning](https://proceedings.neurips.cc/paper/2021/file/d3e2e8f631bd9336ed25b8162aef8782-Paper.pdf) (NeurIPS 2021):
    
    **TASK:** Long Text Generation
    
    **TL;DR:** The objective here is post-hoc checking of the next sentence. To do this, a world model is extracted from the previously generated text and candidate next sentences are checked symbolically against the world model. Only consistent next sentences are chosen for the final generation.

    **Excluded** as LSTM is used for the sake of generating sequences of actions in a grid world game to demonstrate the generalization power of this approach, questioning the paper's coherence.

    
- [CURI: A Benchmark for Productive Concept Learning Under Uncertainty](https://proceedings.mlr.press/v139/vedantam21a/vedantam21a.pdf) (ICML 2021):
    
    **TASK:** Visual Reasoning Under Uncertainty
    
    **TL;DR:** The authors here extended the work of “The neuro-symbolic concept learning” to a framework with uncertainty.
    
    **Excluded** as the paper doesn’t seem to compare methods that solve the tasks displayed on CLEVR, leaving a huge evaluation gap in this work. Furthermore, the reviews of this paper to a submission to the ICLR conference give further reasons not to include this paper in the discourse (see this: https://openreview.net/forum?id=LuyryrCs6Ez). 
    

- [Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting With Their Explanations](https://openaccess.thecvf.com/content/CVPR2021/papers/Stammer_Right_for_the_Right_Concept_Revising_Neuro-Symbolic_Concepts_by_Interacting_CVPR_2021_paper.pdf) (CVPR 2021):
    
    **TASK:** Explanatory Interactive Learning
    
    **TL;DR:** Here the CLEVR dataset is extended for the task of using the NeSy rules as feedback to train the network in order to recognize concepts in a picture even in the presence of confounding factors. 

    **Excluded** as the analysis was carried out on synthetic datasets only, with an example on the MNIST dataset.

    
- [PLANS: Neuro-Symbolic Program Learning from Videos](https://proceedings.neurips.cc/paper_files/paper/2020/file/fe131d7f5a6b38b23cc967316c13dae2-Paper.pdf) (NeurIPS 2020):
    
    **TASK:** program synthesis from videos
    
    **TL;DR:** The method consists of a neural-based feature extractor needed to obtain high-level informations from videos and a rule-based system that synthesizes the program using a grammar. 

    **Excluded** as the benchmarks used for evaluation are related to games and mostly used for RL tasks, questioning the applicability of this approach to other forms of videos.

    
- [Learning Neurosymbolic Generative Models via Program Synthesis](https://proceedings.mlr.press/v97/young19a.html) (ICML 2019):
    
    **TASK:** Image Generation
    
    **TL;DR:** The authors here had the idea of using programs to represent patterns in the process of generating images (as windows repeat themselves with a fixed distance).

    **Excluded** as the realization of this approach is qualitatively poor.

    
- [The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://arxiv.org/pdf/1904.12584) (ICLR 2019):
    
    **TASK:** Concept Learning.
    
    **TL;DR:** a foundational model for concept learning on the synthetic dataset CLEVR. This paper is also referred to as NS-CL.
    
- [Visual Concept-Metaconcept Learning](https://proceedings.neurips.cc/paper/2019/file/98d8a23fd60826a2a474c5b4f5811707-Paper.pdf) (NeurIPS 2019):
    
    **TASK:** Learning Visual Metaconcepts
    
    **TL;DR:** Follow-up work w.r.t NS-CL where the focus is put on learning metaconcepts (e.g. sphere and cylinder are keywords for the shape meta-concept).

    **Excluded** as the meta-level reasoning seems to be more a property of the lanugage encoder than the learnt representation space.  
    
- [DeepProbLog: Neural Probabilistic Logic Programming](https://proceedings.neurips.cc/paper_files/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf) (NeurIPS 2018):
    
    **TASK:** symbolic and subsymbolic representation/inference, program induction, probabilistic programming and learning success probability of a query from examples.

    **TL;DR:** this foundational model has gain a lot of popularity in the NeSy community for extending ProLog with probabilistic reasoning, enabling the integration of deep learning with logical inference. DeepProbLog extends Prolog by allowing probabilistic facts and neural predicates, making it capable of handling uncertain knowledge while leveraging neural networks for perception tasks.
    
- [Neuro-Symbolic Visual Reasoning: Disentangling "Visual" from "Reasoning”](https://proceedings.mlr.press/v119/amizadeh20a.html) (ICML 2020):
    **TASK:** VQA
    
    **TL;DR:** In this paper images are parsed to FOL rules instead of scene graphs, thus enabling reasoning instead of purely informative features. Interestingly, the paper suggests that this capability of extracting FOL rules from images should be evaluated in leaderboards.
    
    **Excluded** as it’s a prototypical work that can’t be integrated with the discourse of NS-CL as it uses a different proxy task and a totally different dataset. Furthermore, it didn’t gain popularity.
    
- [Neurosymbolic Transformers for Multi-Agent Communication](https://proceedings.neurips.cc/paper_files/paper/2020/file/9d740bd0f36aaa312c8d504e28c42163-Paper.pdf) (NeurIPS 2020):

**TASK:** multi-agent planning

**TL;DR:** This paper employs transformers to synthesize plans for communication of agents, where each agent must reach a goal position.

**Excluded** as the task is unclear, the goal is even more unclear and the plots are totally unclear.

