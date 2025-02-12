- [Predicated Diffusion: Predicate Logic-Based Attention Guidance for Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Sueyoshi_Predicated_Diffusion_Predicate_Logic-Based_Attention_Guidance_for_Text-to-Image_Diffusion_Models_CVPR_2024_paper.pdf) (CVPR 2024):

    **TASK:** Image Generation

    **TL;DR:** A method is proposed for constrained image generation using First-Order Logic (FOL) rules, extracted from text and enforced via attention-based loss functions.


- [Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees](https://ojs.aaai.org/index.php/AAAI/article/view/29455) (AAAI 2024):
    
    **TASK:** Abductive Learning
    
    **TL;DR:** A position-based loss is proposed for NeSy methods that exploit KBs for predictions.
    
    **Excluded** as the addition of this new loss is the main contribution of this paper and experiments on real-world datasets are not provided.

    
- [On the Hardness of Probabilistic Neurosymbolic Learning](https://arxiv.org/pdf/2406.04472) (ICML 2024):
    
    **TASK:** Probabilistic Reasoning

    **TL;DR:** This paper investigates the complexity of differentiating probabilistic reasoning in neurosymbolic models trained via gradient descent.

    **Excluded** as benchmarking is limited to datasets of CNF formulas.

    
- [End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations](https://openreview.net/pdf?id=0P3kaNluGj) (ICML 2024):
    
    **Task:** Gaming on Atari Videogames
    
    **TL;DR:** The key innovation is distilling a vision foundation model into an efficient perception module, refining it through policy learning.

    **Excluded** as the main contribution is not of NeSy nature.

    
- [On the Independence Assumption in Neurosymbolic Learning](https://arxiv.org/pdf/2404.08458) (ICML 2024):
    
    **Task:** Probabilistic Reasoning

    **TL;DR:** Here it is investigated how independence assumption discards valuable assignments and the losses are non-convex and with sparse local-minima.

    **Excluded** as it is a strictly theoretical paper.

    
- [LogicMP: A Neuro-symbolic Approach for Encoding First-order Logic Constraints](https://arxiv.org/pdf/2309.15458) (ICLR 2024):
    
    **TASK:** Constrained Prediction
    
    **TL;DR:** This paper examines the conditional independence assumption in probabilistic neuro-symbolic learning systems, commonly used to simplify reasoning over logical constraints.

    **Excluded** as no benchmarks are provided.
    

- [Unveiling Implicit Deceptive Patterns in Multi-Modal Fake News via Neuro-Symbolic Reasoning](https://ojs.aaai.org/index.php/AAAI/article/view/28677) (AAAI 2024):
    
    **TASK:** Image-Text verification

    **TL;DR:** The model extracts patterns using encoders and employs a teacher-student network to estimate three latent variables: image manipulation, cross-modal inconsistency, and image repurposing.

- [Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs](https://proceedings.neurips.cc/paper_files/paper/2023/file/5965f3a748a8d41415db2bfa44635cc3-Paper-Conference.pdf) (NeurIPS 2023)

    **TASK:** KG reasoning

    **TL;DR:** This method employs embeddings to parametrize the truth value of ground triplets, while optimizing to assign a score to a given set of logic rules containing such triplets as costituents.

    **Excluded** as more recent techniques are treated for the same task with intersection on the employed benchmark datasets. Furthermore, the complexity of the algorithm, requiring a two-step expectation maximization technique is another reason to exclude it in the comparison with other models.
    

- [Simple Augmentations of Logical Rules for Neuro-Symbolic Knowledge Graph Completion.](https://aclanthology.org/2023.acl-short.23.pdf) (ACL 2023)

    **TASK:** KG reasoning

    **TL;DR:** a bunch of rule augmentation techniques are proposed to boost performance of an old NeSy method.

    **Excluded** as more recent papers tackle the problem with more substantial contributions.


- [A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference](https://proceedings.neurips.cc/paper_files/paper/2023/file/4d9944ab3330fe6af8efb9260aa9f307-Paper-Conference.pdf) (NeurIPS 2023):
    
    **TASK:** Probabilistic Inference
    
    **TL;DR:** The scalability issue of methods such as DeepProbLog is mitigated with NN approaches. 

    **Excluded** since evaluation is limited to a toy tasks.

    
- [ART: rule bAsed futuRe-inference deducTion](https://aclanthology.org/2023.emnlp-main.592.pdf) (EMNLP 2023):
    
    **TASK:** Reasoning over Videos
    
    **TL;DR:** In this paper a benchmark for video resoning is introduced together with a network capable of performing inference on future events of such videos starting from a set of rules written in natural language. 
    
    **Excluded** as the resoning process is casted as a graph pathfinding problem. In our survey, we don't consider any graph to be valid for rule-enforcement, but DFAs.

    
- [Soft-Unification in Deep Probabilistic Logic](https://proceedings.neurips.cc/paper_files/paper/2023/file/bf215fa7fe70a38c5e967e59c44a99d0-Paper-Conference.pdf) (NeurIPS 2023):
    
    **TASK:** Probabilistic Inference 
    
    **TL;DR:** The methodology proposed in this paper uses probabilistic semantics to ensure non-redundancy, well-defined proof scores, and non-sparse gradients.

    **Excluded** as benchmarking is limited to tasks defined on the MNIST dataset

    
- [Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts](https://proceedings.neurips.cc/paper_files/paper/2023/file/e560202b6e779a82478edb46c6f8f4dd-Paper-Conference.pdf) (NeurIPS 2023):
    
    **TASK:** Probabilistic Inference
    
    **TL;DR:** This paper investigates reasoning shortcuts in NeSy approaches. A definition of the NeSy approaches that are covered in the paper is given in section 2, with MNIST sum as an example.
    
    **Excluded** as the contribution of the paper is the analysis of reasoning shortcuts. Furthermore, the main benchmark is MNIST addition even though the autonomous veichle benchmark is quite promising.

    
- [Softened Symbol Grounding for Neuro-symbolic Systems](https://arxiv.org/pdf/2403.00323) (ICLR 2023):
    
    **TASK:** Represented symbol grounding
    
    **TL;DR:** The paper presents a novel neuro-symbolic learning framework that improves symbol grounding.

    **Excluded** as benchmarking is limited to handwritten formula evaluation, visual Sudoku classification, and graph pathfinding.


- [Parallel Neurosymbolic Integration with Concordia](https://proceedings.mlr.press/v202/feldstein23a/feldstein23a.pdf) (ICML 2023):
    
    **TASK:** Recommendation, Classification and Entity Linking.
    
    **TL;DR:** This method is similar to LogicMP as it proposes to merge a NN with a known logic theory through means of additional bridging components. Interestingly, one may add useless rules whose weight will be set to 0 if their uselessness is discovered at training time.

    **Excluded** as improvements on the recoomendation task were marginal and reported using RMSE instead of nDCG.

    
- [Neurosymbolic Reasoning and Learning with Restricted Boltzmann Machines](https://ojs.aaai.org/index.php/AAAI/article/view/25806) (AAAI 2023):
    
    **TASK:** Logical Reasoning

    **TL;DR:** In this work we have an equivalence demonstration between propositional logic and restricted Boltzmann machines.
    
    **Excluded** as rather than adding an interpretable component to black-boxes, a neural sat-solver is formulated as future work, which is out of the scope of our survey.


- [NeuPSL: Neural Probabilistic Soft Logic](https://arxiv.org/pdf/2205.14268) (IJCAI 2023):
    
    **TASK:** Probabilistic Inference
    
    **TL;DR:** A NeSy approach that extends Probabilistic Soft Logic by performing neural encoding of the input. 

    **Excluded** as the first two benchmarks were MNIST addition and visual sudoku, while the last benchmark proves how this method falls behind black box approaches by a huge gap of 20% accuracy (Citeseer node classification).

    
- [Safe Reinforcement Learning via Probabilistic Logic Shields](https://arxiv.org/pdf/2303.03226) (IJCAI 2023):
    
    **TASK:** RL shielding 
    
    **TL;DR:** Here it is proposed a neuro-symbolic reinforcement learning policy that guarantees action safety by using a probabilistic program divided into three parts: an annotated disjunction representing the policy, probabilistic facts for the current state, and a safety specification.
    

- [Using Domain Knowledge to Guide Dialog Structure Induction via Neural Probabilistic Soft Logic](https://arxiv.org/pdf/2403.17853) (ACL 2023):
    
    **TASK:** Dialog Structure Induction
    
    **TL;DR:** Here PSL is integrated with a Neural Model to inject constraints over the construction of the dialog. 
    

- [A Logic-based Approach to Contrastive Explainability for Neurosymbolic Visual Question Answering](https://www.ijcai.org/proceedings/2023/0408.pdf) (IJCAI 2023):
    
    **TASK:** Visual Question Answering
    
    **TL;DR:** The idea behind this paper is that we want to generate contrastive explanations (what has to be changed to make a statement called "foil" be true) by suggesting changes in the images. The method is NeSy as it leverages Answer Set Programming.
    
    **Excluded** as contrastive explainability can be counterintuitive if many things have to be changed to make the foil true. Additionally, other techniques evaluated on CLEVR use DSL instead of ASP.

    
- [Mitigating Reporting Bias in Semi-supervised Temporal Commonsense Inference with Probabilistic Soft Logic](https://ojs.aaai.org/index.php/AAAI/article/view/21288) (AAAI 2022):
    
    **TASK:** Temporal Common Sense Reasoning

    **TL;DR:** The method tackles the tendency to report unusual time-related events more frequently than routine ones.
    
    
- [Neuro-Symbolic Verification of Deep Neural Networks](https://arxiv.org/pdf/2203.00938) (IJCAI 2022):
    
    **TASK:** Formal Verification
    
    **TL;DR:** introduce a formal framework for verifying neural networks, centered around the Neuro-Symbolic Assertion Language (NeSAL), a fragment of first-order logic used to specify and verify properties of neural networks.

    
- [Safe Neurosymbolic Learning with Differentiable Symbolic Execution](https://arxiv.org/pdf/2203.07671) (ICLR 2022):
    
    **TASK:** Safety Contraints Enforcement on NeSy Programs
    
    **TL;DR:** In this paper the scenario is represented by a program that includes a value taken from a NN. To garant safeness, a formalism to express the constraints over the variables, referencing the lines of code, is specified. The NN can be optimized to output values that satisfy the safety constraint of the program. For this sake, the loss has a data-fidelity component and an unsafeness component relative to the states of the program.

    **Excluded** as both the task and the benchmarks are oddly specific.
    

- [Semantic Probabilistic Layers for Neuro-Symbolic Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/c182ec594f38926b7fcb827635b9a8f4-Paper-Conference.pdf) (NeurIPS 2022):
    
    **TASK:** Structured-Output Prediction
    
    **TL;DR:** This is rule-enforcement kind of paper where the objective is to perform structured output tasks, such as graph pathfinding or hierarchical multi-label classification through a neural circuit. 

    **Excluded** as rules are enforced by the NN circuit, so it’s hard to specify them externally.

- [Linguistic Frameworks Go Toe-to-Toe at Neuro-Symbolic Language Modeling](https://arxiv.org/pdf/2112.07874) (NAACL 2022):
    
    **TASK:** Language Modelling
    
    **TL;DR:** This paper showcases the efficacy of using parsing mechanisms to build trees that are used for the purpose of next token prediction. The paper displays four different parsing techniques and demonstrates the usefullness of Prague Tectogrammatical Graphs (PTG) and Elementary Dependency Structures (EDS) for this task.

    **Excluded** as including semantic parsing techniques in NLP pipelines is not a novel practice and most importantly it's not NeSy.


- [VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming](https://arxiv.org/pdf/2202.04178)(https://arxiv.org/pdf/2202.04178):

    **TASK:** Image Generation

    **TL;DR:** A pipeline bridging VAEs and probabilistic logic programming.

    **Excluded** as evaluation is carried out on MNIST and a custom block-world dataset.


- [Temporal-Logic-Based Reward Shaping for Continuing Reinforcement Learning Tasks](https://ojs.aaai.org/index.php/AAAI/article/view/16975) (AAAI 2021):
    
    **TASK:** RL Shaping
    
    **TL;DR:** The contribution lies in modifying the behavior policy by incorporating a potential-based reward term, while still guaranteeing that the optimal policy can be recovered. This potential term is defined using a deterministic state automaton (DSA) that encodes the temporal logic constraints we want the agent to adhere to.

    
- [Self-Supervised Self-Supervision by Combining Deep Learning and Probabilistic Logic](https://ojs.aaai.org/index.php/AAAI/article/view/16631) (AAAI 2021):
    
    **TASK:** Text Classification
    
    **TL;DR:** In this paper, self-supervision (the task of automatically finding the labels for data by solving a proxy problem) is enhanced by means of Deep Probabilistic Logic. More specifically, a set of initial rules is specified and the potentials are used to guide the self-supervised self-supervision process. An initial seed of virtual evidence is used, the labels are inferred and the features are learnt, then the process is iterated.

    **Excluded** as a set of initial rules is required, active learning should be avoided, but it's still present. Furthermore, this is a very sophisticated technique for text classification, which already has plenty of literature regarding explainable methods. 

    
- [Clinical Temporal Relation Extraction with Probabilistic Soft Logic Regularization and Global Inference](https://ojs.aaai.org/index.php/AAAI/article/view/17721) (AAAI 2021):
    
    **TASK:** Temporal Relation Extraction
    
    **TL;DR:** The paper presents a method for extracting temporal relations between clinical events at the document level. It leverages Probabilistic Soft Logic Regularization and Global Inference to model global dependencies among events.

    **Excluded** as the baselines used for comparison are quite obsolete (SVM-based). Furthermore, more recent techniques have been found in the "temporal" scope.

    
- [Generative Neurosymbolic Machines](https://proceedings.neurips.cc/paper/2020/hash/94c28dcfc97557df0df6d1f7222fc384-Abstract.html) (NeurIPS 2020):
    
    **TASK:** Image Generation
    
    **TL;DR:** Here the NeSy part lies in the simbolic role of a latent variable of the markovian process, for instance presence, position, depth and appearance can be symbolic roles for objects in the final picture. This can be added in the context of the constrained diffusion process.

    **Excluded** as this work is evaluated on custom-made datasets derived from MNIST and CLEVR. Additionally, a more recent approach (Predicated Diffusion) achieved better results due to the evolution of image generation techniques.
    
    
- [Neurosymbolic Reinforcement Learning with Formally Verified Exploration](https://proceedings.neurips.cc/paper_files/paper/2020/file/448d5eda79895153938a8431919f4c9f-Paper.pdf) (NeurIPS 2020):
    
    **TASK:** RL shielding
    
    **TL;DR:** Here symbolic policies are lifted to the neural space to allow shielded RL in a continuous environment.

    **Excluded** as more recent shielding techniques have been found.


- [BOWL: Bayesian Optimization for Weight Learning in Probabilistic Soft Logic](https://ojs.aaai.org/index.php/AAAI/article/view/6589) (AAAI 2020)

    **TASK:** Reasoning on a Knowledge Base

    **TL;DR:** Follow-up work of Probabilistic Soft Logic (check below).

    **Excluded** as this is an improvement of a foundational work. 
    

- [Deep Probabilistic Logic: A Unifying Framework for Indirect Supervision](https://arxiv.org/pdf/1808.08485) (EMNLP 2018):
    
    **TASK:** Text-based Relation Extraction
    
    **TL;DR:** DPL models label decisions as latent variables, represents prior knowledge on their relations using weighted first-order logical formulas, and alternates between learning a deep neural network for the end task and refining uncertain formula weights for indirect supervision, using variational EM.

    **Excluded** since this foundational model is quite ubiquitous in the NeSy literature.


- [Hinge-Loss Markov Random Fields and Probabilistic Soft Logic](https://arxiv.org/pdf/1505.04406) (J. Mach. Learn. Res. 2017):

     **TASK:** Reasoning on a Knowledge Base

     **TL;DR:** Very famous foundational model that creates a framework to learn weights for rules using gradient-based techniques.

     **Excluded** as this method is employed by many NeSy methods, thus it is more convenient to treat those works instead of this.