import logging
import pickle
import json, os

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .SummarizationModels import BaseSummarizationModel
from .CodeDescriberModels import BaseCodeDescriberModel, OpenAICodeDescriberModel
from .CodeGeneratorModels import BaseCodeGeneratorModel, OpenAICodeGeneratorModel, PlainOpenAICodeGeneratorModel, GeminiFlashCodeGeneratorModel
from .CodeRetrievalModels import OpenAICodeRetrievalModel, GeminiFlashCodeRetrievalModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

from .utils import extract_functions_from_repo, add_imports, add_constructors

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        code_generator_model=None,
        embedding_model=None,
        summarization_model=None,
        code_describer_model=None,
        code_retrieval_model=None,
        tree_builder_type="cluster",
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=3,
        tr_selection_mode="top_k",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_code_retrieval_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_code_describer_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="OpenAI",
        tb_reduction_dimension=10,
        tb_max_node_in_cluster=5
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Validate code_generator_model
        if code_generator_model is not None and not isinstance(code_generator_model, BaseCodeGeneratorModel):
            raise ValueError("code_generator_model must be an instance of BaseCodeGeneratorModel")
        
        tr_code_retrieval_model = code_retrieval_model
        
        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model
            
        if code_describer_model is not None and not isinstance(
            code_describer_model, BaseCodeDescriberModel
        ):
            raise ValueError(
                "code_describer_model must be an instance of BaseCodeDescriberModel"
            )

        elif code_describer_model is not None:
            if tb_code_describer_model is not None:
                raise ValueError(
                    "Only one of 'tb_code_describer_model' or 'code_describer_model' should be provided, not both."
                )
            tb_code_describer_model = code_describer_model

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                code_describer_model=tb_code_describer_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
                reduction_dimension=tb_reduction_dimension,
                max_node_in_cluster=tb_max_node_in_cluster,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                code_retrieval_model=tr_code_retrieval_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.code_generator_model = code_generator_model or OpenAICodeGeneratorModel()
        self.code_retrieval_model = code_retrieval_model or OpenAICodeRetrievalModel("", "")
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            Code Generator Model: {code_generator_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            code_generator_model=self.code_generator_model,
            code_retrieval_model=self.code_retrieval_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.code_generator_model = config.code_generator_model
        self.code_retrieval_model = config.code_retrieval_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_projects(self, path_to_functions=None, path_to_repo=None, repo_name="", experiment="", verbose=False, clustering_method="GMM"):
        """
        Adds documents to the tree and creates a TreeRetriever instance.

        Args:
            path_to_functions (str): path to json file contains described functions
            path_to_repo (str): path to repo to construct the tree from
            experiment (str): explanation about the experiment
        """
        
        if path_to_functions is None and path_to_repo is None:
            raise ValueError("path_to_functions or path_to_repo must be provided.")
        
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                # self.add_to_existing(docs)
                return

        # Constructing the tree from repository extracting all the functions
        if path_to_repo is not None:
            functions = extract_functions_from_repo(path_to_repo)
            removed_functions = []
            functions_with_description = []
            for f in functions:
                description = self.tree_builder.describe(f)
                f["function_description"] = description
                if description == "Not sure" or description == "Not Implemented":
                    removed_functions.append(f)
                else:
                    functions_with_description.append(f)
                if verbose:
                    print(f["body"])
                    print(f["function_description"])
                    print()
            os.makedirs(f"repo_func_descr/{repo_name}/{experiment}", exist_ok=True)
            with open(f"repo_func_descr/{repo_name}/{experiment}/removed_functions.json", "w") as file:
                json.dump(removed_functions, file)
            with open(f"repo_func_descr/{repo_name}/{experiment}/functions_with_description.json", "w") as file:
                json.dump(functions_with_description, file)
                
        # Constructing the tree from already described functions (skipping function describing part)
        elif path_to_functions is not None:
            with open(os.path.join(path_to_functions, 'functions_with_description.json'), 'r') as file:
                functions_with_description = json.load(file)
            
            if verbose:
                for f in functions_with_description:
                    print(f["body"])
                    print(f["function_description"])
                    print()
                
        self.tree = self.tree_builder.build_from_functions(functions_with_description, clustering_method=clustering_method)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        file_content,
        target_function,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = False,
        rcg_retr: bool = True,          # RCG Implementation switch
        dense_rag: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            file_content (str): The content of the file to retrieve information from.
            target_function (str): The function to generate.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            file_content,
            target_function,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            return_layer_information,
            collapse_tree,
            rcg_retr=rcg_retr,
            dense_rag=dense_rag,
            rcg_retr_no_llm=False
        )

    def generate_code(
        self,
        file_content,
        target_function,
        repo_functions,
        context,
        dependencies
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            file_content (str): The content of the file to generate code from.
            target_function (str): The function to generate.
            repo_functions (list): The list of all functions with class names in the repository.
            context (list): retrieved nodes for code generation model
            dependencies (str): The dependencies in the environment where the code is being generated.
        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """

        if isinstance(self.code_generator_model, OpenAICodeGeneratorModel) or isinstance(self.code_generator_model, GeminiFlashCodeGeneratorModel):
            add_constructors(context, repo_functions)
            add_imports(context)
            answer = self.code_generator_model.generate_code(context, file_content, dependencies, target_function)
            return answer
        elif isinstance(self.code_generator_model, PlainOpenAICodeGeneratorModel):
            answer = self.code_generator_model.generate_code(target_function)
            return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
