from dotenv import load_dotenv
load_dotenv()
from RepoLevelCodeGen import tree_structures, RetrievalAugmentation
from visualise import visualize_tree_structure
import argparse
import os
import openai
from openai import OpenAI

# Set up argument parser
parser = argparse.ArgumentParser(description='Visualize tree structure with configurable repository name and experiment.')
parser.add_argument('--repo_name', type=str, default='pypara', help='Name of the repository')
parser.add_argument('--experiment', type=str, default='', help='Name of the experiment')

args = parser.parse_args()

openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

experiment = ("_" + args.experiment) if args.experiment else args.experiment
SAVE_PATH = f"demo/{args.repo_name}{experiment}"

RA = RetrievalAugmentation(tree=SAVE_PATH)

TREE = RA.tree
ROOT_NODES = TREE.root_nodes.values()

TOP_ROOT_NODE = tree_structures.Node(
    "TOP_ROOT_NODE",
    index=-1,
    children=list(map(lambda x: x.index, ROOT_NODES)),
    embeddings=[],
)

visualize_tree_structure(TOP_ROOT_NODE, TREE, repo_name=args.repo_name, experiment=args.experiment)