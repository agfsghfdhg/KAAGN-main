import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    'csqa': {
        'train': '/home/xyNLP/data/kl/DRGN-main/data/csqa/train_rand_split.jsonl',
        'dev': '/home/xyNLP/data/kl/DRGN-main/data/csqa/dev_rand_split.jsonl',
        'test': '/home/xyNLP/data/kl/DRGN-main/data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/conceptnet.en.csv',
        'vocab': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/concept.txt',
        'patterns': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/matcher_patterns.json',
        'unpruned-graph': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': '/home/xyNLP/data/kl/DRGN-main/data/cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'statement': {
            'train': '/home/xyNLP/data/kl/DRGN-main/data/csqa/statement/train.statement.jsonl',
            'dev': '/home/xyNLP/data/kl/DRGN-main/data/csqa/statement/dev.statement.jsonl',
            'test': '/home/xyNLP/data/kl/DRGN-main/data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': '/home/xyNLP/data/kl/DRGN-main/data/csqa/grounded/train.grounded.jsonl',
            'dev': '/home/xyNLP/data/kl/DRGN-main/data/csqa/grounded/dev.grounded.jsonl',
            'test': '/home/xyNLP/data/kl/DRGN-main/data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/home/xyNLP/data/kl/DRGN-main/data/csqa/graph/train.graph.adj.pk',
            'adj-dev': '/home/xyNLP/data/kl/DRGN-main/data/csqa/graph/dev.graph.adj.pk',
            'adj-test': '/home/xyNLP/data/kl/DRGN-main/data/csqa/graph/test.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/train.statement.jsonl',
            'dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/dev.statement.jsonl',
            'test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/test.statement.jsonl',
            'train-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/grounded/train.grounded.jsonl',
            'dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/grounded/dev.grounded.jsonl',
            'test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/graph/train.graph.adj.pk',
            'adj-dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/graph/dev.graph.adj.pk',
            'adj-test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/graph/test.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/train-fact.statement.jsonl',
            'dev': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/dev-fact.statement.jsonl',
            'test': '/home/xyNLP/data/kl/DRGN-main/data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': '/home/xyNLP/data/kl/DRGN-main/data/obqa/fairseq/official/test-fact.jsonl',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['csqa'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'common': [
        #     # conceptnet.en.csv: 关系  头实体  尾实体  权重
        #     # concept.txt： ConceptNet图谱中 词汇表（实体表）
        #     {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
        #
        #     #   不进行修剪操作conceptnet.en.unpruned.graph
        #     #   Nodes:799272个结点  例：   53
        #     #   Edge:4302606条边   例： (1, 74038, 0)
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['unpruned-graph'], False)},
        #
        #     #   进行修剪操作conceptnet.en.pruned.graph
        #     #   Nodes:784110个结点  例：   53
        #     #   Edge:4302606条边   例： (1, 74038, 0)
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['pruned-graph'], True)},
        #
        #
        #     {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        # ],
        'csqa': [
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    print("***********")
    main()
    # pass
