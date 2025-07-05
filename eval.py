"""
clean version of evaluation
"""
import os
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
import logging
import numpy as np
import argparse
import csv
import random
from text_dataset import TextDataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                        RobertaConfig, RobertaModel, RobertaTokenizer)
from embedding import ModelEmbedding, get_embedding_model
import metric
from prettytable import PrettyTable
from attention import MultiHeadedAttention, gMultiHeadedAttention, SelfAttention
from llm_gen import generate_rank, generate_exinfo

logger = logging.getLogger(__name__)


class ResultTable:
    def __init__(self, title):
        self.table = PrettyTable(title)       
    def add_row(self,row):
        self.table.add_row(row)
    def print_table(self):
        print(self.table)
def get_datasets(args, tokenizer,data_file, use_origin=True):
    """
    get datasets
    :param use_origin: return text dataset
    :return tokenized_dataloader:sequential dataloader for encoded input
    :return origin_dataset: origin dataset for text
    """
    tokenized_dataset = TextDataset(args, tokenizer, "tokenize", data_file)
    origin_dataset = None
    if use_origin:
        origin_dataset = TextDataset(args, tokenizer, "text", data_file)
    sampler = SequentialSampler(tokenized_dataset)
    tokenized_dataloader = DataLoader(tokenized_dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)
    return tokenized_dataloader, origin_dataset


def get_embeddings(device, embedding_model,dataloader, is_nl, save_vector_path=None):
    """
    return matrix of embeddings (num, hidden_size)
    """
    vecs = []
    for batch in tqdm(dataloader):
        # batch[0]: code_input
        # btach[1]: nl_input
        if is_nl:
            inputs = batch[1].to(device)
        else:
            inputs = batch[0].to(device)
        embeds = embedding_model.get_embedding(inputs, is_nl)
        vecs.append(embeds.cpu().numpy())
    vecs = np.concatenate(vecs,0)
    if save_vector_path:
        logger.info(f"saving query vector to {save_vector_path} {vecs.shape}")
        np.save(save_vector_path, vecs)
    return vecs


def get_rank(args, scores):
    query_dataset = TextDataset(args, None, "text", args.query_data_file)
    code_dataset = TextDataset(args, None, "text", args.code_data_file)

    nl_urls = []
    nl_in = []
    for ex in query_dataset:
        nl_in.append({'url': ex['url'], 'query': ex['nl_input']})
        nl_urls.append(ex['url'])

    code_urls = []
    code_in = []
    for ex in code_dataset:
        code_in.append({'url': ex['url'], 'code': ex['code_input']})
        code_urls.append(ex['url'])

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    sort_in = [[code_in[idx] for idx in sort_id] for sort_id in sort_ids]
    sort_urls = [[code_urls[idx] for idx in sort_id] for sort_id in sort_ids]

    final_ans = []
    k = 10
    for i in tqdm(range(len(sort_in))):
        top_codes = [{'url': code['url'], 'code': code['code']} for code in sort_in[i][:k]]
        res = {'nl_input': nl_in[i]['query'], 'code_input': top_codes, 'url':  nl_in[i]['url']}
        final_ans.append(generate_rank(res))

    for i in range(len(sort_urls)):
        if len(final_ans[i]) >= k:
            sort_urls[i][:k] = final_ans[i][:k]
        else:
            sort_urls[i][:len(final_ans[i])] = final_ans[i]

    title = ["MRR", "Top-1", "Top-5", "Top-10"]
    result_table = ResultTable(title)

    mrrs = metric.cal_mrr(sort_urls, nl_urls)
    recalls = metric.cal_recall(sort_urls, nl_urls)
    mrrs = [round(float(r), 3) for r in list(mrrs.values())]
    # mrr10 mrr1000, hr1,hr5, hr10
    mrrs = [mrrs[5]]
    recalls = [round(float(r), 3) for r in list(recalls.values())][:3]
    ans = mrrs + recalls
    result_table.add_row(ans)
    result_table.print_table()


def get_exinfomation(args, scores):
    query_dataset = TextDataset(args, None, "text", args.query_data_file)
    code_dataset = TextDataset(args, None, "text", args.code_data_file)

    nl_urls = []
    for ex in query_dataset:
        nl_urls.append({'url': ex['url'], 'query': ex['nl_input']})
    code_urls = []
    for ex in code_dataset:
        code_urls.append({'url': ex['url'], 'code': ex['code_input']})

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    sort_urls = [[code_urls[idx] for idx in sort_id] for sort_id in sort_ids]

    # k = [1, 5, 7, 9]
    k = [3]
    for x in k:
        final_ans = []
        for i in tqdm(range(len(sort_urls))):
            top_codes = [{'url': code['url'], 'code': code['code']} for code in sort_urls[i][:x]]
            res = {'nl_input': nl_urls[i]['query'], 'code_input': top_codes, 'url': nl_urls[i]['url']}
            exinfo = generate_exinfo(args, res)
            final_ans.append({'nl_input': exinfo, 'url': nl_urls[i]['url']})

        # print(final_ans[0])
        filename = os.path.join(args.output_path, f'{args.lang}_test_exquery_1_{x}.jsonl')
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(final_ans, file, indent=4)
        print(f"Data has been written to {filename}")


def evaluate(args,query2code_scores,query2comment_scores, code2code_scores, gendes2code_scores=None):
    if args.mode == 'eval':
        assert args.w2 != None and args.w3 != None

        query_dataset = TextDataset(args,None,"text",args.query_data_file)
        code_dataset = TextDataset(args, None, "text",args.code_data_file)
        gen_code_dataset = TextDataset(args, None, "text", args.gen_code_data_file)
        
        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])
            
        # test different weights
        title = ["MRR","Top-1","Top-5","Top-10"]
        result_table = ResultTable(title)   
        w = [args.w1,args.w2, args.w3]
            
        res = get_results(w[0]*query2code_scores + w[1]*query2comment_scores + w[2]*code2code_scores,nl_urls, code_urls)
    
        result_table.add_row(res)
        result_table.print_table()
    elif args.mode == 'eval1':
        query_dataset = TextDataset(args, None, "text", args.query_data_file)
        code_dataset = TextDataset(args, None, "text", args.code_data_file)

        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])

        # test different weights
        title = ["MRR", "Top-1", "Top-5", "Top-10"]
        result_table = ResultTable(title)
        w = [args.w1, args.w2, args.w3, args.w4]
        if gendes2code_scores is not None:
            res = get_results(w[0] * query2code_scores + w[1] * query2comment_scores + w[2] * code2code_scores + w[3] * gendes2code_scores, nl_urls,
                          code_urls)
        elif code2code_scores is not None:
            res = get_results(w[0] * query2code_scores + w[1] * query2comment_scores + w[2] * code2code_scores, nl_urls,
                          code_urls)
        else:
            res = get_results(query2code_scores, nl_urls, code_urls)

        result_table.add_row(res)
        result_table.print_table()

    elif args.mode == 'traverse':
        query_dataset = TextDataset(args, None, "text", args.query_data_file)
        code_dataset = TextDataset(args, None, "text", args.code_data_file)
        nl_urls = []
        for ex in query_dataset:
            nl_urls.append(ex['url'])
        code_urls = []
        for ex in code_dataset:
            code_urls.append(ex['url'])

        # create result table
        title = ["weights","MRR","Top-1","Top-5","Top-10"]
        result_table = ResultTable(title)   
        # traverse with different weights
        step_size = 0.01
        # traverse all the weights, where w1+w2+w3=1, with step_size
        w1_list = np.arange(0,1+step_size,step_size)
        w2_list = np.arange(0,1+step_size,step_size)
        w3_list = np.arange(0,1+step_size,step_size)
        ans_list = []
        control_ans_list = []
        for w1 in tqdm(w1_list):
            for w2 in w2_list:
                if w1 + w2 > 1:
                    continue
                for w3 in w3_list:
                    # check if the sum of w1,w2,w3 is 1
                    if w1 + w2 + w3 != 1:
                        continue
                    res = get_results(w1*query2code_scores + w2*query2comment_scores + w3*code2code_scores,nl_urls, code_urls)
                    r = ["-".join([str(round(w1,2)),str(round(w2,2)),str(round(w3,2))])] + res
                    ans_list.append(r)
        with open('traverse.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(ans_list)


def get_results(scores, nl_urls, code_urls):
    """
    given scores matrix(nl,cl) and labeld urls, return a list containing:
    """
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1] 
        
    sort_urls = [[code_urls[idx] for idx in sort_id] for sort_id in sort_ids]

    mrrs = metric.cal_mrr(sort_urls, nl_urls)
    recalls = metric.cal_recall(sort_urls, nl_urls)
    mrrs = [round(float(r),3) for r in list(mrrs.values())]
    # mrr10 mrr1000, hr1,hr5, hr10
    mrrs = [mrrs[5]]
    recalls = [round(float(r),3) for r in list(recalls.values())][:3]
    return mrrs + recalls
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--code_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--comment_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).") 
    parser.add_argument("--gen_code_data_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).") 
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    # there're two queries embeddings maybe
    parser.add_argument("--query2code_cache_path", default=None, type=str, 
                        help="tt")
    parser.add_argument("--query2comment_cache_path", default=None, type=str, 
                        help="tt")
    parser.add_argument("--query_target_code_cache_path", default=None, type=str, 
                        help="maching query2code")
    parser.add_argument("--gencode_target_code_cache_path", default=None, type=str,help="matching gencode cache")

    parser.add_argument("--gencode_target_recode_cache_path", default=None, type=str,help="exteng gencode cache")

    parser.add_argument("--gen_code_python_cache_path", default=None, type=str,help="extend gencode cache")

    parser.add_argument("--query_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--query_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--query_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--code_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gencode_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gendes1_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--comment1_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gpt_exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--ds_exquery_unixcoder_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_cocosoda_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_bge_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--qwen_exquery_unixcoder_path", default=None, type=str,
                        help="tt")


    parser.add_argument("--comment_cache_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--gen_code_cache_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--output_path", default=None, type=str,
                        help="tt")
    parser.add_argument("--w1", default=None, type=float, 
                        help="tt")
    parser.add_argument("--w2", default=None, type=float, 
                        help="tt")    
    parser.add_argument("--w3", default=None, type=float, 
                        help="tt")
    parser.add_argument("--w4", default=None, type=float,
                        help="tt")
    
    parser.add_argument("--mode",default=None, type=str,help="eval/traverse", required=True)
    
    parser.add_argument("--model_name_or_path",default=None, type=str,help="embedding/eval")
    parser.add_argument("--format",default=None, type=str,help="query/code/comment/gencode")

    parser.add_argument("--output_dir",default=None, type=str,help="embedding/eval")
    parser.add_argument("--device",default='cuda', type=str,help="embedding/eval")
    parser.add_argument("--datafile",default=None, type=str,help="embedding/eval")
    


    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    print(args)
    n_gpu = torch.cuda.device_count()
    print(f"n_gpu: {n_gpu}")
    
    
    if args.mode == "embedding":
        # get embeddings and store
        embedding_model=get_embedding_model(n_gpu, args.device, args.model_name_or_path)
        dataloader, _ = get_datasets(args,embedding_model.tokenizer, args.datafile, use_origin=False)
        is_nl = "query" in args.format or "comment" in args.format or "gendes" in args.format or "exquery" in args.format
        dir_path = os.path.join(
                           args.output_dir,
                           args.model_name_or_path, 
                           args.lang)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_vector_path=os.path.join(dir_path,f"{args.lang}-{args.format}-qwen-3.npy")
        # save_vector_path=os.path.join(dir_path,f"{args.lang}-{args.format}.npy")
        logger.info(f"getting embedding of {args.lang} {args.format}")
        logger.info(f"save vectors to {save_vector_path}, is_nl = {is_nl}")
        get_embeddings(args.device, 
                       embedding_model,
                       dataloader,
                       is_nl,
                       save_vector_path=save_vector_path
                       )
    elif args.mode == "eval":
        print("=====evaluating {}=====".format(args.lang))

        gencode_recode_embedding = np.load(args.gencode_target_recode_cache_path)
        gencode_python_embedding = np.load(args.gen_code_python_cache_path)

        # test different models and settings 
        query2code_embedding = np.load(args.query2code_cache_path)
        query2comment_embedding = np.load(args.query2comment_cache_path)
        # target_code_embedding = np.load(args.code_cache_path)
        comment_embedding = np.load(args.comment_cache_path)
        gen_code_embedding = np.load(args.gen_code_cache_path)
        query_target_code_embedding = np.load(args.query_target_code_cache_path)
        gencode_target_code_embedding = np.load(args.gencode_target_code_cache_path)
          
        # query-comment + code-2code
        query2comment_scores = query2comment_embedding @ comment_embedding.T
        query2code_scores = query2code_embedding @ query_target_code_embedding.T
        code2code_scores = gen_code_embedding @ gencode_target_code_embedding.T

        evaluate(args,query2code_scores, query2comment_scores, code2code_scores,)
    # elif args.mode == "eval1":
    #     print("=====evaluating {}=====".format(args.lang))
    #     device = args.device
    #     global_att = MultiHeadedAttention(4, 1024 * 1, 0.1, device)
    #     # global_att = gMultiHeadedAttention(4, 1024 * 3, 0.3, device)
    #
    #     global_att.to(device)
    #
    #     # 加载numpy数组并转换为PyTorch张量
    #     query2code_embedding = torch.tensor(np.load(args.query2code_cache_path)).to(device)
    #     query2comment_embedding = torch.tensor(np.load(args.query2comment_cache_path)).to(device)
    #     gen_code_embedding = torch.tensor(np.load(args.gen_code_cache_path)).to(device)
    #     query_target_code_embedding = torch.tensor(np.load(args.query_target_code_cache_path)).to(device)
    #     comment_embedding = torch.tensor(np.load(args.comment_cache_path)).to(device)
    #     gencode_target_code_embedding = torch.tensor(np.load(args.gencode_target_code_cache_path)).to(device)
    #
    #     # global_query = torch.cat((query2code_embedding, query2comment_embedding, gen_code_embedding), dim=-1).unsqueeze(1)
    #     # global_target = torch.cat((query_target_code_embedding, comment_embedding, gencode_target_code_embedding), dim=-1).unsqueeze(1)
    #
    #     global_query = torch.cat((query2code_embedding, query2comment_embedding, gen_code_embedding), dim=-1).unsqueeze(1)
    #     global_target = torch.cat((query_target_code_embedding, comment_embedding, gencode_target_code_embedding), dim=-1).unsqueeze(1)
    #
    #     query2code_embedding_ex = query2code_embedding.unsqueeze(1)
    #     query2comment_embedding_ex = query2comment_embedding.unsqueeze(1)
    #     gen_code_embedding_ex = gen_code_embedding.unsqueeze(1)
    #     query_target_code_embedding_ex = query_target_code_embedding.unsqueeze(1)
    #     comment_embedding_ex = comment_embedding.unsqueeze(1)
    #     gencode_target_code_embedding_ex = gencode_target_code_embedding.unsqueeze(1)
    #
    #
    #     # global_query = torch.cat((query2code_embedding, query2comment_embedding, gen_code_embedding), dim=-1)
    #     # global_target = torch.cat((query_target_code_embedding, comment_embedding, gencode_target_code_embedding), dim=-1)
    #
    #     # query = global_att(global_query, global_query, global_query).squeeze(1)
    #     # target = global_att(global_target, global_target, global_target).squeeze(1)
    #
    #     query = global_att(gen_code_embedding_ex, query2comment_embedding_ex, query2code_embedding_ex).squeeze(1)
    #     target = global_att(gencode_target_code_embedding_ex, comment_embedding_ex, query_target_code_embedding_ex).squeeze(1)
    #
    #     # query2target_scores = global_query.cpu().numpy() @ global_target.cpu().numpy().T
    #     query2target_scores = query.detach().cpu().numpy() @ target.detach().cpu().numpy().T
    #     evaluate(args, query2target_scores, None, None)
    # elif args.mode == "eval1":
    #     print("=====evaluating {}=====".format(args.lang))
    #
    #     # test different models and settings
    #     query2code_embedding = np.load(args.query2code_cache_path)
    #     query2comment_embedding = np.load(args.query2comment_cache_path)
    #     comment_embedding = np.load(args.comment_cache_path)
    #     gen_code_embedding = np.load(args.gen_code_cache_path)
    #     query_target_code_embedding = np.load(args.query_target_code_cache_path)
    #     gencode_target_code_embedding = np.load(args.gencode_target_code_cache_path)
    #
    #     gencode_recode_embedding = np.load(args.gencode_target_recode_cache_path)
    #     gencode_python_embedding = np.load(args.gen_code_python_cache_path)
    #
    #     # query2code_sum = np.sum(query2code_embedding, axis=1).reshape(1, -1)
    #     # query2comment_sum = np.sum(query2comment_embedding, axis=1).reshape(1, -1)
    #     # comment_sum = np.sum(comment_embedding, axis=1).reshape(1, -1)
    #     # gen_code_sum = np.sum(gen_code_embedding, axis=1).reshape(1, -1)
    #     # query_target_code_sum = np.sum(query_target_code_embedding, axis=1).reshape(1, -1)
    #     # gencode_target_code_sum = np.sum(gencode_target_code_embedding, axis=1).reshape(1, -1)
    #
    #     # n = query2code_sum.shape[1]
    #     # m = query_target_code_sum.shape[1]
    #     #
    #     # query2code_n_by_n = np.tile(query2code_sum, (m, 1))
    #     # query2comment_n_by_n = np.tile(query2comment_sum, (m, 1))
    #     # comment_n_by_n = np.tile(comment_sum, (n, 1))
    #     # gen_code_n_by_n = np.tile(gen_code_sum, (m, 1))
    #     # query_target_code_n_by_n = np.tile(query_target_code_sum, (n, 1))
    #     # gencode_target_code_n_by_n = np.tile(gencode_target_code_sum, (n, 1))
    #
    #     query2comment_scores = query2comment_embedding @ comment_embedding.T
    #     query2code_scores = query2code_embedding @ query_target_code_embedding.T
    #     code2code_scores = gen_code_embedding @ gencode_target_code_embedding.T
    #
    #
    #     # print("query2codeembedding shape:", query2code_embedding.shape)
    #     # print("query2commentembedding shape:", query2comment_embedding.shape)
    #     # print("commentembedding shape:", comment_embedding.shape)
    #     # print("gencodeembedding shape:", gen_code_embedding.shape)
    #     # print("querytargetcodeembedding shape:", query_target_code_embedding.shape)
    #     # print("gencodetargetcodeembedding shape:", gencode_target_code_embedding.shape)
    #
    #
    #     # print("query2codeembedding shape:", query2code_n_by_n.shape)
    #     # print("query2commentembedding shape:", query2comment_n_by_n.shape)
    #     # print("commentembedding shape:", comment_n_by_n.shape)
    #     # print("gencodeembedding shape:", gen_code_n_by_n.shape)
    #     # print("querytargetcodeembedding shape:", query_target_code_n_by_n.shape)
    #     # print("gencodetargetcodeembedding shape:", gencode_target_code_n_by_n.shape)
    #
    #
    #     # global_query_1 = np.concatenate((query2code_embedding, query2comment_embedding), axis=-1)
    #     # global_query_2 = np.concatenate((query2code_embedding, gen_code_embedding), axis=-1)
    #     # global_query_3 = np.concatenate((query2comment_embedding, gen_code_embedding), axis=-1)
    #     global_query_4 = np.concatenate((query2code_embedding, query2comment_embedding, gen_code_embedding), axis=-1)
    #
    #     # global_target_1 = np.concatenate((query_target_code_embedding, comment_embedding), axis=-1)
    #     # global_target_2 = np.concatenate((query_target_code_embedding, gencode_target_code_embedding), axis=-1)
    #     # global_target_3 = np.concatenate((comment_embedding, gencode_target_code_embedding), axis=-1)
    #     global_target_4 = np.concatenate((query_target_code_embedding, comment_embedding, gencode_target_code_embedding), axis=-1)
    #
    #     global_query_1 = np.concatenate((query2code_embedding, query2comment_embedding), axis=-1)
    #     global_query_2 = np.concatenate((query2code_embedding, gen_code_embedding), axis=-1)
    #     global_query_3 = np.concatenate((query2comment_embedding, gen_code_embedding), axis=-1)
    #
    #     global_target_1 = np.concatenate((query_target_code_embedding, comment_embedding), axis=-1)
    #     global_target_2 = np.concatenate((query_target_code_embedding, gencode_target_code_embedding), axis=-1)
    #     global_target_3 = np.concatenate((comment_embedding, gencode_target_code_embedding), axis=-1)
    #
    #     global_query_1 = np.concatenate((gencode_python_embedding, gencode_python_embedding), axis=-1)
    #     global_target_1 = np.concatenate((gencode_target_code_embedding, gencode_target_code_embedding), axis=-1)
    #
    #     # global_query_1 = np.concatenate((query2code_embedding, query2comment_embedding, gen_code_embedding, gencode_python_embedding), axis=-1)
    #     # global_target_1 = np.concatenate((query_target_code_embedding, comment_embedding, gencode_target_code_embedding, gencode_recode_embedding), axis=-1)
    #
    #
    #     scores_1 = global_query_1 @ global_target_1.T
    #     scores_2 = global_query_2 @ global_target_2.T
    #     scores_3 = global_query_3 @ global_target_3.T
    #     scores_4 = global_query_4 @global_target_4.T
    #
    #     evaluate(args, scores_1, None, None)
    #     # evaluate(args, query2comment_scores, query2code_scores, code2code_scores)
    # elif args.mode == "eval1":
    #     print("=====evaluating {}=====".format(args.lang))
    #     device = args.device
    #     # global_att = MultiHeadedAttention(8, 1024 + 768 * 2, 0.1, device)
    #     global_att_1 = MultiHeadedAttention(8, 768, 0, device)
    #     global_att_2 = MultiHeadedAttention(8, 1024, 0, device)
    #     global_att_3 = MultiHeadedAttention(8, 768, 0, device)
    #
    #     global_att_4 = SelfAttention(1024 + 768 + 768)
    #
    #     # global_att.to(device)
    #     global_att_1.to(device)
    #     global_att_2.to(device)
    #     global_att_3.to(device)
    #
    #     global_att_4.to(device)
    #
    #     # query_codet5p_embedding = np.load(args.query_codet5p_path)
    #     # code_codet5p_embedding = np.load(args.code_codet5p_path)
    #
    #     query_cocosoda_embedding = torch.tensor(np.load(args.query_cocosoda_path)).to(device)
    #     code_cocosoda_embedding = torch.tensor(np.load(args.code_cocosoda_path)).to(device)
    #     query_bge_embedding = torch.tensor(np.load(args.query_bge_path)).to(device)
    #     code_bge_embedding = torch.tensor(np.load(args.code_bge_path)).to(device)
    #     query_unixcoder_embedding = torch.tensor(np.load(args.query_unixcoder_path)).to(device)
    #     code_unixcoder_embedding = torch.tensor(np.load(args.code_unixcoder_path)).to(device)
    #
    #     cocosoda_scores = query_cocosoda_embedding @ code_cocosoda_embedding.T
    #     bge_scores = query_bge_embedding @ code_bge_embedding.T
    #     unixcoder_scores = query_unixcoder_embedding @ code_unixcoder_embedding.T
    #     # codet5p_score = query_codet5p_embedding @ code_codet5p_embedding.T
    #
    #     query_cocosoda_embedding = query_cocosoda_embedding.unsqueeze(1)
    #     query_bge_embedding = query_bge_embedding.unsqueeze(1)
    #     query_unixcoder_embedding = query_unixcoder_embedding.unsqueeze(1)
    #
    #     code_cocosoda_embedding = code_cocosoda_embedding.unsqueeze(1)
    #     code_bge_embedding = code_bge_embedding.unsqueeze(1)
    #     code_unixcoder_embedding = code_unixcoder_embedding.unsqueeze(1)
    #
    #     query_1 = query_cocosoda_embedding.squeeze(1) + global_att_1(query_cocosoda_embedding, query_cocosoda_embedding, query_cocosoda_embedding).squeeze(1)
    #     query_2 = query_bge_embedding.squeeze(1) + global_att_2(query_bge_embedding, query_bge_embedding, query_bge_embedding).squeeze(1)
    #     query_3 = query_unixcoder_embedding.squeeze(1) + global_att_3(query_unixcoder_embedding, query_unixcoder_embedding, query_unixcoder_embedding).squeeze(1)
    #
    #     target_1 = code_cocosoda_embedding.squeeze(1) + global_att_1(code_cocosoda_embedding, code_cocosoda_embedding, code_cocosoda_embedding).squeeze(1)
    #     target_2 = code_bge_embedding.squeeze(1) + global_att_2(code_bge_embedding, code_bge_embedding, code_bge_embedding).squeeze(1)
    #     target_3 = code_unixcoder_embedding.squeeze(1) + global_att_3(code_unixcoder_embedding, code_unixcoder_embedding, code_unixcoder_embedding).squeeze(1)
    #
    #     global_query = torch.cat((query_1, query_2, query_3), dim=-1)
    #     global_target = torch.cat((target_1, target_2, target_3), dim=-1)
    #
    #     # global_query = torch.cat((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding), dim=-1).unsqueeze(1)
    #     # global_target = torch.cat((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), dim=-1).unsqueeze(1)
    #
    #     # query = global_query.squeeze(1) + global_att(global_query, global_query, global_query).squeeze(1)
    #     # target = global_target.squeeze(1) + global_att(global_target, global_target, global_target).squeeze(1)
    #     query = global_query.squeeze(1) + global_att_4(global_query, global_query, global_query).squeeze(1)
    #     target = global_target.squeeze(1) + global_att_4(global_target, global_target, global_target).squeeze(1)
    #
    #
    #
    #     # print("query shape:", query.shape)
    #     # print("target shape:", target.shape)
    #
    #     # global_query_1 = np.concatenate(
    #     #     (query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding, query_codet5p_embedding),
    #     #     axis=-1)
    #     # global_target_1 = np.concatenate(
    #     #     (code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding, code_codet5p_embedding), axis=-1)
    #
    #     query2target_scores = query.detach().cpu().numpy() @ target.detach().cpu().numpy().T
    #     # query2target_scores = global_query.detach().cpu().numpy() @ global_target.detach().cpu().numpy().T
    #     evaluate(args, query2target_scores, None, None)
    #     # evaluate(args, cocosoda_scores, bge_scores, unixcoder_scores)
    # elif args.mode == "eval1":
    #     print("=====evaluating {}=====".format(args.lang))
    #
    #     query_cocosoda_embedding = np.load(args.query_cocosoda_path)
    #     code_cocosoda_embedding = np.load(args.code_cocosoda_path)
    #     query_bge_embedding = np.load(args.query_bge_path)
    #     code_bge_embedding = np.load(args.code_bge_path)
    #     query_unixcoder_embedding = np.load(args.query_unixcoder_path)
    #     code_unixcoder_embedding = np.load(args.code_unixcoder_path)
    #
    #     gencode_cocosoda_embedding = np.load(args.gencode_cocosoda_path)
    #     comment_cocosoda_embedding = np.load(args.comment_cocosoda_path)
    #     gencode_bge_embedding = np.load(args.gencode_bge_path)
    #     comment_bge_embedding = np.load(args.comment_bge_path)
    #     gencode_unixcoder_embedding = np.load(args.gencode_unixcoder_path)
    #     comment_unixcoder_embedding = np.load(args.comment_unixcoder_path)
    #
    #     gendes_cocosoda_embedding = np.load(args.gendes_cocosoda_path)
    #     gendes_bge_embedding = np.load(args.gendes_bge_path)
    #     gendes_unixcoder_embedding = np.load(args.gendes_unixcoder_path)
    #
    #     gendes1_cocosoda_embedding = np.load(args.gendes1_cocosoda_path)
    #     gendes1_bge_embedding = np.load(args.gendes1_bge_path)
    #     gendes1_unixcoder_embedding = np.load(args.gendes1_unixcoder_path)
    #
    #     comment1_cocosoda_embedding = np.load(args.comment1_cocosoda_path)
    #     comment1_bge_embedding = np.load(args.comment1_bge_path)
    #     comment1_unixcoder_embedding = np.load(args.comment1_unixcoder_path)
    #
    #     exquery_cocosoda_embedding = np.load(args.exquery_cocosoda_path)
    #     exquery_bge_embedding = np.load(args.exquery_bge_path)
    #     exquery_unixcoder_embedding = np.load(args.exquery_unixcoder_path)
    #
    #     gpt_exquery_cocosoda_embedding = np.load(args.gpt_exquery_cocosoda_path)
    #     gpt_exquery_bge_embedding = np.load(args.gpt_exquery_bge_path)
    #     gpt_exquery_unixcoder_embedding = np.load(args.gpt_exquery_unixcoder_path)
    #
    #     ds_exquery_cocosoda_embedding = np.load(args.ds_exquery_cocosoda_path)
    #     ds_exquery_bge_embedding = np.load(args.ds_exquery_bge_path)
    #     ds_exquery_unixcoder_embedding = np.load(args.ds_exquery_unixcoder_path)
    #
    #     qwen_exquery_cocosoda_embedding = np.load(args.qwen_exquery_cocosoda_path)
    #     qwen_exquery_bge_embedding = np.load(args.qwen_exquery_bge_path)
    #     qwen_exquery_unixcoder_embedding = np.load(args.qwen_exquery_unixcoder_path)
    #
    #     i = 1000
    #     if args.lang == 'cosqa':
    #         i = 500
    #
    #     qwen_exquery_cocosoda_embedding_1 = qwen_exquery_cocosoda_embedding[i * 0:i * 1]
    #     qwen_exquery_bge_embedding_1 = qwen_exquery_bge_embedding[i * 0:i * 1]
    #     qwen_exquery_unixcoder_embedding_1 = qwen_exquery_unixcoder_embedding[i * 0:i * 1]
    #
    #     qwen_exquery_cocosoda_embedding_5 = qwen_exquery_cocosoda_embedding[i * 1:i * 2]
    #     qwen_exquery_bge_embedding_5 = qwen_exquery_bge_embedding[i * 1:i * 2]
    #     qwen_exquery_unixcoder_embedding_5 = qwen_exquery_unixcoder_embedding[i * 1:i * 2]
    #
    #     qwen_exquery_cocosoda_embedding_7 = qwen_exquery_cocosoda_embedding[i * 2:i * 3]
    #     qwen_exquery_bge_embedding_7 = qwen_exquery_bge_embedding[i * 2:i * 3]
    #     qwen_exquery_unixcoder_embedding_7 = qwen_exquery_unixcoder_embedding[i * 2:i * 3]
    #
    #     qwen_exquery_cocosoda_embedding_9 = qwen_exquery_cocosoda_embedding[i * 3:i * 4]
    #     qwen_exquery_bge_embedding_9 = qwen_exquery_bge_embedding[i * 3:i * 4]
    #     qwen_exquery_unixcoder_embedding_9 = qwen_exquery_unixcoder_embedding[i * 3:i * 4]
    #
    #     cocosoda_scores = query_cocosoda_embedding @ code_cocosoda_embedding.T
    #     bge_scores = query_bge_embedding @ code_bge_embedding.T
    #     unixcoder_scores = query_unixcoder_embedding @ code_unixcoder_embedding.T
    #
    #     # global_query_1 = np.concatenate((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding), axis=-1)
    #     # global_target_1 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_1 = np.concatenate((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding), axis=-1)
    #     global_target_1 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     # global_query_2 = np.concatenate((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding), axis=-1)
    #     # global_target_2 = np.concatenate((comment_cocosoda_embedding, comment_bge_embedding, comment_unixcoder_embedding), axis=-1)
    #
    #     global_query_2 = np.concatenate((query_bge_embedding, query_bge_embedding, query_bge_embedding), axis=-1)
    #     global_target_2 = np.concatenate((comment_bge_embedding, comment_bge_embedding, comment_bge_embedding), axis=-1)
    #
    #     # global_query_3 = np.concatenate((gencode_cocosoda_embedding, gencode_bge_embedding, gencode_unixcoder_embedding), axis=-1)
    #     # global_target_3 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_3 = np.concatenate((gencode_unixcoder_embedding, gencode_unixcoder_embedding, gencode_unixcoder_embedding), axis=-1)
    #     global_target_3 = np.concatenate((code_unixcoder_embedding, code_unixcoder_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     # global_query_4 = np.concatenate((gendes_cocosoda_embedding, gendes_bge_embedding, gendes_unixcoder_embedding), axis=-1)
    #     # global_target_4 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_4 = np.concatenate((gendes_cocosoda_embedding, gendes_cocosoda_embedding, gendes_cocosoda_embedding), axis=-1)
    #     global_target_4 = np.concatenate((code_cocosoda_embedding, code_cocosoda_embedding, code_cocosoda_embedding), axis=-1)
    #
    #     global_query_5 = np.concatenate((gendes_cocosoda_embedding, gendes_bge_embedding, gendes_unixcoder_embedding), axis=-1)
    #     global_target_5 = np.concatenate((comment_cocosoda_embedding, comment_bge_embedding, comment_unixcoder_embedding), axis=-1)
    #
    #     # global_query_5 = np.concatenate((gendes_bge_embedding, gendes_bge_embedding, gendes_bge_embedding), axis=-1)
    #     # global_target_5 = np.concatenate((code_bge_embedding, code_bge_embedding, code_bge_embedding), axis=-1)
    #
    #     global_query_6 = np.concatenate((gendes1_cocosoda_embedding, gendes1_cocosoda_embedding, gendes1_cocosoda_embedding), axis=-1)
    #     global_target_6 = np.concatenate((code_cocosoda_embedding, code_cocosoda_embedding, code_cocosoda_embedding), axis=-1)
    #
    #     global_query_7 = np.concatenate((gendes1_cocosoda_embedding, gendes1_bge_embedding, gendes1_unixcoder_embedding), axis=-1)
    #     global_target_7 = np.concatenate((comment_cocosoda_embedding, code_bge_embedding, comment_unixcoder_embedding), axis=-1)
    #
    #     global_query_8 = np.concatenate((query_bge_embedding, query_bge_embedding, query_bge_embedding), axis=-1)
    #     global_target_8 = np.concatenate((comment1_bge_embedding, comment1_bge_embedding, comment1_bge_embedding), axis=-1)
    #
    #     # global_query_9 = np.concatenate((exquery_cocosoda_embedding, exquery_bge_embedding, exquery_unixcoder_embedding), axis=-1)
    #     # global_target_9 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_9 = np.concatenate((exquery_cocosoda_embedding, exquery_bge_embedding, exquery_unixcoder_embedding), axis=-1)
    #     global_target_9 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_10 = np.concatenate((exquery_cocosoda_embedding, exquery_bge_embedding, exquery_unixcoder_embedding), axis=-1)
    #     global_target_10 = np.concatenate((comment_cocosoda_embedding, comment_bge_embedding, comment_unixcoder_embedding), axis=-1)
    #
    #     global_query_11 = np.concatenate((gpt_exquery_cocosoda_embedding, gpt_exquery_bge_embedding, gpt_exquery_unixcoder_embedding), axis=-1)
    #     global_target_11 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_12 = np.concatenate((ds_exquery_cocosoda_embedding, ds_exquery_bge_embedding, ds_exquery_unixcoder_embedding), axis=-1)
    #     global_target_12 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_9_1 = np.concatenate((qwen_exquery_cocosoda_embedding_1, qwen_exquery_bge_embedding_1, qwen_exquery_unixcoder_embedding_1), axis=-1)
    #     global_target_9_1 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_9_5 = np.concatenate((qwen_exquery_cocosoda_embedding_5, qwen_exquery_bge_embedding_5, qwen_exquery_unixcoder_embedding_5), axis=-1)
    #     global_target_9_5 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_9_7 = np.concatenate((qwen_exquery_cocosoda_embedding_7, qwen_exquery_bge_embedding_7, qwen_exquery_unixcoder_embedding_7), axis=-1)
    #     global_target_9_7 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #     global_query_9_9 = np.concatenate((qwen_exquery_cocosoda_embedding_9, qwen_exquery_bge_embedding_9, qwen_exquery_unixcoder_embedding_9), axis=-1)
    #     global_target_9_9 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)
    #
    #
    #
    #     # global_query_1 = np.concatenate(
    #     #     (query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding, query_codet5p_embedding),
    #     #     axis=-1)
    #     # global_target_1 = np.concatenate(
    #     #     (code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding, code_codet5p_embedding), axis=-1)
    #
    #     scores_1 = global_query_1 @ global_target_1.T
    #     scores_2 = global_query_2 @ global_target_2.T
    #     scores_3 = global_query_3 @ global_target_3.T
    #     scores_4 = global_query_4 @ global_target_4.T
    #     scores_5 = global_query_5 @ global_target_5.T
    #     scores_6 = global_query_6 @ global_target_6.T
    #     scores_7 = global_query_7 @ global_target_7.T
    #     scores_8 = global_query_8 @ global_target_8.T
    #     scores_9 = global_query_9 @ global_target_9.T
    #     scores_10 = global_query_10 @ global_target_10.T
    #     scores_11 = global_query_11 @ global_target_11.T
    #     scores_12 = global_query_12 @ global_target_12.T
    #     scores_9_1 = global_query_9_1 @ global_target_9_1.T
    #     scores_9_5 = global_query_9_5 @ global_target_9_5.T
    #     scores_9_7 = global_query_9_7 @ global_target_9_7.T
    #     scores_9_9 = global_query_9_9 @ global_target_9_9.T
    #
    #     scores = scores_1 * 0.65 + scores_2 * 0.25 + scores_3 * 0.00 + scores_4 * 0.00 + scores_5 * 0.00 + scores_6 * 0.10 + scores_7 * 0.0
    #     # scores = scores + scores_9 * 0.05 + scores_11 * 0.05 + scores_12 * 0.05
    #
    #     # get_rank(args, scores)
    #     # get_exinfomation(args, scores)
    #
    #     # evaluate(args, scores, None, None)
    #     # args.mode = 'traverse'
    #     # evaluate(args, scores_1, scores_2, scores_9)
    #
    #     res = scores_1 * 0.65 + scores_2 * 0.25 + scores_9 * 0.1
    #     evaluate(args, res, None, None)
    elif args.mode == "eval1":
        print("=====evaluating {}=====".format(args.lang))

        query_cocosoda_embedding = np.load(args.query_cocosoda_path)
        code_cocosoda_embedding = np.load(args.code_cocosoda_path)
        query_bge_embedding = np.load(args.query_bge_path)
        code_bge_embedding = np.load(args.code_bge_path)
        query_unixcoder_embedding = np.load(args.query_unixcoder_path)
        code_unixcoder_embedding = np.load(args.code_unixcoder_path)

        comment_cocosoda_embedding = np.load(args.comment_cocosoda_path)
        comment_bge_embedding = np.load(args.comment_bge_path)
        comment_unixcoder_embedding = np.load(args.comment_unixcoder_path)

        gendes_cocosoda_embedding = np.load(args.gendes_cocosoda_path)
        gendes_bge_embedding = np.load(args.gendes_bge_path)
        gendes_unixcoder_embedding = np.load(args.gendes_unixcoder_path)

        exquery_cocosoda_embedding = np.load(args.exquery_cocosoda_path)
        exquery_bge_embedding = np.load(args.exquery_bge_path)
        exquery_unixcoder_embedding = np.load(args.exquery_unixcoder_path)


        global_query_1 = np.concatenate((query_cocosoda_embedding, query_bge_embedding, query_unixcoder_embedding),
                                        axis=-1)
        global_target_1 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding),
                                         axis=-1)

        global_query_2 = np.concatenate((query_bge_embedding, query_bge_embedding, query_bge_embedding), axis=-1)
        global_target_2 = np.concatenate((comment_bge_embedding, comment_bge_embedding, comment_bge_embedding), axis=-1)

        global_query_4 = np.concatenate((gendes_cocosoda_embedding, gendes_bge_embedding, gendes_unixcoder_embedding), axis=-1)
        global_target_4 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)

        global_query_5 = np.concatenate((exquery_cocosoda_embedding, exquery_bge_embedding, exquery_unixcoder_embedding), axis=-1)
        global_target_5 = np.concatenate((code_cocosoda_embedding, code_bge_embedding, code_unixcoder_embedding), axis=-1)

        scores_1 = global_query_1 @ global_target_1.T
        scores_2 = global_query_2 @ global_target_2.T
        scores_4 = global_query_4 @ global_target_4.T
        scores_5 = global_query_5 @ global_target_5.T

        # scores = scores_1 * 0.4 + scores_2 * 0.25 + scores_4 * 0.00 + scores_5 * 0.35

        # get_exinfomation(args, scores)

        # evaluate(args, scores, None, None)
        args.mode = 'traverse'
        evaluate(args, scores_1, scores_2, scores_5)

