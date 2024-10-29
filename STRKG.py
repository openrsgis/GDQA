# -*- coding: utf-8 -*-
# @Time    : 2024/3/24 20:56
# @Author  : LiHao


from tqdm import tqdm
import argparse
# from utils import *
# from freebase_func import *
from KG_func import *
import random
# from client import *
import os
import time
import os
import logging
import sys
import json
import datetime
import maintainInferencepath
import KG_func
import yaml



def storeHistoryNode(historyNode, currentNode_id, candidatesNodeList_id):
    for i in range(len(candidatesNodeList_id)):
        historyNode[candidatesNodeList_id[i]] = currentNode_id
    return historyNode

def filterCandidate(historyNode, current_node_id, candidatesNodeList_id):
    current_history_value = historyNode.get(current_node_id)
    filtered_candidates = [node_id for node_id in candidatesNodeList_id if node_id != current_history_value]

    return filtered_candidates


def carryOutMain_New(question, topic_entity, depth, directory_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_attri_need_prune", type=str, default=config.get('entity_attri_need_prune'))
    # tongyi的 Key在 KG_func.py
    parser.add_argument("--LLM_type", type=str, default=config.get('LLM'))  # gpt-3.5-turbo   gpt-4
    parser.add_argument("--width", type=int, default=int(config.get('width')))
    parser.add_argument("--depth", type=int, default=depth)
    parser.add_argument("--relation_one_direct", type=bool, default=True)
    parser.add_argument("--IfPureHistoryNode", type=bool, default=True)
    parser.add_argument("--ifInferencePath", type=bool, default=True)
    parser.add_argument("--collectedInstructionData", type=bool, default=bool(config.get('collectedInstructionData')))
    parser.add_argument("--NodeNeedSearchAttri", type=list, default=config.get('NodeNeedSearchAttri'))  # gpt-3.5-turbo   gpt-4
    parser.add_argument("--dataset", type=str, default="Neo4j")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature_exploration", type=float, default=0.1)
    parser.add_argument("--temperature_reasoning", type=float, default=0)
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True)
    parser.add_argument("--num_retain_entity", type=int, default=5)
    parser.add_argument("--prune_tools", type=str, default="llm")
    parser.add_argument("--relation_prune", type=bool, default=True)
    parser.add_argument("--opeani_api_keys", type=str, default="xxx")
    args = parser.parse_args()

    ##
    rlc = run_llm_class()

    directory_name = config.get('Instructionlog_path')
    if args.collectedInstructionData and directory_name is not None:  # 只有在batch实验中才有directory_name
        logging.basicConfig(filename=directory_name, level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        pass
    # datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    print("所使用的大模型", args.LLM_type)
    start_time = time.time()  # 记录开始时间

    question_t = question
    print(question_t)
    topic_entity = topic_entity

    cluster_chain_of_entities = []  
    entities_id = list(topic_entity.keys())  
    historyNode = {}

    pre_relations = []  
    pre_heads = [-1] * len(topic_entity)  
    flag_printed = False
    Inferencepath = 'None, 由于未进入到Depth探索'
    results = 'None, 由于未进入到Depth探索'

    select_attri = attri_select_TF(question, rlc, args)
    print(select_attri)
    pre_select_relations = rela_select_TF(question, rlc, args)  
    print(pre_select_relations)

    for depth in range(1, args.depth + 1):
        current_entity_relations_list = []  
        i = 0
        topic_entity = {entity_id: id2entity_name_or_type(entity_id) for entity_id in entities_id}
        for entity_id in topic_entity:
            if entity_id != "[FINISH_ID]":
                if args.entity_attri_need_prune in entity_label_search(entity_id):
                    retrieve_attri = attri_search_from_select(entity_id, topic_entity[entity_id], question_t, select_attri, rlc, args)
                    current_entity_relations_list.extend(retrieve_attri)
                    # print(retrieve_attri)

                retrieve_relations_with_scores = relation_search_select(entity_id, topic_entity[entity_id], pre_select_relations,
                                                                       pre_heads[i], question_t, rlc,
                                                                       args)  # best entity triplet, entitiy_id


                current_entity_relations_list.extend(retrieve_relations_with_scores)
            i += 1

    #
        total_candidates = []  
        total_scores = []  
        total_relations = []
        total_entities_id = []  
        total_topic_entities = []
        total_head = []

        for entity in current_entity_relations_list:
            head = entity.get('head', None)
            if head is not None:  
                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                    if args.IfPureHistoryNode:
                        entity_candidates_id = filterCandidate(historyNode, entity['entity'], entity_candidates_id)
                        historyNode = storeHistoryNode(historyNode, entity['entity'], entity_candidates_id)

                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                    if args.IfPureHistoryNode:
                        entity_candidates_id = filterCandidate(historyNode, entity['entity'], entity_candidates_id)
                        historyNode = storeHistoryNode(historyNode, entity['entity'], entity_candidates_id)

                if args.prune_tools == "llm":  
                    if len(entity_candidates_id) >= 20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)  # 实体搜索期间保留的实体个数
                if len(entity_candidates_id) == 0:
                    continue
                if entity['score']!=0:
                    scores, entity_candidates, entity_candidates_id = entity_score(question_t, entity_candidates_id, entity['score'], entity['relation'], rlc, args)
                else:
                    continue
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
                    entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                    total_relations, total_entities_id, total_topic_entities, total_head)


            else:

                entity_current_for_attri = id2entity_name_or_type(entity['entity'])
                attribution_for_attri = entity['attribution']
                attribution_for_value = entity['value']
                cluster_chain_of_entities.append([[(entity_current_for_attri, attribution_for_attri, attribution_for_value)]])


        if len(total_candidates) == 0:
            half_stop(question_t, cluster_chain_of_entities, depth, args)
            flag_printed = True
            break

        flag, chain_of_entities, chain_of_entities_head, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id,
                                                                                      total_relations,
                                                                                      total_candidates,
                                                                                      total_topic_entities,
                                                                                      total_head, total_scores,
                                                                                      args)
        cluster_chain_of_entities.append(chain_of_entities_head)
        Inferencepath = maintainInferencepath.entranceMaintainInferencePath(cluster_chain_of_entities)
        assert Inferencepath is not None, "Inferencepath create error" # 如果推理路径生成有误，直接报错
        if flag:
            if args.ifInferencePath: 
                stop, results = reasoning_inferencePath(question_t, Inferencepath, rlc, args)
                print("在depth %d.阶段的推理路径如下" % depth, Inferencepath)
            else:  
                stop, results = reasoning(question_t, cluster_chain_of_entities, rlc, args)
                print("在depth %d.阶段的实体链-路径如下" % depth, cluster_chain_of_entities)
                Inferencepath='None, 因为使用了ToG的实体链'



            if stop:
                print("在depth %d.阶段存在有用的信息" % depth)
                print(results)
                # save_2_jsonl(question_t, results, cluster_chain_of_entities, file_name=args.dataset)
                flag_printed = True
                # break
            else:
                print("depth %d not find the answer." % depth)
                flag_finish, entities_id = if_finish_list(entities_id)
                if flag_finish:
                    half_stop(question_t, cluster_chain_of_entities, depth, args)
                    flag_printed = True
                else:
                    topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                    continue
        else:
            half_stop(question_t, cluster_chain_of_entities, depth, args)
            flag_printed = True
            Inferencepath='None'
    end_time = time.time()  # 记录结束时间
    run_time = end_time - start_time  # 计算运行时间
    print(f"程序运行时间：{run_time}秒")

    if not flag_printed:

    return rlc.total_tokens, run_time, results, Inferencepath






if __name__ == '__main__':

    # 获取当前时间
    now = datetime.datetime.now()
    print("当前时间：", now)
    formatted_now = now.strftime("%m-%d-%H-%M")   # print("格式化的当前时间：", formatted_now)

    ## ------------------------- 导入config yaml文件
    config_file_name = 'config_v1'     # config_v1
    config_file_path = '../config/'+config_file_name+'.yaml'
    # 读取YAML文件
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    print(config)

    LLM_used = config.get('LLM')           
    cycleNum = config.get('cycleNum')      
    benchmarkName = config.get('benchmark_path')
    # print(config.get('collectedInstructionData'))
    # print(type(config.get('Instructionlog_path')))
    benchmark_path = "../data/"+benchmarkName  
    InstructionlogPath = config.get('Instructionlog_path') 

    directory_name = f"./result/{LLM_used}_{config_file_name}_{benchmarkName[:-5]}_{formatted_now}"
    os.makedirs(directory_name, exist_ok=True)  # 如果目录不存在，则创建它

    add_benchmark_path = directory_name +'/add_' + config.get('benchmark_path')
    with open(add_benchmark_path, 'w', encoding='utf-8') as new_file:
        json.dump([], new_file, ensure_ascii=False, indent=4)

    with open(benchmark_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for index, item in enumerate(data):
            itemID = item['ID']
            for i in range(1, cycleNum+1): 
                depth = int(item.get("Depth"))              
                question = item.get("Question")       
                initialNode = item.get("initialNode")      
                filename = os.path.join(directory_name, f'output_{index}_{i}.txt')
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        try:
                            total_tokens, run_time, results, Inferencepath = carryOutMain_New(question, initialNode, depth, InstructionlogPath)
                            results_list = [line.strip() for line in results.strip().splitlines()]
                            Inferencepath_list = [line.strip() for line in Inferencepath.strip().splitlines()]
                            item['ID'] = itemID + '_c_' + str(i)
                            item['settingList']['LLM'] = LLM_used
                            item['performance']['TimeCost'] = run_time
                            item['performance']['TokenCost'] = total_tokens
                            item['Output']['Inferencepath'] = Inferencepath_list
                            item['Output']['LLMOutput'] = results_list

                            with open(add_benchmark_path, 'r', encoding='utf-8') as new_file:
                                add_benchmark_data = json.load(new_file)
                            add_benchmark_data.append(item)
                            with open(add_benchmark_path, 'w', encoding='utf-8') as new_file:
                                json.dump(add_benchmark_data, new_file, ensure_ascii=False, indent=4)
                        except Exception as e:
                            print(f"发生错误: {e}")
                        #恢复标准输出
                        # sys.stdout = original_stdout
                except IOError as e:
                    print(f"文件操作出错: {e}")
                print(f"已将输出保存到 {filename}")
                # 休眠20秒
                time.sleep(10)
