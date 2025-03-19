import itertools
import os
import shutil
import logging
from util.s3util import S3Util
from util.bedrock_utils import KnowledgeBaseUtils
from config.config import get_config
from decimal import Decimal
from util.pdf_utils import extract_text_from_pdf_pymudf
from app.price_calculator import estimate_embedding_model_bedrock_price,estimate_retrieval_model_bedrock_price,estimate_opensearch_price,estimate_sagemaker_price, estimate_fargate_price, estimate_effective_kb_tokens, estimate_times
from .dependencies.database import get_execution_db
from constants.validation_status import ValidationStatus
from functools import lru_cache
import time 

configs = get_config()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


S3_BUCKET = configs.s3_bucket
bedrock_price_df = S3Util().read_csv_from_s3(configs.bedrock_limit_csv_path, S3_BUCKET, as_dataframe=True)

def is_valid_combination(config, data):
    # Define your rules here
    regions = ["us-east-1", "us-west-2"]
    if config['bedrock_knowledge_base'] or not config['knowledge_base']:
        return True
    if config["region"] not in regions:
        return False
    if config["n_shot_prompts"] > 0 and data["n_shot_prompt_guide"] is None:
        return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-image-v1":
        if config['vector_dimension'] != 1024 and config['vector_dimension'] != 384 and config['vector_dimension'] != 256:
            return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-text-v2:0":
        if config['vector_dimension'] != 1024 and config['vector_dimension'] != 512 and config['vector_dimension'] != 256:
            return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-text-v1":
        if config['vector_dimension'] != 1536:
            return False
    if (config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "cohere.embed-english-v3") or (
            config['embedding']["service"] == "bedrock" and config["embedding"][
        "model"] == "cohere.embed-multilingual-v3"):
        if config['vector_dimension'] != 1024:
            return False
    if (config['embedding']["service"] == "sagemaker" and config["embedding"]["model"] == "huggingface-sentencesimilarity-bge-large-en-v1-5") or (
            config['embedding']["service"] == "sagemaker" and config["embedding"]["model"] == "huggingface-sentencesimilarity-bge-m3"):
        if config['vector_dimension'] != 1024:
            return False
    if (config['embedding']["service"] == "sagemaker" and config["embedding"]["model"] == "huggingface-textembedding-gte-qwen2-7b-instruct"):
        if config['vector_dimension'] != 3584:
            return False
    valid_values = {Decimal('0.5'), Decimal('0.3'), Decimal('0.7'), Decimal('0'), Decimal('0.1')}
    if config['temp_retrieval_llm'] not in valid_values:
        return False
    if config['knn_num'] != 3 and config['knn_num'] != 5 and config['knn_num'] != 10 and config['knn_num'] != 15:
        return False
    if config.get('chunking_strategy', None) == "hierarchical":
        # child chunk size should be less than parent chunk size
        if config.get("hierarchical_child_chunk_size") > config.get("hierarchical_parent_chunk_size"):
            return False
    return True


def parse_dynamodb(item):
    if isinstance(item, dict):
        if "M" in item:
            return {k: parse_dynamodb(v) for k, v in item["M"].items()}
        elif "L" in item:
            return [parse_dynamodb(v) for v in item["L"]]
        elif "S" in item:
            return item["S"]
        elif "N" in item:
            return float(item["N"]) if "." in item["N"] else int(item["N"])
    return item


def flatten_parameters(stage_data):
    flattened = {}
    for key, value in stage_data.items():
        if isinstance(value, list):
            flattened[key] = value
        else:
            flattened[key] = [value]
    return flattened


def restructure_combination(combination):
    # Define the expected structure
    result = {}

    # Prestep keys
    prestep_keys = ['region', 'gt_data', 'kb_data']
    for key in prestep_keys:
        if key in combination:
            result[key] = combination[key]

    # Indexing keys
    indexing_keys = [
        'chunking_strategy',
        'vector_dimension',
        'chunk_size',
        'chunk_overlap',
        'indexing_algorithm'
    ]
    for key in indexing_keys:
        if key in combination:
            result[key] = combination[key]

    # Embedding (nested)
    if 'service' in combination and 'model' in combination:
        result['embedding'] = {
            'service': combination['service'],
            'model': combination['model']
        }

    # Retrieval keys
    retrieval_keys = [
        'n_shot_prompts',
        'knn_num',
        'temp_retrieval_llm'
    ]
    for key in retrieval_keys:
        if key in combination:
            result[key] = combination[key]

    # Retrieval (nested)
    if 'retrieval_service' in combination and 'retrieval_model' in combination:
        result['retrieval'] = {
            'service': combination['retrieval_service'],
            'model': combination['retrieval_model']
        }

    return result

@lru_cache(maxsize=100)
def count_characters_in_file(file_path):
    file_path = S3Util().download_directory_from_s3(file_path)
    character_counts = 0
    try:
        for file in os.listdir(file_path):
            full_file = os.path.join(file_path, file)
            if file.endswith('.txt'):
                with open(full_file, 'r') as file:
                    content = file.read()
                    character_counts+= len(content)
            elif file.endswith('.pdf'):
                with open(full_file, 'rb') as file:
                    text_data = extract_text_from_pdf_pymudf(file)
                    character_counts += len(text_data)
            else:
                character_counts+= 1
        return character_counts
    except FileNotFoundError:
        return "File not found."
    finally:
        # Clean up the downloaded files and directory
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


def read_gt_data(file_path):
    """
    Reads a GT-data file and outputs the number of JSON entries and the total number of characters in the file.

    Parameters:
    file_path (str): The path to the GT-data file.

    Returns:
    tuple: A tuple containing the number of JSON entries and the total number of characters in the file.
    """
    try:
        json_entries, data = S3Util().read_json_and_content_from_s3(file_path)
        num_entries = len(json_entries)
        total_characters = len(data)
        return num_entries, total_characters
    except Exception as e:
        logger.error(f"Error reading the GT-data file: {e}")
        return None, None

def remove_invalid_combinations_keys(combinations):
    """
    Adjusts combinations to set irrelevant keys to None based on the value of 'chunking_strategy'.

    Parameters:
        combinations: List of combinations.

    Returns:
        list of dict: Updated combinations with irrelevant keys set to None.
    """
    for combination in combinations:
        if combination.get("chunking_strategy", None) == "fixed":
            combination["hierarchical_chunk_overlap_percentage"] = None
            combination["hierarchical_parent_chunk_size"] = None
            combination["hierarchical_child_chunk_size"] = None

        elif combination.get("chunking_strategy", None) == "hierarchical":
            combination["chunk_overlap"] = None
            combination["chunk_size"] = None

    return combinations


def unpack_knowledebases(combinations):
    for combination in combinations:
        kb_data = combination.get('kb_data')
        if isinstance(kb_data, dict):
            combination["kb_name"] = kb_data.get("name", "")
            combination["kb_data"] = kb_data.get("id", "")
            
        elif isinstance(kb_data, str):
            combination["kb_data"] = kb_data
            combination["kb_name"] = ""
            
    return combinations


def unpack_guardrails(combinations):
    for combination in combinations:
        combination["enable_guardrails"] = True if "guardrails" in combination else False
        combination["guardrail_id"] = combination.get("guardrails", {}).get("guardrails_id", "")
        combination["guardrail_name"] = combination.get("guardrails", {}).get("name", "")
        combination["guardrail_version"] = combination.get("guardrails", {}).get("guardrail_version", "")
        combination["enable_prompt_guardrails"] = combination.get("guardrails", {}).get("enable_prompt_guardrails", False)
        combination["enable_context_guardrails"] = combination.get("guardrails", {}).get("enable_context_guardrails", False)
        combination["enable_response_guardrails"] = combination.get("guardrails", {}).get("enable_response_guardrails", False)

        if "guardrails" in combination:
            del combination["guardrails"]

    return combinations

def add_kb_info(parameters_all):
    if 'kb_data' in parameters_all:
        if parameters_all['bedrock_knowledge_base']:
            kb_ids = parameters_all['kb_data']
            kb_data = []
            for kb_id in kb_ids:
                kb_name = KnowledgeBaseUtils(parameters_all['region']).get_kb_name(kb_id)
                kb_data.append({
                    "id": kb_id,
                    "name": kb_name
                })
            parameters_all['kb_data'] = kb_data 
    return parameters_all
    
def generate_all_combinations(data):
    
    # Parse the DynamoDB-style JSON
    parsed_data = {k: parse_dynamodb(v) for k, v in data.items()}

    parameters_all = parsed_data["prestep"]
    parameters_all.update(parsed_data["indexing"])
    parameters_all.update(parsed_data["retrieval"])
    if "guardrails" in parsed_data and parsed_data["guardrails"]:
        parameters_all.update({"guardrails": parsed_data["guardrails"]})
    parameters_all.update(parsed_data["evaluation"])
    parameters_all = add_kb_info(parameters_all)
    parameters_all = {key: value if isinstance(value, list) else [value] for key, value in parameters_all.items()}
    # Convert single values to lists and replace empty values with [0]
    parameters_all = {key: value if isinstance(value, list) else [value] for key, value in parameters_all.items()}
    parameters_all = {key: value if value else [0] for key, value in parameters_all.items()}
    
    keys = parameters_all.keys()
    combinations = [dict(zip(keys, values)) for values in itertools.product(*parameters_all.values())]
    combinations = remove_invalid_combinations_keys(combinations)
    combinations = unpack_guardrails(combinations)
    combinations = unpack_knowledebases(combinations)
    
    gt_data = parameters_all["gt_data"][0]
    [num_prompts, num_chars] = read_gt_data(gt_data)

    avg_prompt_length = round(num_chars / num_prompts / 4)
    if parameters_all["bedrock_knowledge_base"][0] or not parameters_all['knowledge_base'][0]:
        num_tokens_kb_data = 0
    else:
        num_tokens_kb_data = count_characters_in_file(parameters_all["kb_data"][0]) / 4
        
    configurations = []
    valid_configurations = []
    MAX_VALID_EXPERIMENTS = 1000
    experiment_count = 0

    for combination in combinations:
        # Generate a unique GUID
        
        if experiment_count >= MAX_VALID_EXPERIMENTS:
            break  # Exit the loop when the limit is reached
       
        configuration = {
            **combination
        }
        configurations.append(configuration)
        if is_valid_combination(configuration, data):
            
            configuration = {
                **{k: v for k, v in configuration.items() if k not in ["embedding", "retrieval", "gt_data", "evaluation"]},
                "embedding_service": configuration["embedding"]["service"],
                "embedding_model": configuration["embedding"]["model"],
                "retrieval_service": configuration["retrieval"]["service"],
                "retrieval_model": configuration["retrieval"]["model"],
                "eval_service": configuration["evaluation"]["service"],
                "eval_embedding_model": configuration["evaluation"]["embedding_model"],
                "eval_retrieval_model": configuration["evaluation"]["retrieval_model"],
                }
            valid_configurations.append(configuration)
            experiment_count += 1

    if len(valid_configurations) > 0:
        for configuration in valid_configurations:
            configuration["is_opensearch"] = True if configs.opensearch_host else False
            #TODO: Organize the pricing code, break into static methods
            configuration["directional_pricing"] = 0
            configuration["indexing_cost_estimate"] = 0 
            configuration["retrieval_cost_estimate"] = 0
            configuration["inferencing_cost_estimate"] = 0 
            configuration["eval_cost_estimate"] = 0

            # kb data tokens would be zero if it is Bedrock knowledge bases
            effective_num_tokens_kb_data = 0 if configuration["bedrock_knowledge_base"] or not configuration["knowledge_base"] else estimate_effective_kb_tokens(configuration, num_tokens_kb_data)

            indexing_time, retrieval_time, eval_time = estimate_times(effective_num_tokens_kb_data, num_prompts, configuration)

            # Bedrock knowledge bases price not supported at the moment
            if configuration["bedrock_knowledge_base"] or not configuration["knowledge_base"]:
                configuration["indexing_cost_estimate"] = 0
            elif configuration['embedding_service'] == "bedrock" :
                embedding_price = estimate_embedding_model_bedrock_price(bedrock_price_df, configuration, num_tokens_kb_data)
                configuration["indexing_cost_estimate"] += embedding_price
            else:
                configuration["indexing_cost_estimate"] += estimate_sagemaker_price(indexing_time)

            #Calculate the inferencing price - doesn't include OpenSearch pricing
            if configuration["retrieval_service"] == "bedrock":
                inferencing_price = estimate_retrieval_model_bedrock_price(bedrock_price_df, configuration, avg_prompt_length, num_prompts)
                configuration["inferencing_cost_estimate"] += inferencing_price
            else:
                configuration["inferencing_cost_estimate"] += estimate_sagemaker_price(retrieval_time)

            # evaluation price for ragas not added at the moment considering 
            # tokens information is not being returned
            # Adding sagemaker endpoint cost as it would be still running for 
            # the duration of the experiment
            if configuration['embedding_service'] == "sagemaker":
                configuration["eval_cost_estimate"] += estimate_sagemaker_price(eval_time)
            if configuration["retrieval_service"] == "sagemaker":
                configuration["eval_cost_estimate"] += estimate_sagemaker_price(eval_time)
            
            # adding fargate container costs
            if not configuration["bedrock_knowledge_base"]:
                configuration["indexing_cost_estimate"] += estimate_fargate_price(indexing_time)
            configuration["retrieval_cost_estimate"] += estimate_fargate_price(retrieval_time)
            configuration["eval_cost_estimate"] += estimate_fargate_price(eval_time)

            # add opensearch provisioned costs
            if not configuration["bedrock_knowledge_base"] or configuration["is_opensearch"]:
                configuration["indexing_cost_estimate"] += estimate_opensearch_price(indexing_time)
                configuration["retrieval_cost_estimate"] += estimate_opensearch_price(retrieval_time)
                configuration["eval_cost_estimate"] += estimate_opensearch_price(eval_time)

            configuration["directional_pricing"] = configuration["indexing_cost_estimate"] + configuration["retrieval_cost_estimate"] + configuration["inferencing_cost_estimate"] + configuration["eval_cost_estimate"]
            configuration["directional_pricing"] +=configuration["directional_pricing"]*0.05 #extra
            configuration["directional_pricing"] = round(configuration["directional_pricing"],2)    

    return valid_configurations

def generate_all_combinations_in_background(execution_id: str, execution_config_data):
    """       
    Generate all possible valid experiment configurations for a given execution and stores result in S3.
    Progress status and result s3 url is stored in execution db under valid execution_id
    """

    try:
        # update status of execution_id to InProgress
        get_execution_db().update_item(
            key={"id": execution_id}, 
            update_expression="SET validation_status = :status_value", 
            expression_values={":status_value": ValidationStatus.INPROGRESS.value}
        )

        combinations = generate_all_combinations(execution_config_data)
        deserialized_combinations_data = []
        for combination in combinations:
            updated_combination = {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in combination.items()
            }
            deserialized_combinations_data.append(updated_combination)

        S3Util().write_json_to_s3(f"experiment_combination/{execution_id}.json", S3_BUCKET, deserialized_combinations_data)

        # update status of execution id to Completed
        get_execution_db().update_item(
            key={"id": execution_id}, 
            update_expression="SET validation_status = :status_value", 
            expression_values={":status_value": ValidationStatus.COMPLETED.value}
        ) 
    except Exception as e:
        # update status of execution id to failed
        logger.error(f"Error in generate_all_combinations_in_background: {e}")
        get_execution_db().update_item(
            key={"id": execution_id}, 
            update_expression="SET validation_status = :status_value", 
            expression_values={":status_value": ValidationStatus.FAILED.value}
        )