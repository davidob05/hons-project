from hons.config import Config
from hons.registry import build_stages
from pathlib import Path
import json
from hons.types import QuestionKey

def run(config: Config):

    #Store current config in json file for reference
    file_path = Path(f'runs/{config.run.runtag}')
    with open(file_path,'w') as f:
        json.dump(config.raw,f)

    # One topic per line: tid, label, narrative, and questions (qid, question, rel_docs)
    with open(config.run.topics) as f:
        topics = [json.loads(line) for line in f if line.strip()]

    examples = {}
    for topic in topics:
        for question in topic['questions']:
            for doc_id in question['rel_docs']:
                if examples[doc_id] not in examples:
                    examples[doc_id] = [QuestionKey(topic['tid'],question['qid'])]
                else:
                    examples[doc_id].append(QuestionKey(topic['tid'],question['qid']))


    # Every stage is built before day 0, so an unknown impl or bad parameter fails here
    stages = build_stages(config)



    

    
