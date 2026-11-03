from hons.config import Config
from hons.registry import build_stages
from pathlib import Path
import json
from hons.types import QuestionKey

def run(config: Config):

    # Store the config in the run directory for reference. The digest in the name ties results to the exact toggles
    run_dir = Path('runs') / f'{config.run.runtag}-{config.digest()}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.json','w') as f:
        json.dump(config.raw,f,indent=2,default=str)

    # One topic per line: tid, label, narrative, and questions (qid, question, rel_docs)
    with open(config.run.topics) as f:
        topics = [json.loads(line) for line in f if line.strip()]

    examples = {}
    for topic in topics:
        for question in topic['questions']:
            for doc_id in question['rel_docs']:
                examples.setdefault(doc_id, []).append(QuestionKey(topic['tid'],question['qid']))


    # Every stage is built before day 0, so an unknown impl or bad parameter fails here
    stages = build_stages(config)



    

    
