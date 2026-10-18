# evaluation/sanity_faithfulness.py

"""Can the faithfulness judge detect an unsupported claim?  Expected: ~1.0 then well below 1.
    python -m evaluation.sanity_faithfulness
"""

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness

from evaluation.score_ragas import build_judge

context = ["The model reaches an accuracy of 97.0% for the motor, 95.4% for the gearbox, "
           "99.8% for the left axle box, and 99.97% for the right axle box."]
question = "Quelle précision est obtenue sur le gearbox ?"

samples = [
    SingleTurnSample(user_input=question, retrieved_contexts=context,
                     response="La précision sur le gearbox est de 95,4 %."),
    SingleTurnSample(user_input=question, retrieved_contexts=context,
                     response="La précision sur le gearbox est de 95,4 %, obtenue avec un "
                              "réseau de 12 couches entraîné pendant 3 jours."),
]

result = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=[Faithfulness()],
    llm=build_judge(),
    run_config=RunConfig(max_workers=1, timeout=300),
)
print(result.to_pandas()[["response", "faithfulness"]].to_string())