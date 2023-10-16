from zenml import pipelines

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.

@pipelines()
def training_pipeline(data_path: str):
